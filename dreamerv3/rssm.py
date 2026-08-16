import math

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from .graph_encoder import GraphEncoder, onehot_embed

f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient


def align_slots(obs_uid, slot_uid, slot_mask):
  """Packed vertex -> recurrent slot, as a ``[B, N, S]`` boolean map.

  Matching is uid equality against the slots already occupied. Slot 0 is the end
  effector, which the packer always seats at vertex 0. Remaining new uids take
  free object slots in packed order, so an occupied slot never moves and a uid
  that reappears later rejoins the slot it left. Returns the map, which slots
  carry an observation this step, and which were admitted by this step.
  """
  N, S = obs_uid.shape[1], slot_uid.shape[1]
  live = obs_uid > 0
  occupied = slot_mask > 0
  match = (
      (obs_uid[:, :, None] == slot_uid[:, None, :]) &
      live[:, :, None] & occupied[:, None, :])
  seen = match.any(-1)

  is_ee = (jnp.arange(N) == 0)[None, :]
  ee_slot = (jnp.arange(S) == 0)[None, :]
  ee = (live & is_ee & ~seen)[:, :, None] & ee_slot[:, None, :]

  new = live & ~seen & ~is_ee
  free = (~occupied) & ~ee_slot
  rank = jnp.cumsum(new, 1, dtype=i32) - new
  hole = jnp.cumsum(free, 1, dtype=i32) - free
  admit = (
      new[:, :, None] & free[:, None, :] &
      (rank[:, :, None] == hole[:, None, :]))

  align = match | admit | ee
  return align, align.any(1), (admit | ee).any(1)


class RSSM(nj.Module):
  """Plain Dreamer RSSM, optionally extended with semantic slots.

  With ``semantic=False``, this constructs the original DreamerV3 state and
  parameter shapes. With ``semantic=True``, the state additionally carries one
  slot per scene-graph vertex. Slots are matched to observed vertices by uid,
  advance under a shared transition, and reach the low-level dynamics only
  through the deterministic state.
  """

  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  slots: int = 6
  slot_dim: int = 256
  slot_heads: int = 4
  semantic: bool = True
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: int = 1
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0

  def __init__(self, act_space, graph_kw=None, **kw):
    assert self.deter % self.blocks == 0
    assert self.slot_dim % self.slot_heads == 0, (
        self.slot_dim, self.slot_heads)
    self.act_space = act_space
    self.kw = kw
    graph_kw = dict(graph_kw or {})
    self.entity_vocab = int(graph_kw.get('entity_vocab', 64))
    self.embed = int(graph_kw.get('embed', 64))
    self.graph_kw = dict(
        **graph_kw, slot_dim=self.slot_dim, act=self.act, norm=self.norm, **kw)

  @property
  def entry_space(self):
    spaces = dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))
    if self.semantic:
      spaces.update(
          slots=elements.Space(np.float32, (self.slots, self.slot_dim)),
          slot_uid=elements.Space(np.uint16, self.slots),
          slot_ent=elements.Space(np.uint16, self.slots),
          slot_mask=elements.Space(bool, self.slots),
          target_mask=elements.Space(bool, self.slots))
    return spaces

  def initial(self, bsize):
    state = dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32))
    if self.semantic:
      state.update(
          slots=jnp.zeros([bsize, self.slots, self.slot_dim], f32),
          slot_uid=jnp.zeros([bsize, self.slots], jnp.uint16),
          slot_ent=jnp.zeros([bsize, self.slots], jnp.uint16),
          slot_mask=jnp.zeros([bsize, self.slots], bool),
          target_mask=jnp.zeros([bsize, self.slots], bool))
    return nn.cast(state)

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    return jax.tree.map(lambda x: x[:, -1], entries)

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  # ------------------------------------------------------------------ graph

  def encode_graph(self, graph, single=False):
    """Node embeddings for every timestep at once.

    The encoder reads no recurrent state, so it runs outside the scan exactly
    like the image encoder: one pass over B*T instead of batch_length
    sequential launches of batch_size work. Built through sub() so its params
    live under dyn/, which is what policy_keys ships to the policy worker.
    """
    enc = self.sub('graphenc', GraphEncoder, **self.graph_kw)
    if single:
      return enc(graph)
    B, T = jax.tree.leaves(graph)[0].shape[:2]
    flat = jax.tree.map(lambda x: x.reshape((B * T, *x.shape[2:])), graph)
    nodes = enc(flat)
    return nodes.reshape((B, T, *nodes.shape[1:]))

  # ------------------------------------------------------------------ observe

  def observe(self, carry, tokens, graph, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if not self.semantic:
      step = lambda c, inputs: self._observe_plain(c, *inputs, training)
      if single:
        carry, (entry, feat) = step(carry, (tokens, action, reset))
        return carry, entry, feat, {}
      unroll = max(1, min(self.unroll, jax.tree.leaves(tokens)[0].shape[1]))
      carry, (entries, feat) = nj.scan(
          step, carry, (tokens, action, reset), unroll=unroll, axis=1)
      return carry, entries, feat, {}

    nodes = self.encode_graph(graph, single=single)
    obs = (nodes, graph['graph_node_uid'], graph['graph_node_ent'],
           graph['graph_node_target'])
    step = lambda c, inputs: self._observe_slots(c, *inputs, training)
    if single:
      carry, (entry, feat, aux) = step(carry, (tokens, *obs, action, reset))
      return carry, entry, feat, aux
    unroll = max(1, min(self.unroll, jax.tree.leaves(tokens)[0].shape[1]))
    carry, (entries, feat, aux) = nj.scan(
        step, carry, (tokens, *obs, action, reset), unroll=unroll, axis=1)
    return carry, entries, feat, aux

  def _observe_plain(self, carry, tokens, action, reset, training):
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    deter = self._core(deter, stoch, action)
    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def _observe_slots(
      self, carry, tokens, nodes, obs_uid, obs_ent, obs_tgt, action, reset,
      training):
    deter, stoch, slots = nn.mask(
        (carry['deter'], carry['stoch'], carry['slots']), ~reset)
    slot_uid, slot_ent, slot_mask, target_mask = nn.mask(
        (carry['slot_uid'], carry['slot_ent'], carry['slot_mask'],
         carry['target_mask']), ~reset)
    action = nn.mask(action, ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)

    ctx = self._slot_context(slots, slot_mask, target_mask)
    deter = self._core(deter, stoch, action, ctx)
    prior = self._slot_transition(
        slots, slot_mask, target_mask, slot_ent, deter, stoch, action)

    align, matched, fresh = align_slots(obs_uid, slot_uid, slot_mask)
    take = align.astype(i32)
    picked = lambda v: (take * v[:, :, None]).sum(1)
    slot_uid = jnp.where(fresh, picked(obs_uid), slot_uid).astype(jnp.uint16)
    slot_ent = jnp.where(fresh, picked(obs_ent), slot_ent).astype(jnp.uint16)
    slot_mask = slot_mask | fresh
    target_mask = jnp.where(matched, picked(obs_tgt) > 0, target_mask)

    # Direct replacement, not a learned fusion gate: the observed embedding is
    # the semantic target the prior is trained against, so making the posterior
    # anything else would blur what the slot loss means. Slots whose uid is not
    # in this frame's graph keep the prediction instead of resetting.
    aligned = jnp.einsum(
        'bns,bnu->bsu', nn.cast(align, force=True), nodes, optimize='optimal')
    keep = nn.cast(slot_mask, force=True)[..., None]
    slots = jnp.where(matched[..., None], aligned, prior) * keep

    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

    state = dict(
        deter=deter, stoch=stoch, slots=slots, slot_uid=slot_uid,
        slot_ent=slot_ent, slot_mask=slot_mask, target_mask=target_mask)
    feat = dict(**state, logit=logit)
    aux = dict(prior=prior, align=align, matched=matched, fresh=fresh)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, slots))
    return state, (dict(state), feat, aux)

  def _slot_context(self, slots, slot_mask, target_mask):
    """One compact vector summarising the slot set for the ``h`` transition."""
    m = nn.cast(slot_mask, force=True)
    x = self.sub('slotctxnorm', nn.Norm, self.norm)(slots) * m[..., None]
    x = jnp.concatenate(
        [x.reshape((x.shape[0], -1)), m, nn.cast(target_mask, force=True)], -1)
    return self.sub('slotctx', nn.Linear, self.slot_dim, **self.kw)(x)

  def _slot_transition(
      self, slots, slot_mask, target_mask, slot_ent, deter, stoch, action):
    """One shared predictor over all slots, after one cross-slot exchange.

    The attention block keeps every slot's own output; it never pools them. The
    same parameters run on all six, so a slot's behaviour comes from its content
    and its role embeddings rather than from its index.
    """
    B, S, D = slots.shape
    dim = D // self.slot_heads
    m = nn.cast(slot_mask, force=True)[..., None]
    h = self.sub('slotnorm', nn.Norm, self.norm)(slots) * m
    qkv = self.sub(
        'slotqkv', nn.Linear, (3, self.slot_heads, dim), **self.kw)(h)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    logits = jnp.einsum(
        'bihd,bjhd->bijh', q, k, optimize='optimal') / math.sqrt(dim)
    live = m[:, None, :, :]
    attn = jax.nn.softmax(jnp.where(live > 0, logits, -1e9), 2)
    inter = jnp.einsum(
        'bijh,bjhd->bihd', attn, v, optimize='optimal').reshape((B, S, D))
    inter = self.sub('slotattnout', nn.Linear, D, **self.kw)(inter) * m

    glob = jnp.concatenate([
        self.sub('slothctx', nn.Linear, D, **self.kw)(deter),
        self.sub('slotzctx', nn.Linear, D, **self.kw)(
            stoch.reshape((B, -1))),
        self.sub('slotactx', nn.Linear, self.embed, **self.kw)(action),
    ], -1)
    x = jnp.concatenate([
        h, inter, jnp.repeat(glob[:, None], S, 1),
        onehot_embed(
            self, 'slotent', slot_ent, self.entity_vocab, self.embed,
            slots.dtype),
        onehot_embed(
            self, 'slottgt', target_mask.astype(i32), 2, self.embed,
            slots.dtype),
    ], -1)
    x = self.sub('slotmlp', nn.Linear, 2 * D, **self.kw)(x)
    x = nn.act(self.act)(self.sub('slotmlpnorm', nn.Norm, self.norm)(x))
    x = self.sub('slotout', nn.Linear, D, **self.kw)(x)
    return self.sub('slotpost', nn.Norm, self.norm)(slots + x) * m

  # ------------------------------------------------------------------ imagine

  def imagine(self, carry, policy, length, training, single=False):
    if not single:
      unroll = max(1, min(self.unroll, length))
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
      return carry, feat, action

    action = policy(sg(carry)) if callable(policy) else policy
    actemb = nn.DictConcat(self.act_space, 1)(action)
    if not self.semantic:
      deter = self._core(carry['deter'], carry['stoch'], actemb)
      logit = self._prior(deter)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      carry = nn.cast(dict(deter=deter, stoch=stoch))
      feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
      return carry, (feat, action)

    # Identity, occupancy and role are latched: imagination never sees a graph,
    # so nothing may admit, evict or rename a slot inside the rollout.
    mask, tgt = carry['slot_mask'], carry['target_mask']
    ctx = self._slot_context(carry['slots'], mask, tgt)
    deter = self._core(carry['deter'], carry['stoch'], actemb, ctx)
    logit = self._prior(deter)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    slots = self._slot_transition(
        carry['slots'], mask, tgt, carry['slot_ent'], deter, carry['stoch'],
        actemb)
    state = dict(
        deter=deter, stoch=stoch, slots=slots, slot_uid=carry['slot_uid'],
        slot_ent=carry['slot_ent'], slot_mask=mask, target_mask=tgt)
    return nn.cast(state), (nn.cast(dict(**state, logit=logit)), action)

  # ------------------------------------------------------------------ loss

  def loss(self, carry, tokens, graph, acts, reset, training, step_valid=None):
    metrics = {}
    carry, entries, feat, aux = self.observe(
        carry, tokens, graph, acts, reset, training)
    prior = self._prior(feat['deter'])
    post = feat['logit']
    losses = {
        'dyn': self._dist(sg(post)).kl(self._dist(prior)),
        'rep': self._dist(post).kl(self._dist(sg(prior))),
    }
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    if self.free_nats:
      losses = {k: jnp.maximum(v, self.free_nats) for k, v in losses.items()}

    if self.semantic:
      losses['slot'], mets = self._slot_loss(feat, aux, step_valid)
      metrics.update(mets)
    return carry, entries, losses, feat, aux, metrics

  def _slot_loss(self, feat, aux, step_valid):
    """One-step slot prediction against the observed embedding.

    The target is stop-gradient: without it the encoder could move the target to
    wherever the transition already points, which the relation heads would never
    notice. A slot is supervised only when this frame's graph actually carried
    its uid, and never on the frame it was admitted, whose previous slot was
    zero and whose prior therefore predicts nothing.
    """
    target = sg(feat['slots'].astype(f32))
    pred = aux['prior'].astype(f32)
    delta = pred - target
    huber = jnp.where(
        jnp.abs(delta) < 1.0, 0.5 * delta ** 2, jnp.abs(delta) - 0.5).mean(-1)
    num = (pred * target).sum(-1)
    den = (jnp.sqrt(jnp.maximum((pred * pred).sum(-1), 1e-12)) *
           jnp.sqrt(jnp.maximum((target * target).sum(-1), 1e-12)))
    cos = num / den
    per_slot = huber + 0.25 * (1.0 - cos)

    mask = (
        feat['slot_mask'] & aux['matched'] & ~aux['fresh']).astype(f32)
    if step_valid is not None:
      mask = mask * step_valid.astype(f32)[..., None]
    total = mask.sum(-1)
    loss = (per_slot * mask).sum(-1) / jnp.maximum(total, 1.0)
    metrics = dict(
        slot_cos=(cos * mask).sum() / jnp.maximum(mask.sum(), 1.0),
        slot_supervised=total.mean(),
        slot_occupancy=feat['slot_mask'].astype(f32).sum(-1).mean())
    return loss, metrics

  # ------------------------------------------------------------------ core

  def _core(self, deter, stoch, action, slots=None):
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))
    inputs = [x0, x1, x2]
    if slots is not None:
      x3 = self.sub('dynin3', nn.Linear, self.hidden, **self.kw)(slots)
      x3 = nn.act(self.act)(self.sub('dynin3norm', nn.Norm, self.norm)(x3))
      inputs.append(x3)
    x = jnp.concatenate(inputs, -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    return update * cand + (1 - update) * deter

  def _prior(self, deter):
    x = deter
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    return embodied.jax.outs.Agg(out, 1, jnp.sum)


class Encoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, obs, reset, training, single=False):
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape

    if self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      outs.append(x)

    if self.imgkeys:
      K = self.kernel
      # Normalise each image to [0, 1] before concatenation so RGB (uint8,
      # max 255) and depth (uint16, max self.obs_space[k].high in mm) can
      # coexist. The final `- 0.5` shifts the input to roughly zero mean.
      norm_imgs = []
      for k in sorted(self.imgkeys):
        v = obs[k]
        if v.dtype == jnp.uint8:
          scale = 255.0
        elif v.dtype == jnp.uint16:
          scale = float(np.asarray(self.obs_space[k].high).max())
        else:
          scale = 1.0
        norm_imgs.append(nn.cast(v, force=True) / scale)
      x = jnp.concatenate(norm_imgs, -1) - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    tokens = x.reshape((*bshape, *x.shape[1:]))
    return carry, {}, tokens


class Decoder(nj.Module):
  """Pixel and vector reconstruction from ``deter`` and ``stoch`` only.

  Slots are deliberately excluded: reconstruction gradients are the largest in
  the model, and routing them through the semantic state would train the slots
  to carry appearance rather than relations.
  """

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  bspace: int = 8
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
    self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, feat, reset, training, single=False):
    assert feat['deter'].shape[-1] % self.bspace == 0
    K = self.kernel
    recons = {}
    bshape = reset.shape
    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
    inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
    inp = jnp.concatenate(inp, -1)

    if self.veckeys:
      spaces = {k: self.obs_space[k] for k in self.veckeys}
      o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
      outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
      x = x.reshape((*bshape, *x.shape[1:]))
      kw = dict(**self.kw, outscale=self.outscale)
      outs = self.sub('vec', embodied.jax.DictHead, spaces, outputs, **kw)(x)
      recons.update(outs)

    if self.imgkeys:
      factor = 2 ** (len(self.depths) - int(bool(self.outer)))
      minres = [int(x // factor) for x in self.imgres]
      assert 3 <= minres[0] <= 16, minres
      assert 3 <= minres[1] <= 16, minres
      shape = (*minres, self.depths[-1])
      if self.bspace:
        u, g = math.prod(shape), self.bspace
        x0, x1 = nn.cast((feat['deter'], feat['stoch']))
        x1 = x1.reshape((*x1.shape[:-2], -1))
        x0 = x0.reshape((-1, x0.shape[-1]))
        x1 = x1.reshape((-1, x1.shape[-1]))
        x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0)
        x0 = einops.rearrange(
            x0, '... (g h w c) -> ... h w (g c)',
            h=minres[0], w=minres[1], g=g)
        x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
        x1 = nn.act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
        x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
        x = nn.act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
      else:
        x = self.sub('space', nn.Linear, shape, **kw)(inp)
        x = nn.act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))
      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        if self.strided:
          kw = dict(**self.kw, transp=True)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)
        else:
          x = x.repeat(2, -2).repeat(2, -3)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))
      if self.outer:
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      elif self.strided:
        kw = dict(**self.kw, outscale=self.outscale, transp=True)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)
      else:
        x = x.repeat(2, -2).repeat(2, -3)
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      x = jax.nn.sigmoid(x)
      x = x.reshape((*bshape, *x.shape[1:]))
      split = np.cumsum(
          [self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        out = embodied.jax.outs.MSE(out)
        recons[k] = embodied.jax.outs.Agg(out, 3, jnp.sum)

    return carry, {}, recons
