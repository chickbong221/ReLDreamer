import math

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from .graph_encoder import GraphPosterior

f32 = jnp.float32
sg = jax.lax.stop_gradient


class RSSM(nj.Module):
  """RSSM with a semantic stochastic state between h_t and z_t.

  The sequence model advances h_t from the previous recurrent, semantic and
  low-level states; the semantic posterior reads the scene graph while the
  semantic prior predicts the same state from temporal context alone. The
  low-level prior and posterior are both conditioned on the semantic state, so
  imagination never needs a graph.
  """

  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  semstoch: int = 16
  semclasses: int = 16
  semlayers: int = 1
  semantic: bool = True
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
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
    self.act_space = act_space
    self.kw = kw
    self.graph_kw = dict(
        **(graph_kw or {}), act=self.act, norm=self.norm, **kw)

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        sem=elements.Space(np.float32, (self.semstoch, self.semclasses)),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))

  def initial(self, bsize):
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        sem=jnp.zeros([bsize, self.semstoch, self.semclasses], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32)))
    return carry

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe(self, carry, tokens, graph, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat, nodes) = self._observe(
          carry, tokens, graph, action, reset, training)
      return carry, entry, feat, nodes
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      carry, (entries, feat, nodes) = nj.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (tokens, graph, action, reset), unroll=unroll, axis=1)
      return carry, entries, feat, nodes

  def _observe(self, carry, tokens, graph, action, reset, training):
    deter, sem, stoch, action = nn.mask(
        (carry['deter'], carry['sem'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    deter = self._core(deter, sem, stoch, action)

    if self.semantic:
      # Built through sub() so the posterior's params live under dyn/, which is
      # what policy_keys ships to the policy worker.
      enc = self.sub('graphenc', GraphPosterior, **self.graph_kw)
      nodes, token = enc(graph, deter)
      semlogit = self._semhead(
          'semobs', jnp.concatenate([deter, self._flat(sem), token], -1))
      sem = nn.cast(self._dist(semlogit).sample(seed=nj.seed()))
    else:
      nodes = jnp.zeros((deter.shape[0], 0, 0), deter.dtype)
      semlogit = jnp.zeros_like(sem)

    tokens = tokens.reshape((*deter.shape[:-1], -1))
    x = jnp.concatenate([self._flat(sem), tokens], -1)
    if not self.absolute:
      x = jnp.concatenate([deter, x], -1)
    for i in range(self.obslayers):
      x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
    logit = self._logit('obslogit', x)
    stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, sem=sem, stoch=stoch)
    feat = dict(
        deter=deter, sem=sem, stoch=stoch, logit=logit, semlogit=semlogit)
    entry = dict(deter=deter, sem=sem, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, sem, stoch, logit))
    return carry, (entry, feat, nodes)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      deter = self._core(
          carry['deter'], carry['sem'], carry['stoch'], actemb)
      if self.semantic:
        semlogit = self._sem_prior(deter, carry['sem'])
        sem = nn.cast(self._dist(semlogit).sample(seed=nj.seed()))
      else:
        sem = semlogit = carry['sem']
      logit = self._prior(deter, sem)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      carry = nn.cast(dict(deter=deter, sem=sem, stoch=stoch))
      feat = nn.cast(dict(
          deter=deter, sem=sem, stoch=stoch,
          logit=logit, semlogit=semlogit))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, sem, stoch))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
      # We can also return all carry entries but it might be expensive.
      # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
      # return carry, entries, feat, action
      return carry, feat, action

  def loss(self, carry, tokens, graph, acts, reset, training, step_valid=None):
    metrics = {}
    prev_sem = nn.cast(carry['sem'])
    carry, entries, feat, nodes = self.observe(
        carry, tokens, graph, acts, reset, training)
    prior = self._prior(feat['deter'], feat['sem'])
    post = feat['logit']
    losses = {
        'dyn': self._dist(sg(post)).kl(self._dist(prior)),
        'rep': self._dist(post).kl(self._dist(sg(prior))),
    }
    metrics['dyn_ent'] = self._dist(prior).entropy().mean()
    metrics['rep_ent'] = self._dist(post).entropy().mean()
    semkeys = ()
    if self.semantic:
      semprior = self._sem_prior(
          feat['deter'], self._shift(feat['sem'], prev_sem, reset))
      sempost = feat['semlogit']
      losses['semdyn'] = self._dist(sg(sempost)).kl(self._dist(semprior))
      losses['semrep'] = self._dist(sempost).kl(self._dist(sg(semprior)))
      metrics['sem_ent'] = self._dist(sempost).entropy().mean()
      semkeys = ('semdyn', 'semrep')
      # Unclipped, so the semantic KL stays observable below the free-nats
      # floor where the optimised loss is flat. Reduced in f32: this is read
      # against that floor, and bf16 cannot sum a batch without drifting.
      valid = (
          jnp.ones(losses['semdyn'].shape, f32) if step_valid is None
          else step_valid.astype(f32))
      for key in semkeys:
        metrics[f'{key}_raw'] = (
            (losses[key].astype(f32) * valid).sum()
            / jnp.maximum(valid.sum(), 1.0))
    if self.free_nats:
      losses = {k: jnp.maximum(v, self.free_nats) for k, v in losses.items()}
    if step_valid is not None:
      # After clipping: masking first would lift the zeroed terminal entries
      # back to the free-nats floor.
      for key in semkeys:
        losses[key] = losses[key] * nn.cast(step_valid, force=True)
    return carry, entries, losses, feat, nodes, metrics

  def _shift(self, sem, prev, reset):
    """g_{t-1} per timestep: the chunk's incoming carry, then sem shifted right,
    zeroed wherever the episode restarts."""
    shifted = jnp.concatenate([prev[:, None], sem[:, :-1]], 1)
    return nn.mask(shifted, ~reset)

  def _flat(self, x):
    return x.reshape((*x.shape[:-2], -1))

  def _core(self, deter, sem, stoch, action):
    stoch = stoch.reshape((stoch.shape[0], -1))
    sem = sem.reshape((sem.shape[0], -1))
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
    x3 = self.sub('dynin3', nn.Linear, self.hidden, **self.kw)(sem)
    x3 = nn.act(self.act)(self.sub('dynin3norm', nn.Norm, self.norm)(x3))
    x = jnp.concatenate([x0, x1, x2, x3], -1)[..., None, :].repeat(g, -2)
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
    deter = update * cand + (1 - update) * deter
    return deter

  def _prior(self, deter, sem):
    x = jnp.concatenate([deter, self._flat(sem)], -1)
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _sem_prior(self, deter, sem):
    return self._semhead(
        'semprior', jnp.concatenate([deter, self._flat(sem)], -1))

  def _semhead(self, name, x):
    for i in range(self.semlayers):
      x = self.sub(f'{name}{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'{name}{i}norm', nn.Norm, self.norm)(x))
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(
        f'{name}logit', nn.Linear, self.semstoch * self.semclasses, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.semstoch, self.semclasses))

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    x = self.sub(name, nn.Linear, self.stoch * self.classes, **kw)(x)
    return x.reshape(x.shape[:-1] + (self.stoch, self.classes))

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out


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
    entries = {}
    return carry, entries, tokens


class Decoder(nj.Module):

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
    inp = [nn.cast(feat[k]) for k in ('stoch', 'sem', 'deter')]
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
        x0, x1, x2 = nn.cast((feat['deter'], feat['stoch'], feat['sem']))
        x1 = jnp.concatenate([
            x1.reshape((*x1.shape[:-2], -1)),
            x2.reshape((*x2.shape[:-2], -1))], -1)
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
        out = embodied.jax.outs.Agg(out, 3, jnp.sum)
        recons[k] = out

    entries = {}
    return carry, entries, recons
