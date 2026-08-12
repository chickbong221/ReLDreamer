"""Semantic posterior and graph decoder over the maintained scene graph.

The posterior embeds each vertex and each qualified fact, folds the facts into
a dense pair embedding, runs L EGT-Simple attention layers over the vertices,
and attention-pools the result into one fixed-width graph token; the semantic
state is sampled from that token inside the RSSM. The decoder reconstructs
appearance, the bounding box, visibility, and both relation states from the
posterior node representations, so the semantic loss grounds the encoder trunk
the pooling reads from.

EGT-Simple means the pair embedding is built once and every layer reads that
same tensor: facts bias and gate attention but carry no state of their own, so
there is no edge channel to update and no per-layer edge FFN.

A vertex is ``[AppProj_c(a_c), BBoxProj_c(b_c), ..., EntityEmbed(id),
TargetEmbed(flag)]`` over cameras ``c``. Appearance arrives on
``graph_node_app`` as frozen DINO features read straight from replay; nothing
here produces it.

Node count and fact count never reach the semantic state's shape: pooling is
masked and permutation invariant, so padding width is irrelevant and only the
valid content moves the token.
"""

import math

import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS
from scenegraph.adapters.graph_vocab import (
    build_absolute_vocab,
    build_relation_vocab,
    build_temporal_vocab,
)
from scenegraph.core.relation_rules import (
    ABS_LABELS,
    RELATION_TYPES,
    TEMPORAL_RELATIONS,
)

f32 = jnp.float32
i32 = jnp.int32
sg = jax.lax.stop_gradient


def relation_tables():
    """Static relation vocabulary sizes plus the decoder's per-rho masks."""
    rel = build_relation_vocab()
    abs_ = build_absolute_vocab()
    tmp = build_temporal_vocab()
    abs_valid = np.zeros((len(rel), len(abs_)), np.float32)
    temp_valid = np.zeros((len(rel),), np.float32)
    for name in RELATION_TYPES:
        r = rel.encode(name)
        for label in ABS_LABELS[name]:
            abs_valid[r, abs_.encode(label)] = 1.0
        temp_valid[r] = float(name in TEMPORAL_RELATIONS)
    return dict(
        n_rel=len(rel), n_abs=len(abs_), n_temp=len(tmp),
        abs_valid=abs_valid, temp_valid=temp_valid,
    )


SCAN_KEYS = GRAPH_KEYS
_FLOAT_KEYS = ('graph_node_app', 'graph_node_bbox')


def unpack(graph):
    """Cast the narrow packed dtypes back to what the embeddings expect."""
    return {
        k: graph[k].astype(f32 if k in _FLOAT_KEYS else i32)
        for k in SCAN_KEYS
    }


def derive_masks(g):
    """Validity, per-camera visibility and appearance support, in float32.

    Read off the packed content rather than stored alongside it. Computed
    before any compute-dtype cast on purpose: one pixel is 1/112 of the frame,
    which bf16 resolves near the frame edge with under 3x margin.
    """
    bbox = g['graph_node_bbox'].astype(f32)
    app = g['graph_node_app'].astype(f32)
    return dict(
        valid=(g['graph_node_ent'] != 0).astype(f32),
        camera_visible=(
            (bbox[..., 1] > bbox[..., 0]) &
            (bbox[..., 3] > bbox[..., 2])).astype(f32),
        appearance_known=(jnp.abs(app).sum(-1) > 0).astype(f32),
        edge_valid=(g['graph_edge_rel'] != 0).astype(f32),
        temp_mask=(g['graph_edge_temp'] != 0).astype(f32),
    )


def _edge_selectors(src, dst, mask, num_nodes, dtype):
    """Masked dense incidence matrices for a small fixed-capacity graph.

    The packed edge arrays are padded to a static width for JIT compilation.
    Advanced gathers still execute those padded slots and their reverse-mode
    transposes, with every padding index colliding at node zero. With only
    ``num_nodes`` <= 10 nodes, dense one-hot contractions are both small and
    much friendlier to the GPU. Padding rows are exactly zero.
    """
    weight = mask.astype(dtype)[..., None]
    source = jax.nn.one_hot(src, num_nodes, dtype=dtype) * weight
    destination = jax.nn.one_hot(dst, num_nodes, dtype=dtype) * weight
    return source, destination


def _select_nodes(selector, nodes):
    """Select one node per edge without an indexed gather."""
    return jnp.einsum('ben,bnu->beu', selector, nodes, optimize='optimal')


def _embed(module, name, index, classes, units, dtype):
    """Embedding lookup as a one-hot matmul, for a table indexed per fact.

    A gather transposes to a scatter-add. These tables hold a handful of rows
    against ``B * e_max`` indices -- padding included, since every padded slot
    still reads row zero -- so the whole batch collides on a few addresses and
    the backward serialises on atomics. Measured at ``e_max: 270``, that is
    90% of the encoder's backward. The dense form is one small matmul each way.

    ``winit`` is pinned to the output fan so the table initialises exactly as
    ``nn.Embed`` does; the layer only changes how the row is read.
    """
    return module.sub(
        name, nn.Linear, units, bias=False, winit='trunc_normal_out')(
        jax.nn.one_hot(index, classes, dtype=dtype))


class GraphPosterior(nj.Module):

    layers: int = 2
    units: int = 256
    heads: int = 4
    embed: int = 64
    edge_units: int = 64
    clip_logits: float = 5.0
    app: int = 64
    bbox: int = 8
    entity_vocab: int = 64
    norm: str = 'rms'
    act: str = 'gelu'

    def __init__(self, **kw):
        assert self.units % self.heads == 0, (self.units, self.heads)
        self.kw = kw
        self.tables = relation_tables()

    def __call__(self, graph):
        """``graph`` holds (B, ...) arrays; returns nodes (B, N, U) and the
        pooled token (B, U).

        Nothing here reads the recurrent state, so the caller batches every
        timestep into B and runs this once, the way the image encoder does.
        Recurrent conditioning happens downstream, where the semantic head
        sees the token alongside ``deter`` and the previous semantic state.
        """
        g = graph
        m = derive_masks(g)
        ent = g['graph_node_ent']
        app = nn.cast(g['graph_node_app'], force=True)
        bbox = nn.cast(g['graph_node_bbox'], force=True)
        valid = nn.cast(m['valid'], force=True)
        known = nn.cast(m['appearance_known'], force=True)
        seen = nn.cast(m['camera_visible'], force=True)
        src, dst = g['graph_edge_src'], g['graph_edge_dst']
        rel, sig, tau = (
            g['graph_edge_rel'], g['graph_edge_abs'], g['graph_edge_temp'])
        tmask = nn.cast(m['temp_mask'], force=True)
        emask = nn.cast(m['edge_valid'], force=True)
        N = ent.shape[1]
        C = app.shape[-2]

        # Gated after projection, not before: the projections carry a bias, so
        # a zeroed input would still emit a constant that reads as content.
        parts = []
        for cam in range(C):
            a = self.sub(
                f'app{cam}', nn.Linear, self.app, **self.kw)(app[..., cam, :])
            parts.append(a * known[..., cam, None])
            b = self.sub(
                f'bbox{cam}', nn.Linear, self.bbox, **self.kw)(bbox[..., cam, :])
            parts.append(b * seen[..., cam, None])
        parts.append(self.sub('ent', nn.Embed, self.entity_vocab, self.embed)(ent))
        # Which vertex the subtask is acting on. Two tokens rather than a
        # widened entity vocabulary: a category means the same thing whether or
        # not it is this episode's goal, so the appearance statistics behind
        # its embedding should not be split in half.
        parts.append(self.sub('tgt', nn.Embed, 2, self.embed)(
            g['graph_node_target']))
        x = self._mlp('node', jnp.concatenate(parts, -1)) * valid[..., None]

        pair = self._pairs(rel, sig, tau, src, dst, emask, tmask, N, x.dtype)
        for i in range(self.layers):
            x = self._layer(i, x, pair, valid)
        # Pre-norm leaves the residual stream unnormalised, and the pooling and
        # the decoder heads both read x directly.
        x = self.sub('final', nn.Norm, self.norm)(x) * valid[..., None]

        return x, self._pool(x, valid)

    def _pairs(self, rel, sig, tau, src, dst, emask, tmask, N, dtype):
        """One embedding per ordered vertex pair, built once for every layer.

        Facts land in their ordered slot through a single masked one-hot
        contraction; a pair carrying several facts sums them. Each slot then
        reads its own facts next to the opposite direction's, so direction is
        explicit without duplicating the edge axis, and the presence flags
        separate a pair with no fact from one whose facts embed near zero.
        """
        B = rel.shape[0]
        fact = self.sub('fact', nn.Linear, self.edge_units, **self.kw)(
            jnp.concatenate([
                _embed(self, 'rel', rel, self.tables['n_rel'],
                       self.embed, dtype),
                _embed(self, 'abs', sig, self.tables['n_abs'],
                       self.embed, dtype),
                tmask[..., None] * _embed(
                    self, 'temp', tau, self.tables['n_temp'],
                    self.embed, dtype),
            ], -1))
        # The slot weights already zero the padding rows, so the projection
        # bias a padded fact carries never reaches a pair.
        slot = jax.nn.one_hot(
            src * N + dst, N * N, dtype=dtype) * emask[..., None]
        dense = jnp.einsum('bep,bec->bpc', slot, fact, optimize='optimal')
        forward = dense.reshape((B, N, N, self.edge_units))
        present = (slot.sum(1) > 0).astype(dtype).reshape((B, N, N, 1))
        return self._mlp('pair', jnp.concatenate([
            forward, forward.swapaxes(1, 2),
            present, present.swapaxes(1, 2)], -1), self.edge_units)

    def _layer(self, i, x, pair, valid):
        """One EGT-Simple block: pair-biased gated attention, then an FFN.

        Attention is global over the vertex set, so every pair of vertices is
        one hop apart and the pair embedding is what makes an actual fact
        different from the absence of one. The QK logits are clipped before the
        bias is added, exactly as in EGT: the bias is the term meant to carry
        structure, and an unclipped dot product can saturate the softmax before
        it ever gets read.
        """
        B, N, _ = x.shape
        dim = self.units // self.heads
        h = self.sub(f'attnnorm{i}', nn.Norm, self.norm)(x)
        qkv = self.sub(
            f'qkv{i}', nn.Linear, (3, self.heads, dim), **self.kw)(h)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        logits = jnp.clip(
            jnp.einsum('bihd,bjhd->bijh', q, k, optimize='optimal') /
            math.sqrt(dim), -self.clip_logits, self.clip_logits)
        logits += self.sub(f'bias{i}', nn.Linear, self.heads, **self.kw)(pair)
        # Padding vertices are dropped as keys twice over: out of the softmax,
        # where they would otherwise take mass, and out of the gate, which is
        # not normalised and would otherwise pass them through.
        live = valid[:, None, :, None]
        attn = jax.nn.softmax(jnp.where(live > 0, logits, -1e9), 2)
        attn = attn * jax.nn.sigmoid(
            self.sub(f'gate{i}', nn.Linear, self.heads, **self.kw)(pair)) * live
        att = jnp.einsum('bijh,bjhd->bihd', attn, v, optimize='optimal')
        x += self.sub(f'attnout{i}', nn.Linear, self.units, **self.kw)(
            att.reshape((B, N, self.units)))

        h = self.sub(f'ffnnorm{i}', nn.Norm, self.norm)(x)
        h = nn.act(self.act)(
            self.sub(f'ffn{i}', nn.Linear, 2 * self.units, **self.kw)(h))
        x += self.sub(f'ffnout{i}', nn.Linear, self.units, **self.kw)(h)
        return x * valid[..., None]

    def _mlp(self, name, x, units=None):
        x = self.sub(name, nn.Linear, units or self.units, **self.kw)(x)
        return nn.act(self.act)(self.sub(f'{name}norm', nn.Norm, self.norm)(x))

    def _pool(self, x, valid):
        """Attention pooling under one learned task-independent query.

        Masked to valid vertices, so widening the padding cannot move the
        token, and order free, so the vertex index never leaks into it. A graph
        with no valid vertex pools to exactly zero rather than to the uniform
        average a bare softmax would produce.
        """
        B = x.shape[0]
        query = self.sub('query', nn.Embed, 1, self.units)(jnp.zeros((B,), i32))
        keys = self.sub('key', nn.Linear, self.units, **self.kw)(x)
        values = self.sub('val', nn.Linear, self.units, **self.kw)(x)
        score = (keys * query[:, None]).sum(-1) / math.sqrt(self.units)
        attn = jax.nn.softmax(jnp.where(valid > 0, score, -1e9), -1) * valid
        attn = attn / jnp.maximum(attn.sum(-1, keepdims=True), 1e-6)
        pooled = (attn[..., None] * values).sum(1)
        live = nn.cast(valid.sum(-1, keepdims=True) > 0, force=True)
        return self.sub('out', nn.Linear, self.units, **self.kw)(pooled) * live


class GraphDecoder(nj.Module):

    units: int = 256
    embed: int = 64
    entity_vocab: int = 64
    norm: str = 'rms'
    act: str = 'gelu'
    # Normalised box coordinates live in [0, 1], so a beta of 1 would leave the
    # loss purely quadratic and never reach its L1 regime.
    bbox_beta: float = 0.1

    def __init__(self, app_dim, **kw):
        self.app_dim = app_dim
        self.kw = kw
        self.tables = relation_tables()

    def __call__(self, nodes, graph, sem, step_valid):
        """Reconstruct the graph from posterior node representations.

        ``nodes`` is (B, T, N, U); ``graph`` holds the already-unpacked
        (B, T, ...) arrays; ``sem`` is the semantic state (B, T, S, C), which
        only the target head reads; ``step_valid`` is (B, T) and drops the
        terminal transition, whose graph is the previous frame's re-emitted
        copy. Returns per-head (B, T) losses and scalar metrics.
        """
        B, T = nodes.shape[:2]
        x = nodes.reshape((B * T, *nodes.shape[2:]))
        g = {k: v.reshape((B * T, *v.shape[2:])) for k, v in graph.items()}
        m = derive_masks(g)
        step = nn.cast(step_valid.reshape((B * T, 1)), force=True)

        valid = nn.cast(m['valid'], force=True) * step
        seen = nn.cast(m['camera_visible'], force=True)
        known = nn.cast(m['appearance_known'], force=True)
        emask = nn.cast(m['edge_valid'], force=True) * step
        tmask = nn.cast(m['temp_mask'], force=True) * emask
        C = g['graph_node_app'].shape[-2]
        losses, metrics = {}, {}

        # The target is the frozen encoder's own embedding, so it is held fixed
        # here. node_app_var is the collapse detector: a cosine that keeps
        # improving on a degenerate target reads as success.
        pred = self.sub('app', nn.Linear, C * self.app_dim, **self.kw)(x)
        pred = pred.reshape((*x.shape[:-1], C, self.app_dim))
        target = sg(nn.cast(g['graph_node_app'], force=True))
        cos = self._cosine(pred, target)
        # A camera that has never seen the node stores zero and is excluded;
        # a cached embedding for a node currently out of view is not.
        amask = valid[..., None] * known
        node_app = self._masked_mean(1.0 - cos, amask, B, T)
        metrics['node_app_cos'] = self._masked(cos, amask)
        metrics['node_app_var'] = self._spread(target, amask)

        # Only a camera that sees the node has a box; an unseen one packs
        # [0, 0, 0, 0], a legal degenerate box the head would otherwise chase.
        box = self.sub('bbox', nn.Linear, C * 4, **self.kw)(x)
        box = box.reshape((*x.shape[:-1], C, 4))
        btarget = nn.cast(g['graph_node_bbox'], force=True)
        bmask = valid[..., None] * seen
        node_bbox = self._masked_mean(
            self._smooth_l1(box - btarget).sum(-1), bmask, B, T)
        metrics['node_bbox_iou'] = self._masked(self._iou(box, btarget), bmask)

        logit = self.sub('vis', nn.Linear, C, **self.kw)(x)
        vmask = jnp.broadcast_to(valid[..., None], logit.shape)
        node_vis = self._masked_mean(
            jnp.logaddexp(0.0, logit) - seen * logit, vmask, B, T)
        metrics['node_vis_acc'] = self._masked(
            ((logit > 0) == (seen > 0)).astype(f32), vmask)

        losses['node'] = (node_app + node_bbox + node_vis) / 3

        rel = g['graph_edge_rel']
        pair = self._pair(
            x, g['graph_edge_src'], g['graph_edge_dst'], rel, emask)
        losses['relabs'], metrics['relabs_acc'] = self._categorical(
            'abs', pair, g['graph_edge_abs'],
            jnp.asarray(self.tables['abs_valid'])[rel], emask,
            self.tables['n_abs'], B, T)
        losses['reltemp'], metrics['reltemp_acc'] = self._categorical(
            'temp', pair, g['graph_edge_temp'],
            jnp.asarray(self._temp_classes), tmask,
            self.tables['n_temp'], B, T)

        losses['semtgt'], metrics['semtgt_acc'], metrics['semtgt_frac'] = (
            self._target(sem, g, valid, B, T))
        return losses, metrics

    def _target(self, sem, g, valid, B, T):
        """Goal identity read out of the semantic state alone.

        The flag is a per-vertex input to the posterior, so a head reading the
        node representations would copy it back out through a widening path and
        settle at ceiling accuracy with no gradient left. Reading ``sem`` puts
        the discrete bottleneck and the semantic KL in between, which is the
        only thing that makes goal identity persist into h_t -- and imagination
        never sees a graph, so that persistence is the whole point.

        The label is the target's entity id, not its slot. Vertex indices are
        assigned in order of first sight and pooling is permutation invariant
        by construction, so a slot is neither stable nor recoverable here.
        """
        flag = nn.cast(g['graph_node_target'], force=True) * valid
        # valid already carries step_valid, so a terminal frame's re-emitted
        # graph cannot contribute a label.
        present = (flag.sum(-1, keepdims=True) > 0).astype(f32)
        label = (g['graph_node_target'] * g['graph_node_ent']).sum(-1)

        x = self.sub('tgtin', nn.Linear, self.units, **self.kw)(
            nn.cast(sem).reshape((B * T, -1)))
        x = nn.act(self.act)(self.sub('tgtnorm', nn.Norm, self.norm)(x))
        logits = self.sub('tgt', nn.Linear, self.entity_vocab, **self.kw)(x)
        # Index 0 is the pad entity and never a legal target, masked the same
        # way the relation heads mask labels their relation cannot take.
        classes = jnp.arange(self.entity_vocab) > 0
        logits = jnp.where(classes, logits, -1e9)
        logp = jax.nn.log_softmax(logits.astype(f32), -1)
        picked = jnp.take_along_axis(logp, label[..., None], -1)

        loss = self._masked_mean(-picked, present, B, T)
        acc = self._masked(
            (logp.argmax(-1) == label).astype(f32)[..., None], present)
        return loss, acc, present.mean()

    @property
    def _temp_classes(self):
        """All change labels are legal for any rho that carries one; index 0 is
        the pad slot and never a target."""
        mask = np.ones((self.tables['n_temp'],), np.float32)
        mask[0] = 0.0
        return mask

    def _pair(self, x, src, dst, rel, mask):
        source, destination = _edge_selectors(
            src, dst, mask, x.shape[1], x.dtype)
        inp = jnp.concatenate([
            _select_nodes(source, x),
            _select_nodes(destination, x),
            _embed(self, 'reltype', rel, self.tables['n_rel'],
                   self.embed, x.dtype),
        ], -1)
        h = self.sub('pair', nn.Linear, self.units, **self.kw)(inp)
        return nn.act(self.act)(self.sub('pairnorm', nn.Norm, self.norm)(h))

    def _categorical(self, name, h, target, classes, mask, size, B, T):
        logits = self.sub(name, nn.Linear, size, **self.kw)(h)
        logits = jnp.where(classes > 0, logits, -1e9)
        logp = jax.nn.log_softmax(logits.astype(f32), -1)
        picked = jnp.take_along_axis(logp, target[..., None], -1)[..., 0]
        loss = self._masked_mean(-picked, mask, B, T)
        acc = self._masked((logp.argmax(-1) == target).astype(f32), mask)
        return loss, acc

    def _smooth_l1(self, d):
        a = jnp.abs(d)
        b = self.bbox_beta
        return jnp.where(a < b, 0.5 * a ** 2 / b, a - 0.5 * b)

    def _iou(self, pred, target):
        """Axis-aligned IoU over [x0, x1, y0, y1] rows.

        Reported in f32: box coordinates sit in [0, 1], where bf16 resolves
        barely three digits and the overlap is a difference of differences.
        """
        px0, px1, py0, py1 = [pred[..., i].astype(f32) for i in range(4)]
        tx0, tx1, ty0, ty1 = [target[..., i].astype(f32) for i in range(4)]
        iw = jnp.maximum(jnp.minimum(px1, tx1) - jnp.maximum(px0, tx0), 0.0)
        ih = jnp.maximum(jnp.minimum(py1, ty1) - jnp.maximum(py0, ty0), 0.0)
        inter = iw * ih
        pa = jnp.maximum(px1 - px0, 0.0) * jnp.maximum(py1 - py0, 0.0)
        ta = jnp.maximum(tx1 - tx0, 0.0) * jnp.maximum(ty1 - ty0, 0.0)
        return inter / jnp.maximum(pa + ta - inter, 1e-6)

    def _cosine(self, pred, target):
        """Both norms are floored inside the square root, not after it.

        A padding node's representation is exactly zero, so its prediction is
        the zero-initialised bias and the norm's gradient there is infinite.
        Flooring afterwards leaves 0 * inf, which masking cannot remove.
        """
        p, t = pred.astype(f32), target.astype(f32)
        eps = 1e-12
        num = (p * t).sum(-1)
        den = (jnp.sqrt(jnp.maximum((p * p).sum(-1), eps)) *
               jnp.sqrt(jnp.maximum((t * t).sum(-1), eps)))
        return num / den

    def _spread(self, target, mask):
        """Per-channel variance of the appearance target across known entries.

        Flat here means appearance has collapsed to a constant, the one failure
        the cosine cannot see from its own value.
        """
        w = mask[..., None].astype(f32)
        t = target.astype(f32)
        axes = tuple(range(t.ndim - 1))
        mu = (t * w).sum(axes, keepdims=True) / jnp.maximum(
            w.sum(axes, keepdims=True), 1.0)
        return ((t - mu) ** 2 * w).sum() / jnp.maximum(
            w.sum() * t.shape[-1], 1.0)

    def _masked_mean(self, values, mask, B, T):
        v = values.astype(f32) * mask
        axes = tuple(range(1, v.ndim))
        num = v.sum(axes)
        den = jnp.maximum(mask.astype(f32).sum(axes), 1.0)
        return (num / den).reshape((B, T))

    def _masked(self, values, mask):
        m = mask.astype(f32)
        return (values.astype(f32) * m).sum() / jnp.maximum(m.sum(), 1.0)
