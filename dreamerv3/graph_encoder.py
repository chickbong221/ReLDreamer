"""Semantic posterior and graph decoder over the maintained scene graph.

The posterior embeds each vertex and each qualified fact, runs L rounds of
message passing, and attention-pools the result into one fixed-width graph
token; the semantic state is sampled from that token inside the RSSM. The
decoder reconstructs appearance, the bounding box, visibility, and both
relation states from the posterior node representations, so the semantic loss
grounds the encoder trunk the pooling reads from.

A vertex is ``[AppProj_c(a_c), BBoxProj_c(b_c), ..., EntityEmbed(id)]`` over
cameras ``c``. Appearance arrives on ``graph_node_app`` as frozen DINO features
read straight from replay; nothing here produces it.

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


class GraphPosterior(nj.Module):

    layers: int = 2
    units: int = 256
    embed: int = 64
    app: int = 64
    bbox: int = 8
    reverse_edges: bool = True
    condition_on_deter: bool = True
    entity_vocab: int = 64
    norm: str = 'rms'
    act: str = 'gelu'

    def __init__(self, **kw):
        self.kw = kw
        self.tables = relation_tables()

    def __call__(self, graph, deter):
        """One timestep. ``graph`` holds (B, ...) arrays, ``deter`` is (B, D).

        Returns the node representations (B, N, U) and the pooled graph token
        (B, U).
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
        B, N = ent.shape
        C = app.shape[-2]
        U = self.units

        cond = nn.cast(deter) if self.condition_on_deter else None

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
        x = self._mlp('node', jnp.concatenate(parts, -1), cond) * valid[..., None]

        fact = jnp.concatenate([
            self.sub('rel', nn.Embed, self.tables['n_rel'], self.embed)(rel),
            self.sub('abs', nn.Embed, self.tables['n_abs'], self.embed)(sig),
            tmask[..., None] * self.sub(
                'temp', nn.Embed, self.tables['n_temp'], self.embed)(tau),
        ], -1)
        fact = self._mlp('fact', fact, cond) * emask[..., None]

        if self.reverse_edges:
            # I_t(i) is one-directional, so without the reverse pass nothing
            # ever flows from an object back to the end effector.
            from_idx = jnp.concatenate([src, dst], 1)
            to_idx = jnp.concatenate([dst, src], 1)
            facts = jnp.concatenate([fact, fact], 1)
            mask = jnp.concatenate([emask, emask], 1)
            direction = jnp.concatenate([
                jnp.zeros_like(emask), jnp.ones_like(emask)], 1)[..., None]
        else:
            from_idx, to_idx, facts, mask = src, dst, fact, emask
            direction = jnp.zeros_like(emask)[..., None]

        bidx = jnp.repeat(jnp.arange(B)[:, None], from_idx.shape[1], 1)
        for i in range(self.layers):
            gathered = x[bidx, from_idx]
            msg = self._mlp(
                f'msg{i}', jnp.concatenate([gathered, facts, direction], -1))
            msg = msg * mask[..., None]
            total = jnp.zeros((B, N, U), msg.dtype).at[bidx, to_idx].add(msg)
            # Counts accumulate in f32: bf16 loses integer precision past 256,
            # which an e_max above that would silently hit.
            count = jnp.zeros((B, N), f32).at[bidx, to_idx].add(mask.astype(f32))
            agg = total / nn.cast(jnp.maximum(count, 1.0))[..., None]
            x = self._mlp(f'upd{i}', jnp.concatenate([x, agg], -1))
            x = x * valid[..., None]

        return x, self._pool(x, valid)

    def _mlp(self, name, x, cond=None):
        x = self.sub(name, nn.Linear, self.units, **self.kw)(x)
        if cond is not None:
            # h_t is identical for every vertex and every fact, so project it
            # once and broadcast. Concatenating instead would materialise
            # (B, N, |h|) and (B, E, |h|) and matmul them at full width.
            x += self.sub(f'{name}cond', nn.Linear, self.units, bias=False,
                          **self.kw)(cond)[:, None]
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
    norm: str = 'rms'
    act: str = 'gelu'
    # Normalised box coordinates live in [0, 1], so a beta of 1 would leave the
    # loss purely quadratic and never reach its L1 regime.
    bbox_beta: float = 0.1

    def __init__(self, app_dim, **kw):
        self.app_dim = app_dim
        self.kw = kw
        self.tables = relation_tables()

    def __call__(self, nodes, graph, step_valid):
        """Reconstruct the graph from posterior node representations.

        ``nodes`` is (B, T, N, U); ``graph`` holds the already-unpacked
        (B, T, ...) arrays; ``step_valid`` is (B, T) and drops the terminal
        transition, whose graph is the previous frame's re-emitted copy.
        Returns per-head (B, T) losses and scalar metrics.
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
        pair = self._pair(x, g['graph_edge_src'], g['graph_edge_dst'], rel)
        losses['relabs'], metrics['relabs_acc'] = self._categorical(
            'abs', pair, g['graph_edge_abs'],
            jnp.asarray(self.tables['abs_valid'])[rel], emask,
            self.tables['n_abs'], B, T)
        losses['reltemp'], metrics['reltemp_acc'] = self._categorical(
            'temp', pair, g['graph_edge_temp'],
            jnp.asarray(self._temp_classes), tmask,
            self.tables['n_temp'], B, T)
        return losses, metrics

    @property
    def _temp_classes(self):
        """All change labels are legal for any rho that carries one; index 0 is
        the pad slot and never a target."""
        mask = np.ones((self.tables['n_temp'],), np.float32)
        mask[0] = 0.0
        return mask

    def _pair(self, x, src, dst, rel):
        bidx = jnp.repeat(jnp.arange(x.shape[0])[:, None], src.shape[1], 1)
        rel_e = self.sub('reltype', nn.Embed, self.tables['n_rel'], self.embed)(rel)
        inp = jnp.concatenate([x[bidx, src], x[bidx, dst], rel_e], -1)
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
        p, t = pred.astype(f32), target.astype(f32)
        num = (p * t).sum(-1)
        den = jnp.linalg.norm(p, axis=-1) * jnp.linalg.norm(t, axis=-1)
        return num / jnp.maximum(den, 1e-6)

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
