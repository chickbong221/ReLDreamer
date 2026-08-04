"""Semantic posterior and graph decoder over the maintained scene graph.

The posterior embeds each vertex and each qualified fact, runs L rounds of
message passing, and attention-pools the result into one fixed-width graph
token; the semantic state is sampled from that token inside the RSSM. The
decoder reconstructs appearance, visibility, and both relation states from the
posterior node representations, so the semantic loss grounds the encoder trunk
the pooling reads from.

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

from teemo_sim_probe.adapters.graph_pack import GRAPH_KEYS
from teemo_sim_probe.adapters.graph_vocab import (
    build_absolute_vocab,
    build_relation_vocab,
    build_temporal_vocab,
)
from teemo_sim_probe.core.relation_rules import (
    ABS_LABELS,
    RELATION_TYPES,
    TEMPORAL_RELATIONS,
)

f32 = jnp.float32
i32 = jnp.int32


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


def unpack(graph):
    """Cast the narrow packed dtypes back to what the embeddings expect."""
    return {
        k: graph[k].astype(f32 if k == 'graph_node_feat' else i32)
        for k in GRAPH_KEYS
    }


class GraphPosterior(nj.Module):

    layers: int = 2
    units: int = 256
    embed: int = 64
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
        ent = g['graph_node_ent']
        vis = nn.cast(g['graph_node_vis'], force=True)
        valid = nn.cast(g['graph_node_valid'], force=True)
        feat = nn.cast(g['graph_node_feat'], force=True)
        src, dst = g['graph_edge_src'], g['graph_edge_dst']
        rel, sig, tau = (
            g['graph_edge_rel'], g['graph_edge_abs'], g['graph_edge_temp'])
        tmask = nn.cast(g['graph_edge_temp_mask'], force=True)
        emask = nn.cast(g['graph_edge_valid'], force=True)
        B, N = ent.shape
        U = self.units

        ent_e = self.sub('ent', nn.Embed, self.entity_vocab, self.embed)(ent)
        x = [nn.symlog(feat), ent_e, vis[..., None]]
        if self.condition_on_deter:
            x.append(jnp.repeat(nn.cast(deter)[:, None], N, 1))
        x = self._mlp('node', jnp.concatenate(x, -1)) * valid[..., None]

        c = [
            self.sub('rel', nn.Embed, self.tables['n_rel'], self.embed)(rel),
            self.sub('abs', nn.Embed, self.tables['n_abs'], self.embed)(sig),
            tmask[..., None] * self.sub(
                'temp', nn.Embed, self.tables['n_temp'], self.embed)(tau),
        ]
        if self.condition_on_deter:
            c.append(jnp.repeat(nn.cast(deter)[:, None], src.shape[1], 1))
        c = self._mlp('fact', jnp.concatenate(c, -1)) * emask[..., None]

        if self.reverse_edges:
            # I_t(i) is one-directional, so without the reverse pass nothing
            # ever flows from an object back to the end effector.
            from_idx = jnp.concatenate([src, dst], 1)
            to_idx = jnp.concatenate([dst, src], 1)
            facts = jnp.concatenate([c, c], 1)
            mask = jnp.concatenate([emask, emask], 1)
            direction = jnp.concatenate([
                jnp.zeros_like(emask), jnp.ones_like(emask)], 1)[..., None]
        else:
            from_idx, to_idx, facts, mask = src, dst, c, emask
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

        return x, self._pool(x, valid, ent, g['graph_target_ent'])

    def _mlp(self, name, x):
        x = self.sub(name, nn.Linear, self.units, **self.kw)(x)
        return nn.act(self.act)(self.sub(f'{name}norm', nn.Norm, self.norm)(x))

    def _pool(self, x, valid, ent, target):
        """Attention pooling with a learned query shifted by the target entity.

        Masked to valid vertices, so widening the padding cannot move the
        token, and order free, so the vertex index never leaks into it.
        """
        is_target = (ent == target[:, None]).astype(i32)
        marked = x + self.sub('marker', nn.Embed, 2, self.units)(is_target)
        query = self.sub('query', nn.Linear, self.units, **self.kw)(
            self.sub('tgt', nn.Embed, self.entity_vocab, self.embed)(target))
        keys = self.sub('key', nn.Linear, self.units, **self.kw)(marked)
        values = self.sub('val', nn.Linear, self.units, **self.kw)(marked)
        score = (keys * query[:, None]).sum(-1) / math.sqrt(self.units)
        score = jnp.where(valid > 0, score, -1e9)
        attn = jax.nn.softmax(score, -1)
        pooled = (attn[..., None] * values).sum(1)
        return self.sub('out', nn.Linear, self.units, **self.kw)(pooled)


class GraphDecoder(nj.Module):

    units: int = 256
    embed: int = 64
    norm: str = 'rms'
    act: str = 'gelu'

    def __init__(self, feat_dim, **kw):
        self.feat_dim = feat_dim
        self.kw = kw
        self.tables = relation_tables()

    def __call__(self, nodes, graph):
        """Reconstruct the graph from posterior node representations.

        ``nodes`` is (B, T, N, U); ``graph`` holds the packed (B, T, ...)
        arrays. Returns per-head (B, T) losses and scalar metrics.
        """
        B, T = nodes.shape[:2]
        x = nodes.reshape((B * T, *nodes.shape[2:]))
        g = {k: v.reshape((B * T, *v.shape[2:]))
             for k, v in unpack(graph).items()}

        valid = nn.cast(g['graph_node_valid'], force=True)
        losses, metrics = {}, {}

        pred = self.sub('app', nn.Linear, self.feat_dim, **self.kw)(x)
        target = nn.symlog(nn.cast(g['graph_node_feat'], force=True))
        losses['semapp'] = self._masked_mean(
            ((pred - target) ** 2).sum(-1), valid, B, T)

        logit = self.sub('vis', nn.Linear, 1, **self.kw)(x)[..., 0]
        vis = nn.cast(g['graph_node_vis'], force=True)
        losses['semvis'] = self._masked_mean(
            jnp.logaddexp(0.0, logit) - vis * logit, valid, B, T)
        metrics['semvis_acc'] = self._masked(
            ((logit > 0) == (vis > 0)).astype(f32), valid)

        rel = g['graph_edge_rel']
        emask = nn.cast(g['graph_edge_valid'], force=True)
        tmask = nn.cast(g['graph_edge_temp_mask'], force=True) * emask
        pair = self._pair(x, g['graph_edge_src'], g['graph_edge_dst'], rel)

        losses['semabs'], metrics['semabs_acc'] = self._categorical(
            'abs', pair, g['graph_edge_abs'],
            jnp.asarray(self.tables['abs_valid'])[rel], emask,
            self.tables['n_abs'], B, T)
        losses['semtemp'], metrics['semtemp_acc'] = self._categorical(
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

    def _masked_mean(self, values, mask, B, T):
        num = (values.astype(f32) * mask).sum(-1)
        den = jnp.maximum(mask.sum(-1), 1.0)
        return (num / den).reshape((B, T))

    def _masked(self, values, mask):
        return (values * mask).sum() / jnp.maximum(mask.sum(), 1.0)
