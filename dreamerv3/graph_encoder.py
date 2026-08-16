"""Relation-only graph encoding and the modules that read semantic slots.

``GraphEncoder`` turns one packed frame into ``n_max`` node embeddings through
two message-passing layers over the supplied facts. It emits no pooled token:
the RSSM keeps one recurrent slot per vertex and aligns observations to slots by
uid, so a compressed graph summary would throw away exactly the structure the
slots exist to hold.

``RelationDecoder`` reads a fact back out of a *pair* of slots, and is applied to
both the posterior slots and the predicted prior slots. Training it on the prior
is what keeps imagined slots inside the space where relations still decode.

``SlotReadout`` is the only place attention pooling appears. It feeds the reward,
continuation, actor and value heads, and nothing in the slot transition, the uid
alignment or the relation heads reads it.
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

SCAN_KEYS = GRAPH_KEYS


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
    return {k: graph[k].astype(i32) for k in SCAN_KEYS}


def onehot_embed(module, name, index, classes, units, dtype):
    """Embedding lookup as a one-hot matmul.

    A gather transposes to a scatter-add. These tables hold a handful of rows
    against ``B * e_max`` indices -- padding included, since every padded slot
    still reads row zero -- so the whole batch collides on a few addresses and
    the backward serialises on atomics. The dense form is one small matmul each
    way. ``winit`` is pinned to the output fan so the table initialises exactly
    as ``nn.Embed`` does; the layer only changes how the row is read.
    """
    return module.sub(
        name, nn.Linear, units, bias=False, winit='trunc_normal_out')(
        jax.nn.one_hot(index, classes, dtype=dtype))


def _selectors(src, dst, mask, num_nodes, dtype):
    """Masked dense incidence matrices, ``[B, E, N]`` each.

    The packed fact arrays are padded to a static width, so an advanced gather
    would execute the padded slots too and collide every one of them on node
    zero in the backward. At ``n_max: 6`` the dense contraction is both smaller
    and free of that scatter. Padding rows are exactly zero.
    """
    weight = mask.astype(dtype)[..., None]
    return (jax.nn.one_hot(src, num_nodes, dtype=dtype) * weight,
            jax.nn.one_hot(dst, num_nodes, dtype=dtype) * weight)


def _gather(selector, nodes):
    """One node vector per fact, without an indexed gather."""
    return jnp.einsum('ben,bnu->beu', selector, nodes, optimize='optimal')


class GraphEncoder(nj.Module):

    layers: int = 2
    slot_dim: int = 256
    embed: int = 64
    entity_vocab: int = 64
    uid_vocab: int = 256
    uid_input: bool = False
    norm: str = 'rms'
    act: str = 'gelu'

    def __init__(self, **kw):
        self.kw = kw
        self.tables = relation_tables()

    def __call__(self, graph):
        """``graph`` holds (B, ...) int arrays; returns nodes ``(B, N, D)``.

        Nothing here reads the recurrent state, so the caller batches every
        timestep into B and runs this once, the way the image encoder does.
        """
        g = graph
        ent = g['graph_node_ent']
        rel = g['graph_edge_rel']
        N = ent.shape[1]
        dtype = nn.COMPUTE_DTYPE

        nvalid = nn.cast((ent != 0).astype(f32), force=True)[..., None]
        evalid = nn.cast((rel != 0).astype(f32), force=True)
        tmask = nn.cast(
            (g['graph_edge_temp'] != 0).astype(f32), force=True)

        table = lambda name, index, classes: onehot_embed(
            self, name, index, classes, self.embed, dtype)

        parts = [
            table('ent', ent, self.entity_vocab),
            table('tgt', g['graph_node_target'], 2),
        ]
        if self.uid_input:
            parts.append(table('uid', g['graph_node_uid'], self.uid_vocab))
        x = self._mlp('node', jnp.concatenate(parts, -1)) * nvalid

        fact = self._mlp('fact', jnp.concatenate([
            table('rel', rel, self.tables['n_rel']),
            table('abs', g['graph_edge_abs'], self.tables['n_abs']),
            tmask[..., None] * table(
                'temp', g['graph_edge_temp'], self.tables['n_temp']),
        ], -1)) * evalid[..., None]

        source, dest = _selectors(
            g['graph_edge_src'], g['graph_edge_dst'], evalid, N, dtype)
        # Fan-in per destination vertex, so aggregation is a mean over real
        # facts rather than a sum that grows with how many the packer seated.
        degree = jnp.maximum(dest.sum(1), 1.0)[..., None]

        for i in range(self.layers):
            msg = self._mlp(f'msg{i}', jnp.concatenate(
                [_gather(source, x), _gather(dest, x), fact], -1))
            msg = msg * evalid[..., None]
            agg = jnp.einsum(
                'ben,beu->bnu', dest, msg, optimize='optimal') / degree
            x = x + self._mlp(f'upd{i}', jnp.concatenate([x, agg], -1))
            x = x * nvalid

        return self.sub('out', nn.Norm, self.norm)(x) * nvalid

    def _mlp(self, name, x):
        x = self.sub(name, nn.Linear, self.slot_dim, **self.kw)(x)
        return nn.act(self.act)(self.sub(f'{name}norm', nn.Norm, self.norm)(x))


class RelationDecoder(nj.Module):
    """Absolute and temporal state of one fact, read from a pair of slots.

    ``align`` maps packed vertex index to recurrent slot index, so the same
    packed facts supervise whichever slot currently holds each endpoint. Only
    supplied facts contribute: padded rows are zeroed out of the selectors and
    dropped from the loss mean.
    """

    slot_dim: int = 256
    embed: int = 64
    norm: str = 'rms'
    act: str = 'gelu'

    def __init__(self, **kw):
        self.kw = kw
        self.tables = relation_tables()

    def __call__(self, slots, graph, align, mask):
        rel = graph['graph_edge_rel']
        N = align.shape[1]
        dtype = slots.dtype
        source, dest = _selectors(
            graph['graph_edge_src'], graph['graph_edge_dst'], mask, N, dtype)
        to_slot = lambda sel: jnp.einsum(
            'ben,bns->bes', sel, align, optimize='optimal')
        h = jnp.concatenate([
            _gather(to_slot(source), slots),
            _gather(to_slot(dest), slots),
            onehot_embed(
                self, 'rel', rel, self.tables['n_rel'], self.embed, dtype),
        ], -1)
        h = self.sub('pair', nn.Linear, self.slot_dim, **self.kw)(h)
        h = nn.act(self.act)(self.sub('pairnorm', nn.Norm, self.norm)(h))
        abs_classes = jnp.asarray(self.tables['abs_valid'])[rel]
        # Every change label is legal for any rho that carries one; index 0 is
        # the pad slot and never a target.
        temp_classes = jnp.zeros(
            self.tables['n_temp'], f32).at[1:].set(1.0)
        return (
            self._logp('abs', h, abs_classes, self.tables['n_abs']),
            self._logp('temp', h, temp_classes, self.tables['n_temp']))

    def _logp(self, name, h, classes, size):
        logits = self.sub(name, nn.Linear, size, **self.kw)(h)
        logits = jnp.where(classes > 0, logits, -1e9)
        return jax.nn.log_softmax(logits.astype(f32), -1)


class SlotReadout(nj.Module):
    """Attention pooling under one learned task-independent query.

    Masked to occupied slots, so an empty slot cannot move the token, and order
    free, so the slot index never leaks into it. A state with no occupied slot
    pools to exactly zero rather than to the uniform average a bare softmax
    would produce.
    """

    slot_dim: int = 256
    norm: str = 'rms'

    def __init__(self, **kw):
        self.kw = kw

    def __call__(self, slots, mask):
        shape = slots.shape[:-2]
        x = slots.reshape((-1, *slots.shape[-2:]))
        m = nn.cast(mask, force=True).reshape((-1, mask.shape[-1]))
        B = x.shape[0]
        query = self.sub('query', nn.Embed, 1, self.slot_dim)(
            jnp.zeros((B,), i32))
        keys = self.sub('key', nn.Linear, self.slot_dim, **self.kw)(x)
        values = self.sub('val', nn.Linear, self.slot_dim, **self.kw)(x)
        score = (keys * query[:, None]).sum(-1) / math.sqrt(self.slot_dim)
        attn = jax.nn.softmax(jnp.where(m > 0, score, -1e9), -1) * m
        attn = attn / jnp.maximum(attn.sum(-1, keepdims=True), 1e-6)
        pooled = (attn[..., None] * values).sum(1)
        live = nn.cast(m.sum(-1, keepdims=True) > 0, force=True)
        out = self.sub('out', nn.Linear, self.slot_dim, **self.kw)(pooled)
        return (out * live).reshape((*shape, self.slot_dim))
