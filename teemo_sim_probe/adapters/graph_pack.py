"""Hyper-relational per-frame packing.

Nodes fill a compact prefix in vertex-index order, ee first; each fact is one
row of (relation, absolute, temporal). Arrays use the narrowest dtype holding
their vocabulary since these land in the replay buffer every step; the encoder
casts back to int32 on read.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..core.affordance import canonical_affordance_key
from ..core.relation_rules import (
    AFFORDANCE_RELATIONS,
    PHYSICAL_RELATIONS,
    TEMPORAL_RELATIONS,
)
from ..core.schema import Graph
from .graph_vocab import GraphVocab, entity_key_for


_PHYSICAL = frozenset(PHYSICAL_RELATIONS)
_AFFORDANCE = frozenset(AFFORDANCE_RELATIONS)


def _edge_priority(edge) -> Tuple[int, int]:
    """Truncation order: physical state, then affordance, then spatial.
    Observed facts outrank retained ones within a family."""
    if edge.relation in _PHYSICAL:
        family = 0
    elif edge.relation in _AFFORDANCE:
        family = 1
    else:
        family = 2
    return (family, int(edge.stale))


def pack_graph(
    graph: Graph,
    vocab: GraphVocab,
    *,
    n_max: int,
    e_max: int,
    n_feat: int,
) -> Dict[str, np.ndarray]:
    if n_max > 255:
        raise ValueError(
            f"n_max={n_max} exceeds 255; edge endpoints are packed as uint8"
        )

    node_ent = np.zeros(n_max, dtype=np.uint16)
    node_vis = np.zeros(n_max, dtype=np.uint8)
    node_valid = np.zeros(n_max, dtype=np.uint8)
    node_feat = np.zeros((n_max, n_feat), dtype=np.float32)

    position: Dict[str, int] = {}
    n_nodes = 0
    for node in graph.nodes:
        if n_nodes >= n_max:
            break
        i = n_nodes
        node_ent[i] = vocab.entity.encode(entity_key_for(node))
        node_vis[i] = 1 if node.visible else 0
        node_valid[i] = 1
        if node.feat is not None:
            width = min(n_feat, len(node.feat))
            node_feat[i, :width] = np.asarray(node.feat[:width], dtype=np.float32)
        position[node.node_id] = i
        n_nodes += 1

    candidates = [
        e for e in graph.edges
        if e.src in position and e.dst in position
    ]
    candidates.sort(key=_edge_priority)
    kept = candidates[:e_max]

    edge_src = np.zeros(e_max, dtype=np.uint8)
    edge_dst = np.zeros(e_max, dtype=np.uint8)
    edge_rel = np.zeros(e_max, dtype=np.uint8)
    edge_abs = np.zeros(e_max, dtype=np.uint8)
    edge_temp = np.zeros(e_max, dtype=np.uint8)
    edge_temp_mask = np.zeros(e_max, dtype=np.uint8)
    edge_valid = np.zeros(e_max, dtype=np.uint8)

    for i, e in enumerate(kept):
        edge_src[i] = position[e.src]
        edge_dst[i] = position[e.dst]
        edge_rel[i] = vocab.relation.encode(e.relation)
        edge_abs[i] = vocab.absolute.encode(e.label)
        edge_valid[i] = 1
        if e.relation in TEMPORAL_RELATIONS and e.temp_label is not None:
            edge_temp[i] = vocab.temporal.encode(e.temp_label)
            edge_temp_mask[i] = 1

    # Active target in entity-vocab space so the encoder can match it against
    # graph_node_ent. A target that is not representable as an entity type
    # (articulation handles) encodes to pad.
    obj_id = graph.meta.get("active_obj_id")
    key = canonical_affordance_key(str(obj_id)) if obj_id else None
    target_ent = vocab.entity.token_to_id.get(
        f"actor:{key}" if key else None, vocab.entity.pad_id
    )

    return {
        "graph_node_ent": node_ent,
        "graph_node_vis": node_vis,
        "graph_node_valid": node_valid,
        "graph_node_feat": node_feat,
        "graph_edge_src": edge_src,
        "graph_edge_dst": edge_dst,
        "graph_edge_rel": edge_rel,
        "graph_edge_abs": edge_abs,
        "graph_edge_temp": edge_temp,
        "graph_edge_temp_mask": edge_temp_mask,
        "graph_edge_valid": edge_valid,
        "graph_n_nodes": np.int32(n_nodes),
        "graph_n_edges": np.int32(len(kept)),
        "graph_target_ent": np.uint16(target_ent),
    }


GRAPH_KEYS = (
    "graph_node_ent", "graph_node_vis", "graph_node_valid", "graph_node_feat",
    "graph_edge_src", "graph_edge_dst", "graph_edge_rel", "graph_edge_abs",
    "graph_edge_temp", "graph_edge_temp_mask", "graph_edge_valid",
    "graph_n_nodes", "graph_n_edges", "graph_target_ent",
)
