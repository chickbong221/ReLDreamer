"""Ragged-friendly per-frame graph packing.

Node prefix is compact: ee at index 0, valid objects at 1..n_nodes-1. Edges
reference these compact positions. Stale edges survive only for physical-state
relations.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from ..core.schema import Graph
from .graph_vocab import EdgeVocab, NodeVocab, node_key_for


_PHYSICAL_STATE = frozenset({"contact", "grasp", "support", "contain"})
_COMPAT = frozenset({
    "grasp-compatibility", "contact-compatibility",
    "support-compatibility", "contain-compatibility",
})


def _edge_priority(edge) -> Tuple[int, int]:
    r = edge.relation
    if r in _PHYSICAL_STATE:
        cat = 0
    elif r in _COMPAT:
        cat = 1
    elif r.endswith("-change"):
        cat = 2
    else:
        cat = 3
    return (cat, int(edge.stale))


def pack_graph(
    graph: Graph,
    node_vocab: NodeVocab,
    edge_vocab: EdgeVocab,
    *,
    n_max: int,
    e_max: int,
    k_soft: float,
    staleness_enabled: bool = True,
) -> Dict[str, np.ndarray]:
    node_ids = np.zeros(n_max, dtype=np.int32)
    node_ee_mask = np.zeros(n_max, dtype=np.float32)
    node_conf = np.zeros(n_max, dtype=np.float32)

    k_soft = max(float(k_soft), 1e-6)
    node_id_to_slot: Dict[str, int] = {}
    n_nodes = 0
    for node in graph.nodes:
        if not node.valid_mask:
            continue
        if n_nodes >= n_max:
            break
        i = n_nodes
        node_ids[i] = node_vocab.encode(node_key_for(node))
        if node.node_type == "ee":
            node_ee_mask[i] = 1.0
            node_conf[i] = 1.0
        elif staleness_enabled:
            node_conf[i] = float(math.exp(-float(node.steps_since_seen) / k_soft))
        else:
            node_conf[i] = 1.0
        node_id_to_slot[node.node_id] = i
        n_nodes += 1

    candidates = [
        e for e in graph.edges
        if not e.masked
        and e.src in node_id_to_slot
        and e.dst in node_id_to_slot
        and not (staleness_enabled and e.stale and e.relation not in _PHYSICAL_STATE)
    ]
    candidates.sort(key=_edge_priority)
    kept = candidates[:e_max]

    edge_src = np.zeros(e_max, dtype=np.int32)
    edge_dst = np.zeros(e_max, dtype=np.int32)
    edge_pred = np.zeros(e_max, dtype=np.int32)

    for i, e in enumerate(kept):
        edge_src[i] = node_id_to_slot[e.src]
        edge_dst[i] = node_id_to_slot[e.dst]
        edge_pred[i] = edge_vocab.encode(e.relation, e.label)

    return {
        "graph_node_ids": node_ids,
        "graph_node_ee_mask": node_ee_mask,
        "graph_node_conf": node_conf,
        "graph_edge_src": edge_src,
        "graph_edge_dst": edge_dst,
        "graph_edge_pred": edge_pred,
        "graph_n_nodes": np.int32(n_nodes),
        "graph_n_edges": np.int32(len(kept)),
    }


GRAPH_KEYS = (
    "graph_node_ids", "graph_node_ee_mask", "graph_node_conf",
    "graph_edge_src", "graph_edge_dst", "graph_edge_pred",
    "graph_n_nodes", "graph_n_edges",
)
