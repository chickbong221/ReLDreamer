"""Fixed-shape numpy packing for one Graph per env.

Layout:
    graph_node_ids     (N_max,) int32   -- NodeVocab ids, pad_id at pad slots
    graph_node_valid   (N_max,) float32 -- 1.0 for valid, 0.0 for pad
    graph_node_ee_mask (N_max,) float32 -- 1.0 at the ee slot
    graph_node_conf    (N_max,) float32 -- exp(-tau_i / K_soft), 0.0 at pad
    graph_edge_src     (E_max,) int32   -- position into N_max
    graph_edge_dst     (E_max,) int32
    graph_edge_pred    (E_max,) int32   -- EdgeVocab id, pad_edge at pad rows
    graph_edge_valid   (E_max,) float32 -- 1.0 for valid, 0.0 for pad

Node positions follow the builder's ordering: index 0 is the ee node, indices
1..n_slots are the object slots (padded slots contribute pad_id / valid=0).

Edges are truncated to E_max under the priority
``physical-state > compat > change > spatial``; fresh (non-stale) beats stale
within a category. Masked edges (e.g. contact suppressed by grasp) are dropped.
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
) -> Dict[str, np.ndarray]:
    node_ids = np.full(n_max, node_vocab.pad_id, dtype=np.int32)
    node_valid = np.zeros(n_max, dtype=np.float32)
    node_ee_mask = np.zeros(n_max, dtype=np.float32)
    node_conf = np.zeros(n_max, dtype=np.float32)

    k_soft = max(float(k_soft), 1e-6)
    node_id_to_slot: Dict[str, int] = {}
    for i, node in enumerate(graph.nodes[:n_max]):
        if not node.valid_mask:
            continue
        node_ids[i] = node_vocab.encode(node_key_for(node))
        node_valid[i] = 1.0
        if node.node_type == "ee":
            node_ee_mask[i] = 1.0
            node_conf[i] = 1.0
        else:
            node_conf[i] = float(math.exp(-float(node.steps_since_seen) / k_soft))
        node_id_to_slot[node.node_id] = i

    candidates = [
        e for e in graph.edges
        if not e.masked
        and e.src in node_id_to_slot
        and e.dst in node_id_to_slot
    ]
    candidates.sort(key=_edge_priority)
    kept = candidates[:e_max]

    edge_src = np.zeros(e_max, dtype=np.int32)
    edge_dst = np.zeros(e_max, dtype=np.int32)
    edge_pred = np.full(e_max, edge_vocab.pad_id, dtype=np.int32)
    edge_valid = np.zeros(e_max, dtype=np.float32)

    for i, e in enumerate(kept):
        edge_src[i] = node_id_to_slot[e.src]
        edge_dst[i] = node_id_to_slot[e.dst]
        edge_pred[i] = edge_vocab.encode(e.relation, e.label)
        edge_valid[i] = 1.0

    return {
        "graph_node_ids": node_ids,
        "graph_node_valid": node_valid,
        "graph_node_ee_mask": node_ee_mask,
        "graph_node_conf": node_conf,
        "graph_edge_src": edge_src,
        "graph_edge_dst": edge_dst,
        "graph_edge_pred": edge_pred,
        "graph_edge_valid": edge_valid,
    }


GRAPH_KEYS = (
    "graph_node_ids", "graph_node_valid", "graph_node_ee_mask", "graph_node_conf",
    "graph_edge_src", "graph_edge_dst", "graph_edge_pred", "graph_edge_valid",
)
