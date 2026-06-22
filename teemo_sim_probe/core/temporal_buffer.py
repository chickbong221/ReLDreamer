"""Temporal relations over a horizon K.

For continuous relations (planar-distance, height-offset): store the raw value
at each frame, compare value[t] vs value[t-K], discretize the signed change.

For binary predicates (contact, grasp, support): compare predicate[t-K] vs
predicate[t] -> gain / lose / maintain / maintain-no.

``maintain-no-*`` is an internal/background class. It is not exported as a
semantic graph edge.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .schema import Edge, Graph
from .relation_rules import bin_label


# Relations handled as continuous (value -> signed change).
_CONTINUOUS = {
    "planar-distance": "planar_distance_change",
    "height-offset": "height_offset_change",
    "approach-alignment": "approach_alignment_change",
}
# Relations handled as binary predicates.
_BINARY = {"contact", "grasp", "support", "containment"}


def _edge_key(src: str, dst: str, relation: str) -> Tuple[str, str, str]:
    return (src, dst, relation)


class TemporalBuffer:
    """Per-relation history keyed by (src, dst, relation)."""

    def __init__(self, K: int = 5):
        self.K = K
        # value history (continuous) and bool history (binary)
        self._values: Dict[Tuple[str, str, str], Deque[float]] = {}
        self._bools: Dict[Tuple[str, str, str], Deque[bool]] = {}

    # ---- ingest one frame's absolute edges ------------------------------ #
    def update(self, graph: Graph) -> None:
        current_bools: Dict[Tuple[str, str, str], bool] = {}

        # Positive/negative binary predicates explicitly present in this frame.
        for e in graph.edges:
            if e.temporal:
                continue
            key = _edge_key(e.src, e.dst, e.relation)
            if e.relation in _CONTINUOUS and e.raw_value is not None:
                self._values.setdefault(key, deque(maxlen=self.K + 1)).append(
                    float(e.raw_value)
                )
            elif e.relation in _BINARY:
                positive = e.label == e.relation  # "contact"==relation, etc.
                current_bools[key] = bool(positive)

        # Support is directed and sparse in the absolute graph: only true
        # support edges are emitted. Seed/update all currently visible object
        # pairs with False unless support is true, so old support histories can
        # become lose-support instead of stale maintain-support.
        object_ids = [n.node_id for n in graph.nodes if n.node_type == "object"]
        for src in object_ids:
            for dst in object_ids:
                if src == dst:
                    continue
                key = _edge_key(src, dst, "support")
                current_bools.setdefault(key, False)

        # For any previously tracked binary relation whose endpoints still
        # exist, absence from the absolute graph means the predicate is false.
        node_ids = set(graph.node_ids())
        for key in list(self._bools):
            src, dst, relation = key
            if (
                relation in _BINARY
                and key not in current_bools
                and src in node_ids
                and dst in node_ids
            ):
                current_bools[key] = False

        for key, positive in current_bools.items():
            self._bools.setdefault(key, deque(maxlen=self.K + 1)).append(positive)

    # ---- hard-drop history for evicted nodes ---------------------------- #
    def purge(self, node_ids) -> None:
        """Drop history for any key whose src or dst is in ``node_ids``."""
        if not node_ids:
            return
        drop_set = set(node_ids)
        for store in (self._values, self._bools):
            for key in [k for k in store if k[0] in drop_set or k[1] in drop_set]:
                del store[key]

    # ---- emit temporal edges for current frame -------------------------- #
    def temporal_edges(self, graph: Graph, cfg: dict) -> List[Edge]:
        prof = cfg["profile"]
        out: List[Edge] = []
        node_ids = set(graph.node_ids())

        # Continuous signed-change.
        for key, hist in self._values.items():
            if len(hist) <= self.K:
                continue
            src, dst, relation = key
            if src not in node_ids or dst not in node_ids:
                continue
            change = hist[-1] - hist[0]          # value[t] - value[t-K]
            change_rel = _CONTINUOUS[relation]
            spec = prof.get(change_rel)
            if spec is None:
                continue
            label = bin_label(change, spec["edges"], spec["labels"])
            out.append(
                Edge(src, dst, change_rel, label, change, temporal=True)
            )

        # Binary transitions.
        for key, hist in self._bools.items():
            if len(hist) <= self.K:
                continue
            src, dst, relation = key
            if src not in node_ids or dst not in node_ids:
                continue
            prev, now = hist[0], hist[-1]
            label, masked = _transition_label(relation, prev, now)
            if masked:
                continue
            out.append(
                Edge(
                    src, dst, f"{relation}-transition", label,
                    1.0 if now else 0.0, temporal=True, masked=masked,
                )
            )
        return out


def _transition_label(relation: str, prev: bool, now: bool) -> Tuple[str, bool]:
    if not prev and now:
        return f"gain-{relation}", False
    if prev and not now:
        return f"lose-{relation}", False
    if prev and now:
        return f"maintain-{relation}", False
    # False -> False: background class, never drawn.
    return f"maintain-no-{relation}", True
