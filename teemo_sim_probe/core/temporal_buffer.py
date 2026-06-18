"""Temporal relations over a horizon K.

For continuous relations (planar-distance, height-offset): store the raw value
at each frame, compare value[t] vs value[t-K], discretize the signed change.

For binary predicates (contact, grasp, support): compare predicate[t-K] vs
predicate[t] -> gain / lose / maintain / maintain-no.

``maintain-no-*`` edges are produced (the draft uses them as a background
training class) but flagged ``masked=True`` so they are never drawn.
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
                self._bools.setdefault(key, deque(maxlen=self.K + 1)).append(
                    bool(positive)
                )

    # ---- emit temporal edges for current frame -------------------------- #
    def temporal_edges(self, graph: Graph, cfg: dict) -> List[Edge]:
        prof = cfg["profile"]
        out: List[Edge] = []

        # Continuous signed-change.
        for key, hist in self._values.items():
            if len(hist) <= self.K:
                continue
            src, dst, relation = key
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
            prev, now = hist[0], hist[-1]
            label, masked = _transition_label(relation, prev, now)
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
