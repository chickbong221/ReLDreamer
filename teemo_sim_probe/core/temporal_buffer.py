"""Temporal relations over a horizon K.

For continuous relations (planar-distance, height-offset, orientation-
alignment): store the raw value at each frame, compare ``value[t]`` vs
``value[t-K]``, discretize the signed change.

For binary predicates (contact, grasp, support): compare ``predicate[t-K]``
vs ``predicate[t]`` -> gain / lose / maintain / maintain-no. ``maintain-no-*``
is a background class and is not exported as a semantic graph edge.

Anchor-bound histories (those whose dst carries an ``affordance_a_star``) have
stricter clearing than the center-based variants: they are reset whenever
``a_star`` switches between frames -- otherwise the affordance reference jumps
without the gripper moving, producing nonsense signed-change labels. They are
also cleared when the corresponding affordance edge disappears (target lost /
asset removed).
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
    "orientation-alignment": "orientation_alignment_change",
}
# Relations handled as binary predicates.
_BINARY = {"contact", "grasp", "support", "containment"}

# Anchor-bound relations whose history must be cleared on a_star switch or
# affordance-edge disappearance. orientation-alignment is always anchor-bound;
# planar-distance / height-offset are anchor-bound ONLY for interactive
# objects with an asset (dst node carries ``affordance_a_star``); the static
# / asset-less interactive variants are center-based and exempt.
_AFFORDANCE_RELATIONS = {
    "orientation-alignment",
    "planar-distance",
    "height-offset",
}


def _edge_key(src: str, dst: str, relation: str) -> Tuple[str, str, str]:
    return (src, dst, relation)


class TemporalBuffer:
    """Per-relation history keyed by (src, dst, relation)."""

    def __init__(self, K: int = 5):
        self.K = K
        # value history (continuous) and bool history (binary)
        self._values: Dict[Tuple[str, str, str], Deque[float]] = {}
        self._bools: Dict[Tuple[str, str, str], Deque[bool]] = {}
        # Last a_star seen per affordance edge key. Used to detect component
        # switches and drop stale change-history.
        self._last_a_star: Dict[Tuple[str, str, str], int] = {}

    # ---- ingest one frame's absolute edges ------------------------------ #
    def update(self, graph: Graph) -> None:
        # ---- Affordance pre-pass: detect a_star switches / edge absence -- #
        # Read the current a_star for each affordance edge from its dst node
        # attributes (set by relation_rules._resolve_active_anchor).
        present_affordance: Dict[Tuple[str, str, str], int] = {}
        for e in graph.edges:
            if e.temporal:
                continue
            if e.relation in _AFFORDANCE_RELATIONS:
                dst_node = graph.get_node(e.dst)
                a_star = (
                    dst_node.attributes.get("affordance_a_star")
                    if dst_node is not None
                    else None
                )
                if a_star is not None:
                    present_affordance[_edge_key(e.src, e.dst, e.relation)] = int(a_star)

        # (1) Drop history for ANCHOR-BOUND keys whose edge disappeared or
        #     whose dst no longer carries an a_star this frame. A key is
        #     anchor-bound only if it was tracked WITH an a_star previously;
        #     center-based planar-distance / height-offset (no a_star) is
        #     exempt so static + asset-less interactive histories survive.
        for key in list(self._values.keys()):
            src, dst, relation = key
            if relation not in _AFFORDANCE_RELATIONS:
                continue
            was_anchor_bound = key in self._last_a_star
            if was_anchor_bound and key not in present_affordance:
                del self._values[key]
                self._last_a_star.pop(key, None)

        # (2) Drop history for affordance keys where a_star switched.
        for key, curr in present_affordance.items():
            prev = self._last_a_star.get(key)
            if prev is not None and curr != prev and key in self._values:
                del self._values[key]
            self._last_a_star[key] = curr

        # ---- Standard continuous / binary ingest -------------------------- #
        current_bools: Dict[Tuple[str, str, str], bool] = {}

        # Bug 3b fix: edge-driven support seeding. Track which (src, dst)
        # pairs actually produced a contact or support edge this frame. Only
        # those pairs get their support history seeded False -- the previous
        # implementation seeded every ordered pair in the visible object set,
        # which is what let a transient phantom support (against a maskless
        # node) emit ``lose-support`` to every other object the next frame.
        seen_pairs: set = set()

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

            # Object-object contact or support edge -> remember the unordered
            # pair so we can seed both directions' support history.
            if e.relation in ("contact", "support") and e.src != "ee" and e.dst != "ee":
                seen_pairs.add(frozenset((e.src, e.dst)))

        for pair in seen_pairs:
            ids = list(pair)
            if len(ids) != 2:
                continue
            a, b = ids
            current_bools.setdefault(_edge_key(a, b, "support"), False)
            current_bools.setdefault(_edge_key(b, a, "support"), False)

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
        for store in (self._values, self._bools, self._last_a_star):
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
