"""Temporal relations over a horizon K.

For continuous relations (planar-distance, height-offset, grasp- and
contact-compatibility): store the raw value at each frame, compare
``value[t]`` vs ``value[t-K]``, discretize the signed change.

For binary predicates (contact, grasp, support): compare ``predicate[t-K]``
vs ``predicate[t]`` -> gain / lose / maintain / maintain-no. ``maintain-no-*``
is a background class and is not exported as a semantic graph edge.

Compatibility relations are anchor-bound: their history must be cleared
whenever ``a_star`` switches between frames (otherwise the reference jumps
without the gripper moving, producing nonsense signed-change labels) and when
the corresponding affordance edge disappears (target lost / not near anymore).
Planar-distance and height-offset are object-center based and therefore
exempt from this stricter reset.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .schema import Edge, Graph
from .relation_rules import bin_label, _get_bin_spec


# Relations handled as continuous (value -> signed change).
_CONTINUOUS = {
    "planar-distance": "planar-distance-change",
    "height-offset": "height-offset-change",
    "grasp-compatibility": "grasp-compatibility-change",
    "contact-compatibility": "contact-compatibility-change",
}
# Relations handled as binary predicates.
_BINARY = {"contact", "grasp", "support", "containment"}

# Anchor-bound relations whose history is reset on a_star switch or
# affordance-edge disappearance. Planar-distance / height-offset are
# center-based and not in this set.
_AFFORDANCE_RELATIONS = {
    "grasp-compatibility",
    "contact-compatibility",
}

# Binary relations whose ``suppressed_by_grasp`` attribute should also wipe
# the parallel compatibility history so a stale grasp transition is not
# emitted while grasp owns the interaction.
_GRASP_SUPPRESSED_PARTNERS = {
    "contact": ("contact-compatibility",),
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
        stale_ids = {
            n.node_id for n in graph.nodes
            if n.valid_mask and n.frozen_pose
        }
        if stale_ids:
            self.purge(stale_ids)
        # ---- Affordance pre-pass: detect a_star switches / edge absence -- #
        present_affordance: Dict[Tuple[str, str, str], int] = {}
        for e in graph.edges:
            if e.temporal or e.stale:
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

        # (1) Drop history for affordance keys whose edge disappeared.
        for key in list(self._values.keys()):
            _src, _dst, relation = key
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
        seen_pairs: set = set()
        suppressed_keys: set = set()

        for e in graph.edges:
            if e.temporal or e.stale:
                continue
            key = _edge_key(e.src, e.dst, e.relation)
            if e.attributes.get("suppressed_by_grasp"):
                suppressed_keys.add(key)
                self._bools.pop(key, None)
                self._values.pop(key, None)
                # Also drop any partner compatibility history (e.g. drop
                # contact-compatibility when contact is grasp-suppressed).
                for partner in _GRASP_SUPPRESSED_PARTNERS.get(e.relation, ()):
                    partner_key = _edge_key(e.src, e.dst, partner)
                    self._values.pop(partner_key, None)
                    self._last_a_star.pop(partner_key, None)
                    suppressed_keys.add(partner_key)
                continue
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
        node_ids = {
            n.node_id for n in graph.nodes
            if n.valid_mask and not n.frozen_pose
        }
        for key in list(self._bools):
            src, dst, relation = key
            if (
                relation in _BINARY
                and key not in current_bools
                and key not in suppressed_keys
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
        out: List[Edge] = []
        node_ids = {
            n.node_id for n in graph.nodes
            if n.valid_mask and not n.frozen_pose
        }

        # Continuous signed-change.
        for key, hist in self._values.items():
            if len(hist) <= self.K:
                continue
            src, dst, relation = key
            if src not in node_ids or dst not in node_ids:
                continue
            change = hist[-1] - hist[0]          # value[t] - value[t-K]
            change_rel = _CONTINUOUS[relation]
            spec = _get_bin_spec(cfg, change_rel)
            if spec is None:
                continue
            label = bin_label(change, spec[0], spec[1])
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
