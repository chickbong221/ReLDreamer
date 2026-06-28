"""Temporal change relations over a horizon K.

Only continuous compatibility / spatial relations get temporal change edges.
The physical-state predicates (``contact`` / ``grasp`` / ``support`` /
``contain``) are NOT separately annotated as transitions -- the new vocabulary
treats consecutive absolute frames as sufficient evidence of their dynamics.

For continuous relations we store the raw value at each frame, compare
``value[t]`` against ``value[t-K]``, and discretize the signed change into a
5-way label.

Compatibility relations are anchor-bound: their history must be cleared
whenever ``a_star`` switches between frames (otherwise the reference jumps
without the gripper moving, producing nonsense signed-change labels) and when
the corresponding affordance edge disappears (target lost / not near anymore).
Planar-distance and height-offset are object-center based and therefore
exempt from this stricter reset.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

from .schema import Edge, Graph
from .relation_rules import bin_label, _get_bin_spec


# Relations handled as continuous (value -> signed change).
_CONTINUOUS = {
    "planar-distance":       "planar-distance-change",
    "height-offset":         "height-offset-change",
    "grasp-compatibility":   "grasp-compatibility-change",
    "contact-compatibility": "contact-compatibility-change",
    "support-compatibility": "support-compatibility-change",
    "contain-compatibility": "contain-compatibility-change",
}

# Anchor-bound relations whose history is reset on a_star switch or
# affordance-edge disappearance. Planar-distance / height-offset are
# center-based and not in this set.
_AFFORDANCE_RELATIONS = {
    "grasp-compatibility",
    "contact-compatibility",
    "support-compatibility",
    "contain-compatibility",
}


def _edge_key(src: str, dst: str, relation: str) -> Tuple[str, str, str]:
    return (src, dst, relation)


class TemporalBuffer:
    """Per-relation continuous-value history keyed by (src, dst, relation).

    Physical-state predicate history is intentionally NOT tracked here; the
    new vocabulary does not emit ``*-transition`` edges.
    """

    def __init__(self, K: int = 5):
        self.K = K
        self._values: Dict[Tuple[str, str, str], Deque[float]] = {}
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

        # ---- Standard continuous ingest --------------------------------- #
        for e in graph.edges:
            if e.temporal or e.stale:
                continue
            key = _edge_key(e.src, e.dst, e.relation)
            if e.attributes.get("suppressed_by_grasp"):
                # Contact-compatibility edges carry this flag while the
                # endpoint is grasped; drop their history so the change label
                # doesn't latch onto a stale anchor.
                self._values.pop(key, None)
                self._last_a_star.pop(key, None)
                continue
            if e.relation in _CONTINUOUS and e.raw_value is not None:
                self._values.setdefault(key, deque(maxlen=self.K + 1)).append(
                    float(e.raw_value)
                )

    # ---- hard-drop history for evicted nodes ---------------------------- #
    def purge(self, node_ids) -> None:
        """Drop history for any key whose src or dst is in ``node_ids``."""
        if not node_ids:
            return
        drop_set = set(node_ids)
        for store in (self._values, self._last_a_star):
            for key in [k for k in store if k[0] in drop_set or k[1] in drop_set]:
                del store[key]

    # ---- emit temporal edges for current frame -------------------------- #
    def temporal_edges(self, graph: Graph, cfg: dict) -> List[Edge]:
        out: List[Edge] = []
        node_ids = {
            n.node_id for n in graph.nodes
            if n.valid_mask and not n.frozen_pose
        }
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
        return out
