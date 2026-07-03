"""Temporal change relations over a horizon K.

Continuous relations only. Physical-state predicates (contact / grasp /
support / contain) rely on consecutive absolute frames instead.

Change label = signed(value[t] - value[t-K]) binned into 5 buckets.

Compatibility change is interpreted as change in the best currently available
interaction opportunity; history is kept across a_star switches (the anchor
cache is sticky per node within a subtask). History is dropped when the
affordance edge disappears entirely, so a resumed history does not bridge a
gap.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Set, Tuple

from .schema import Edge, Graph
from .relation_rules import bin_label, _get_bin_spec


_CONTINUOUS = {
    "planar-distance":       "planar-distance-change",
    "height-offset":         "height-offset-change",
    "grasp-compatibility":   "grasp-compatibility-change",
    "contact-compatibility": "contact-compatibility-change",
    "support-compatibility": "support-compatibility-change",
    "contain-compatibility": "contain-compatibility-change",
}

# Anchor-bound relations whose history must be dropped when their edge stops
# emitting (target lost, no longer near, grasp-suppressed). planar-distance
# and height-offset are exempt: they emit every frame for any visible object.
_AFFORDANCE_RELATIONS = {
    "grasp-compatibility",
    "contact-compatibility",
    "support-compatibility",
    "contain-compatibility",
}


def _edge_key(src: str, dst: str, relation: str) -> Tuple[str, str, str]:
    return (src, dst, relation)


class TemporalBuffer:
    """Per-relation continuous-value history keyed by (src, dst, relation)."""

    def __init__(self, K: int = 5):
        self.K = K
        self._values: Dict[Tuple[str, str, str], Deque[float]] = {}

    def update(self, graph: Graph) -> None:
        stale_ids = {
            n.node_id for n in graph.nodes
            if n.valid_mask and n.frozen_pose
        }
        if stale_ids:
            self.purge(stale_ids)

        present_affordance: Set[Tuple[str, str, str]] = set()
        for e in graph.edges:
            if e.temporal or e.stale:
                continue
            if e.relation in _AFFORDANCE_RELATIONS:
                present_affordance.add(_edge_key(e.src, e.dst, e.relation))

        for key in list(self._values.keys()):
            if key[2] in _AFFORDANCE_RELATIONS and key not in present_affordance:
                del self._values[key]

        for e in graph.edges:
            if e.temporal or e.stale:
                continue
            key = _edge_key(e.src, e.dst, e.relation)
            if e.attributes.get("suppressed_by_grasp"):
                self._values.pop(key, None)
                continue
            if e.relation in _CONTINUOUS and e.raw_value is not None:
                self._values.setdefault(key, deque(maxlen=self.K + 1)).append(
                    float(e.raw_value)
                )

    def purge(self, node_ids) -> None:
        """Drop history for any key whose src or dst is in ``node_ids``."""
        if not node_ids:
            return
        drop_set = set(node_ids)
        for key in [k for k in self._values if k[0] in drop_set or k[1] in drop_set]:
            del self._values[key]

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
            change = hist[-1] - hist[0]
            change_rel = _CONTINUOUS[relation]
            spec = _get_bin_spec(cfg, change_rel)
            if spec is None:
                continue
            label = bin_label(change, spec[0], spec[1])
            out.append(
                Edge(src, dst, change_rel, label, change, temporal=True)
            )
        return out
