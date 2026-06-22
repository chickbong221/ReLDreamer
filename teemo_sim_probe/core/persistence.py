"""Persistent-node registry (MS-HAB only).

Implements the draft's persistent-node mechanism: retain object nodes across
short visibility gaps, gate retention by a bounded cap with a manipulation-
aware tier ranking, and freeze poses for invisible-but-retained nodes.

Non-MS-HAB envs (fixed cam + fixed object set) bypass this entirely.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from .schema import Graph, Node
from ..adapters.privileged_state import PrivilegedState


def _snapshot(node: Node) -> Node:
    """Frozen copy of a visible node used as the retention seed."""
    return Node(
        node_id=node.node_id,
        node_type=node.node_type,
        name=node.name,
        visible=node.visible,
        segmentation_ids=list(node.segmentation_ids),
        pixel_area=node.pixel_area,
        pose_world=list(node.pose_world) if node.pose_world else None,
        persistent=node.persistent,
        steps_since_seen=node.steps_since_seen,
        source=node.source,
        frozen_pose=False,
        attributes=dict(node.attributes),
    )


def _planar_dist(node: Node, ee_xy: Optional[np.ndarray]) -> float:
    if ee_xy is None or node.pose_world is None:
        return float("inf")
    return float(np.linalg.norm(np.asarray(node.pose_world[:2]) - ee_xy))


class PersistentNodeRegistry:
    """Track retained nodes and apply the tier-ranked N_max cap.

    Tier 0 (uncapped): ee, MS-HAB active object / handle.
    Tier 1: currently or recently grasped/contacted by EE, plus one-hop
            supporters of any Tier-0/1 node.
    Tier 2: currently visible objects, tiebreak smaller distance-to-EE.
    Tier 3: retained-but-invisible (frozen pose), tiebreak smaller tau then
            smaller distance-to-EE.
    """

    def __init__(self, n_max: int = 6, w_keep: int = 3, w_manip: int = 5):
        self.n_max = int(n_max)
        self.w_keep = int(w_keep)
        self.w_manip = int(w_manip)
        # Frozen-pose snapshots keyed by node_id (object nodes only).
        self._history: Dict[str, Node] = {}
        # node_id -> frame last visible.
        self._last_seen: Dict[str, int] = {}
        # Ring buffers of node_ids in grasp / EE-contact in the last W_manip
        # frames. Each entry is the set seen at one frame.
        self._recent_grasp: Deque[Set[str]] = deque(maxlen=max(1, self.w_manip))
        self._recent_contact: Deque[Set[str]] = deque(maxlen=max(1, self.w_manip))

    # ---- retention -------------------------------------------------------- #
    def merge_retained(
        self, fresh_nodes: Dict[str, Node], frame: int
    ) -> Dict[str, Node]:
        """Inject retained snapshots (frozen pose) for nodes missing from the
        current frame but seen within ``w_keep`` frames."""
        merged = dict(fresh_nodes)
        for nid, snap in self._history.items():
            if nid in merged:
                continue
            last = self._last_seen.get(nid)
            if last is None or (frame - last) > self.w_keep:
                continue
            merged[nid] = Node(
                node_id=snap.node_id,
                node_type=snap.node_type,
                name=snap.name,
                visible=False,
                segmentation_ids=[],
                pixel_area=0,
                pose_world=list(snap.pose_world) if snap.pose_world else None,
                persistent=True,
                steps_since_seen=frame - last,
                source=snap.source,
                frozen_pose=True,
                attributes=dict(snap.attributes),
            )
        return merged

    def snapshot_visible(self, nodes: Dict[str, Node], frame: int) -> None:
        """Record fresh snapshots for currently visible object nodes."""
        for nid, n in nodes.items():
            if n.node_type != "object" or not n.visible:
                continue
            self._last_seen[nid] = frame
            self._history[nid] = _snapshot(n)

    # ---- tier ranking & cap ---------------------------------------------- #
    def rank_and_cap(
        self,
        nodes: Dict[str, Node],
        state: PrivilegedState,
        cfg: dict,
    ) -> Set[str]:
        """Return the set of node_ids to evict to satisfy ``n_max``."""
        from .relation_rules import _resolve_entity   # local to avoid cycles

        ee_node = nodes.get("ee")
        ee_xy = None
        if ee_node is not None and ee_node.pose_world is not None:
            ee_xy = np.asarray(ee_node.pose_world[:2], dtype=float)

        objs = [(nid, n) for nid, n in nodes.items() if n.node_type == "object"]

        # Tier 0: MS-HAB active targets.
        tier0: Set[str] = set()
        for nid, n in objs:
            if n.attributes.get("is_mshab_active_target"):
                tier0.add(nid)

        # Current EE grasp / contact via privileged predicates.
        eps_contact = cfg["contact"]["eps_force"]
        grasp_angle = cfg["grasp"]["max_angle"]
        current_grasp: Set[str] = set()
        current_contact: Set[str] = set()
        for nid, n in objs:
            ent = _resolve_entity(n, state)
            if ent is None:
                continue
            try:
                if state.ee_object_contact_force(ent) > eps_contact:
                    current_contact.add(nid)
                if state.is_grasping(ent, max_angle=grasp_angle):
                    current_grasp.add(nid)
            except Exception:
                continue

        recent_grasp: Set[str] = set().union(*self._recent_grasp) if self._recent_grasp else set()
        recent_contact: Set[str] = set().union(*self._recent_contact) if self._recent_contact else set()

        tier1: Set[str] = set()
        for nid, n in objs:
            if nid in tier0:
                continue
            if (nid in current_grasp or nid in current_contact
                    or nid in recent_grasp or nid in recent_contact):
                tier1.add(nid)

        # One-hop supporter retention: a node that supports any Tier-0/1 object
        # is itself promoted to Tier 1.
        eps_z = cfg["support"].get("eps_z", 0.02)
        min_vert_ratio = cfg["support"].get("min_vertical_force_ratio", 0.5)
        protected = tier0 | tier1
        for nid, n in objs:
            if nid in tier0 or nid in tier1:
                continue
            for prot_id in protected:
                prot = nodes.get(prot_id)
                if prot is None or prot.node_type != "object":
                    continue
                if _is_supporter(n, prot, state, eps_z, min_vert_ratio):
                    tier1.add(nid)
                    break

        # Tier 2 / Tier 3 split.
        tier2: List[str] = []
        tier3: List[str] = []
        for nid, n in objs:
            if nid in tier0 or nid in tier1:
                continue
            (tier2 if n.visible else tier3).append(nid)

        tier1_sorted = sorted(tier1, key=lambda nid: _planar_dist(nodes[nid], ee_xy))
        tier2.sort(key=lambda nid: _planar_dist(nodes[nid], ee_xy))
        tier3.sort(key=lambda nid: (nodes[nid].steps_since_seen,
                                     _planar_dist(nodes[nid], ee_xy)))

        # Build kept list under cap. Tier 0 always kept and uncounted.
        kept_objs: List[str] = []
        budget = self.n_max
        for tier_list in (tier1_sorted, tier2, tier3):
            for nid in tier_list:
                if len(kept_objs) >= budget:
                    break
                kept_objs.append(nid)
            if len(kept_objs) >= budget:
                break

        kept_ids = tier0 | set(kept_objs) | {nid for nid, n in nodes.items()
                                              if n.node_type == "ee"}
        evicted = {nid for nid in nodes if nid not in kept_ids}
        return evicted

    # ---- post-edge bookkeeping ------------------------------------------- #
    def record_recency(self, graph: Graph) -> None:
        """Push this frame's EE-contact / EE-grasp positives into the rings."""
        grasp_now: Set[str] = set()
        contact_now: Set[str] = set()
        for e in graph.edges:
            if e.src != "ee" or e.temporal:
                continue
            if e.relation == "grasp" and e.label == "grasp":
                grasp_now.add(e.dst)
            elif e.relation == "contact" and e.label == "contact" and not e.masked:
                contact_now.add(e.dst)
        self._recent_grasp.append(grasp_now)
        self._recent_contact.append(contact_now)

    def drop(self, node_ids) -> None:
        """Forget evicted nodes so they cannot resurrect."""
        for nid in node_ids:
            self._history.pop(nid, None)
            self._last_seen.pop(nid, None)


def _is_supporter(
    supporter: Node,
    supported: Node,
    state: PrivilegedState,
    eps_z: float,
    min_vert_ratio: float,
) -> bool:
    """Cheap one-hop supporter test using z-ordering + vertical contact force."""
    if supporter.pose_world is None or supported.pose_world is None:
        return False
    pa = np.asarray(supporter.pose_world[:3], dtype=float)
    pb = np.asarray(supported.pose_world[:3], dtype=float)
    if pa[2] + eps_z >= pb[2]:
        return False
    from .relation_rules import _resolve_entity
    ea = _resolve_entity(supporter, state)
    eb = _resolve_entity(supported, state)
    if ea is None or eb is None:
        return False
    try:
        force_vec = state.pairwise_force_vector(ea, eb)
    except Exception:
        return False
    force = float(np.linalg.norm(force_vec))
    if force <= 1e-9:
        return False
    return (abs(float(force_vec[2])) / force) >= min_vert_ratio
