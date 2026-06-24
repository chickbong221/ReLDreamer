"""R7-R13: candidate construction + continuous rule-score selection.

Pipeline per frame:

    merge_persistent(visible)               # R8 / R10 / R13
    -> expand_local_contact(candidates)      # R7
    -> score(candidates)                     # R10 continuous score
    -> topk_with_refresh(scores)             # R11 refresh quota

Persistence is identity-keyed, k-frame bounded, decay through ``w_persist *
exp(-age/tau_age)`` in the score.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .e_domain import EDomain
from .persistence import _snapshot
from .schema import Node
from ..adapters.privileged_state import PrivilegedState


def _planar_dist(a: Node, b_xy: Optional[np.ndarray]) -> float:
    if b_xy is None or a.pose_world is None:
        return float("inf")
    return float(np.linalg.norm(np.asarray(a.pose_world[:2], dtype=float) - b_xy))


def _xyz(node: Node) -> Optional[np.ndarray]:
    if node.pose_world is None:
        return None
    return np.asarray(node.pose_world[:3], dtype=float)


def _entity_id(node: Node) -> str:
    """Stable cross-frame key. canonical_object_key already encodes this."""
    return node.node_id


class NodeSelector:
    """Stateful selector. One instance per camera/episode."""

    def __init__(self, cfg: dict, e_domain: EDomain, profile_name: str):
        self.cfg = cfg
        self.e_domain = e_domain
        sel = cfg["selection"]
        self.n_slots = int(sel["n_slots"])
        self.n_refresh = int(sel["n_refresh"])
        self.k_persist = int(sel["k_persist"])
        self.oracle = bool(sel.get("oracle_force_active_target", False))
        self.enable_local = bool(cfg["e_domain"].get("enable_local_contact", True))
        self.w = dict(sel["weights"])
        self.tau_age = float(sel.get("tau_age", 3))
        if profile_name == "room_scale":
            self.tau_dist = float(sel.get("tau_dist_room_scale", 1.5))
        else:
            self.tau_dist = float(sel.get("tau_dist_tabletop", 0.5))

        # Frozen snapshots of recently visible nodes (keyed by entity_id).
        self._history: Dict[str, Node] = {}
        self._last_seen: Dict[str, int] = {}
        # V_{t-1}: entity_ids selected in the previous frame.
        self._prev_selected: Set[str] = set()

    # ---------------------------------------------------------------- reset
    def reset_episode(self) -> None:
        self._history.clear()
        self._last_seen.clear()
        self._prev_selected.clear()

    # ---------------------------------------------------------------- R8 / R13
    def merge_persistent(
        self, fresh: Dict[str, Node], frame: int
    ) -> Dict[str, Node]:
        """Inject frozen snapshots for E_domain entities seen within k_persist
        frames that are missing from the current frame's visible set."""
        if self.k_persist <= 0:
            return fresh
        merged = dict(fresh)
        for ent_id, snap in self._history.items():
            if ent_id in merged:
                continue
            last = self._last_seen.get(ent_id)
            if last is None or (frame - last) > self.k_persist:
                continue
            merged[ent_id] = Node(
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

    # ---------------------------------------------------------------- R7
    def expand_local_contact(
        self,
        nodes: Dict[str, Node],
        state: PrivilegedState,
        prev_selected: Set[str],
    ) -> Dict[str, Node]:
        """Add visible non-E_domain entities currently in contact with ee
        or with a V_{t-1} entity. Uses V_{t-1} (not V_t) to keep selection
        acyclic."""
        if not self.enable_local:
            return nodes
        eps = self.cfg["contact"]["eps_force"]
        # Resolve entities for prev-selected nodes whose live handles we know.
        seed_entities: List[object] = []
        for ent_id in prev_selected:
            n = nodes.get(ent_id)
            if n is None:
                continue
            ent = _resolve_live_entity(n, state)
            if ent is not None:
                seed_entities.append(ent)
        # Scan every seg-id entity. Skip ones already in nodes or in E_domain
        # (they'd be added through the normal path).
        for seg_id, ent in state.seg_id_map.items():
            if ent is None or seg_id == 0:
                continue
            name = getattr(ent, "name", None) or str(ent)
            ent_id = f"local:{name}"
            if ent_id in nodes:
                continue
            # Skip robot links.
            if ent in state.robot_links:
                continue
            # In contact with ee finger?
            touches_ee = False
            try:
                if state.ee_object_contact_force(ent) > eps:
                    touches_ee = True
            except Exception:
                touches_ee = False
            touches_seed = False
            if not touches_ee:
                for seed in seed_entities:
                    try:
                        if state.pairwise_force(ent, seed) > eps:
                            touches_seed = True
                            break
                    except Exception:
                        continue
            if not (touches_ee or touches_seed):
                continue
            # Synthesize a minimal node for the local-contact entity.
            pose = None
            p = getattr(ent, "pose", None)
            if p is not None:
                try:
                    from ..adapters.privileged_state import pose_to_world_array
                    pose = list(pose_to_world_array(p, state.env_idx))
                except Exception:
                    pose = None
            nodes[ent_id] = Node(
                node_id=ent_id,
                node_type="object",
                name=name,
                visible=True,
                segmentation_ids=[int(seg_id)],
                pose_world=pose,
                source="local_contact",
                attributes={
                    "is_local_contact": True,
                    "pair_type": "static_object",
                },
            )
        return nodes

    # ---------------------------------------------------------------- R10
    def score(
        self,
        nodes: Dict[str, Node],
        state: PrivilegedState,
    ) -> Dict[str, float]:
        ee = nodes.get("ee")
        ee_xy = None
        ee_xyz = None
        if ee is not None and ee.pose_world is not None:
            ee_xyz = np.asarray(ee.pose_world[:3], dtype=float)
            ee_xy = ee_xyz[:2]

        eps = self.cfg["contact"]["eps_force"]
        grasp_angle = self.cfg["grasp"]["max_angle"]
        aff_set = self.cfg.get("affordance_set")

        scores: Dict[str, float] = {}
        for ent_id, n in nodes.items():
            if n.node_type != "object":
                continue
            live = _resolve_live_entity(n, state)
            i_contact = 0.0
            i_grasp = 0.0
            if live is not None:
                try:
                    if state.ee_object_contact_force(live) > eps:
                        i_contact = 1.0
                except Exception:
                    pass
                try:
                    if state.is_grasping(live, max_angle=grasp_angle):
                        i_grasp = 1.0
                except Exception:
                    pass

            i_persist = 1.0 if ent_id in self._prev_selected else 0.0
            age = float(n.steps_since_seen)
            persist_decay = float(np.exp(-age / max(self.tau_age, 1e-6)))

            i_aff = 0.0
            if aff_set is not None and not getattr(aff_set, "is_empty", lambda: True)():
                from .affordance import has_affordance
                if has_affordance(aff_set, n):
                    i_aff = 1.0

            roles = self.e_domain.roles(n) if not self.e_domain.empty else set()
            i_state = 1.0 if "state" in roles else 0.0
            i_support = 1.0 if "support" in roles else 0.0
            i_local = 1.0 if n.attributes.get("is_local_contact") else 0.0

            d = float("inf")
            if ee_xy is not None:
                d = _planar_dist(n, ee_xy)
            dist_decay = 0.0
            if np.isfinite(d):
                dist_decay = float(np.exp(-d / max(self.tau_dist, 1e-6)))

            s = (
                self.w["contact"] * i_contact
                + self.w["grasp"]   * i_grasp
                + self.w["persist"] * persist_decay * i_persist
                + self.w["afford"]  * i_aff
                + self.w["state"]   * i_state
                + self.w["support"] * i_support
                + self.w["local"]   * i_local
                + self.w["dist"]    * dist_decay
            )
            scores[ent_id] = s

        # R13 oracle ablation row: if forced, give MS-HAB active targets a
        # score guaranteed to win.
        if self.oracle:
            for ent_id, n in nodes.items():
                if n.attributes.get("is_mshab_active_target"):
                    scores[ent_id] = scores.get(ent_id, 0.0) + 1e6
        return scores

    # ---------------------------------------------------------------- R11
    def topk_with_refresh(
        self, scores: Dict[str, float], prev_selected: Set[str]
    ) -> List[str]:
        order = sorted(scores.items(), key=lambda kv: -kv[1])
        order_ids = [ent_id for ent_id, _ in order]
        n_keep = max(self.n_slots - self.n_refresh, 0)

        selected: List[str] = []
        # First fill: top n_keep from anywhere.
        for ent_id in order_ids:
            if len(selected) >= n_keep:
                break
            selected.append(ent_id)

        # Refresh quota: high-scoring candidates NOT in V_{t-1} and not yet
        # selected.
        sel_set = set(selected)
        refresh_picked = 0
        for ent_id in order_ids:
            if refresh_picked >= self.n_refresh:
                break
            if ent_id in sel_set:
                continue
            if ent_id in prev_selected:
                continue
            selected.append(ent_id)
            sel_set.add(ent_id)
            refresh_picked += 1

        # Fallback: if refresh quota was not exhausted, fill remaining slots
        # from top-score pool.
        for ent_id in order_ids:
            if len(selected) >= self.n_slots:
                break
            if ent_id in sel_set:
                continue
            selected.append(ent_id)
            sel_set.add(ent_id)

        return selected[: self.n_slots]

    # ---------------------------------------------------------------- commit
    def commit(self, selected_ids: List[str], nodes: Dict[str, Node], frame: int) -> None:
        """Snapshot fresh selected nodes and record V_{t-1} for next frame."""
        for ent_id in selected_ids:
            n = nodes.get(ent_id)
            if n is None or n.node_type != "object":
                continue
            if n.visible:
                self._last_seen[ent_id] = frame
                self._history[ent_id] = _snapshot(n)
        self._prev_selected = set(selected_ids)

    def evict(self, evicted_ids: Iterable[str]) -> None:
        for ent_id in evicted_ids:
            self._history.pop(ent_id, None)
            self._last_seen.pop(ent_id, None)
            self._prev_selected.discard(ent_id)

    @property
    def prev_selected(self) -> Set[str]:
        return set(self._prev_selected)


# --------------------------------------------------------------------------- #
# Live entity resolution (shared with relation_rules._resolve_entity)
# --------------------------------------------------------------------------- #
def _resolve_live_entity(node: Node, state: PrivilegedState):
    """Map a node back to its live SAPIEN entity for force / grasp queries."""
    name = node.name
    for seg_id in node.segmentation_ids:
        ent = state.seg_id_map.get(seg_id)
        if ent is not None and getattr(ent, "name", None) == name:
            return ent
    if node.attributes.get("is_mshab_active_target"):
        if node.attributes.get("mshab_kind") == "obj":
            return state.active_obj
        if node.attributes.get("mshab_kind") == "handle":
            return state.active_handle_link
    for seg_id in node.segmentation_ids:
        ent = state.seg_id_map.get(seg_id)
        if ent is not None:
            return ent
    return None
