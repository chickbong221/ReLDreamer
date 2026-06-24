"""Track A: candidate construction + HARD whitelist gate + overflow truncation.

Pipeline per frame:

    merge_persistent(visible)                 # k_persist identity-keyed retention
    -> expand_local_contact(candidates)        # one-hop V_{t-1} contact (R7)
    -> apply_whitelist(candidates, whitelist)  # hard eligibility gate
    -> overflow_truncate(eligible, n_slots)    # nearest-to-ee tie-broken by node_id

No scoring, no refresh quota, no persistence decay. Distance is used ONLY in
``overflow_truncate`` when more than ``n_slots`` nodes pass the whitelist gate
in a single frame -- never for admission.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .mask_extractor import MaskAccumulator, mask_for_id
from .node_builder import canonical_object_key
from .persistence import _snapshot
from .schema import Node
from .whitelist import Whitelist, match_key
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
    """Stateful selector. One instance per camera/episode.

    Track A: holds the active subtask's ``Whitelist`` (set by the GraphBuilder
    at episode reset). Persistence state survives unselection within the
    ``k_persist`` window -- ``evict_expired`` is the only path that drops
    history entries.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        sel = cfg["selection"]
        self.n_slots = int(sel["n_slots"])
        self.k_persist = int(sel["k_persist"])
        self.enable_local = bool(sel.get("enable_local_contact", True))

        # Active whitelist; set by GraphBuilder.set_whitelist at reset. None
        # means "no asset loaded yet" and triggers a fail-loud error in the
        # selection path -- Track A explicitly forbids a silent
        # "admit everything" fallback.
        self._whitelist: Optional[Whitelist] = None

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
        *,
        masks: Optional[MaskAccumulator] = None,
    ) -> Dict[str, Node]:
        """Add visible entities currently in contact with ee or with a V_{t-1}
        entity. Uses V_{t-1} (not V_t) to keep selection acyclic.

        Bug 1 fix: keys with ``canonical_object_key(ent)`` so the result lives
        in the same namespace as the normal path. If the canonical key already
        exists, the existing node is flagged ``is_local_contact=True`` instead
        of being shadowed by a duplicate.

        Bug 2 fix: when ``masks`` is provided, admit an entity only if its
        ``seg_id`` is in ``masks.visible_seg_ids`` AND a non-empty mask can be
        extracted from ``masks.seg``. The mask is then registered, so the
        admitted node has the same overlay status as a regular segmented node.
        Without ``masks`` (test stubs) the visibility gate is permissive and
        the synthesized node simply has ``pixel_area=0``.
        """
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
        visible_seg_ids = masks.visible_seg_ids if masks is not None else None
        # Scan every seg-id entity. Already-present canonical keys are flagged,
        # not duplicated.
        for seg_id, ent in state.seg_id_map.items():
            if ent is None or seg_id == 0:
                continue
            # Skip robot links.
            if ent in state.robot_links:
                continue
            # Bug 2: require the entity to actually be visible this frame.
            if visible_seg_ids is not None and int(seg_id) not in visible_seg_ids:
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

            ent_id = canonical_object_key(ent)
            name = getattr(ent, "name", None) or str(ent)

            # Bug 1: same canonical key already in graph -> just flag it.
            if ent_id in nodes:
                existing = nodes[ent_id]
                existing.attributes["is_local_contact"] = True
                if int(seg_id) not in existing.segmentation_ids:
                    existing.segmentation_ids.append(int(seg_id))
                continue

            # Bug 2: must carry a real mask if we have the seg image.
            pixel_area = 0
            if masks is not None and masks.seg is not None:
                m = mask_for_id(masks.seg, int(seg_id))
                if not m.any():
                    continue
                masks.add(ent_id, m)
                pixel_area = masks.area(ent_id)

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
                pixel_area=pixel_area,
                pose_world=pose,
                source="local_contact",
                attributes={
                    "is_local_contact": True,
                    "is_actor": type(ent).__name__ == "Actor",
                    "is_link": type(ent).__name__ == "Link",
                    "is_articulation_link": type(ent).__name__ == "Link",
                    "pair_type": "static_object",
                },
            )
        return nodes

    # ---------------------------------------------------------------- whitelist
    def set_whitelist(self, whitelist: Whitelist) -> None:
        """Bind the active subtask's whitelist. Called once at episode reset."""
        self._whitelist = whitelist

    @property
    def whitelist(self) -> Optional[Whitelist]:
        return self._whitelist

    # ---------------------------------------------------------------- selection
    def apply_whitelist(
        self, nodes: Dict[str, Node],
    ) -> Dict[str, Node]:
        """Hard gate: keep ``ee`` plus every node whose ``match_key`` is in the
        active whitelist. Everything else is dropped before slot assignment.

        Raises if no whitelist is bound -- Track A's fail-loud requirement.
        """
        if self._whitelist is None:
            raise RuntimeError(
                "NodeSelector.apply_whitelist called with no whitelist bound. "
                "GraphBuilder must call set_whitelist() during episode reset."
            )
        wl = self._whitelist
        kept: Dict[str, Node] = {}
        for nid, n in nodes.items():
            if n.node_type == "ee":
                kept[nid] = n
                continue
            key = match_key(n)
            if wl.contains(key):
                kept[nid] = n
        return kept

    def overflow_truncate(
        self, nodes: Dict[str, Node],
    ) -> List[str]:
        """Return at most ``n_slots`` eligible entity_ids.

        Tie-break is deterministic: sort by (planar_distance_to_ee, node_id).
        Distance is the ONLY ordering signal here. The ee node is never in the
        returned list (it's tracked separately in GraphBuilder).
        """
        ee = nodes.get("ee")
        ee_xy: Optional[np.ndarray] = None
        if ee is not None and ee.pose_world is not None:
            ee_xy = np.asarray(ee.pose_world[:2], dtype=float)

        candidates: List[Tuple[float, str]] = []
        for ent_id, n in nodes.items():
            if n.node_type == "ee":
                continue
            d = _planar_dist(n, ee_xy)
            candidates.append((d, ent_id))
        # Distance ascending, node_id ascending. infinite distances sort last
        # but the node_id tiebreak still anchors them deterministically.
        candidates.sort(key=lambda t: (t[0], t[1]))
        return [ent_id for _d, ent_id in candidates[: self.n_slots]]

    # ---------------------------------------------------------------- commit
    def commit(self, selected_ids: List[str], nodes: Dict[str, Node], frame: int) -> None:
        """Snapshot every visible object node and record V_{t-1} for next frame.

        Bug P fix: snapshot ALL visible candidate nodes, not just selected
        ones. This lets ``merge_persistent`` re-inject within the k_persist
        window even for entities that lost their slot for a few frames -- the
        intended behavior of the persistence mechanism.
        """
        for ent_id, n in nodes.items():
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

    def evict_expired(self, frame: int) -> List[str]:
        """Drop history entries whose age exceeds ``k_persist`` frames.

        Bug P fix: age-based eviction. An unselected node is NOT dropped --
        only one that has been unseen longer than the retention window. Used
        by ``graph_builder.step`` instead of the old "evict everything not
        selected this frame" sweep, which collapsed the horizon to ~1 frame.
        """
        if self.k_persist <= 0:
            return []
        expired: List[str] = []
        for ent_id, last in list(self._last_seen.items()):
            if (frame - last) > self.k_persist:
                expired.append(ent_id)
        for ent_id in expired:
            self._history.pop(ent_id, None)
            self._last_seen.pop(ent_id, None)
            self._prev_selected.discard(ent_id)
        return expired

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
