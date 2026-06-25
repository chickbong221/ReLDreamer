"""Whitelist admission, role-aware capacity, and short-term persistence.

Pipeline per frame:

    merge_persistent(visible)                 # k_persist identity-keyed retention
    -> apply_whitelist(candidates, whitelist)  # hard eligibility gate
    -> overflow_truncate(eligible, n_slots)    # role-aware, then distance/id

No scoring or secondary contact-based admission path exists. Distance is used
only to break capacity ties within a whitelist-role priority.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .persistence import _snapshot
from .schema import Node
from .whitelist import Whitelist, match_key


def _planar_dist(a: Node, b_xy: Optional[np.ndarray]) -> float:
    if b_xy is None or a.pose_world is None:
        return float("inf")
    return float(np.linalg.norm(np.asarray(a.pose_world[:2], dtype=float) - b_xy))


class NodeSelector:
    """Stateful selector. One instance per camera/episode.

    Holds the active subtask's ``Whitelist`` (set by the GraphBuilder
    at episode reset). Persistence state survives unselection within the
    ``k_persist`` window -- ``evict_expired`` is the only path that drops
    history entries.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        sel = cfg["selection"]
        self.n_slots = int(sel["n_slots"])
        self.k_persist = int(sel["k_persist"])

        # Active whitelist; set by GraphBuilder.set_whitelist at reset. None
        # means "no asset loaded yet" and triggers a fail-loud error in the
        # selection path -- the hard gate explicitly forbids a silent
        # "admit everything" fallback.
        self._whitelist: Optional[Whitelist] = None

        # Frozen snapshots of recently visible nodes (keyed by entity_id).
        self._history: Dict[str, Node] = {}
        self._last_seen: Dict[str, int] = {}

    # ---------------------------------------------------------------- reset
    def reset_episode(self) -> None:
        self._history.clear()
        self._last_seen.clear()

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

        Raises if no whitelist is bound.
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
                n.attributes["whitelist_key"] = key
                n.attributes["whitelist_roles"] = sorted(wl.roles(key))
                kept[nid] = n
        return kept

    def overflow_truncate(
        self, nodes: Dict[str, Node],
    ) -> List[str]:
        """Return at most ``n_slots`` eligible entity_ids.

        Interacted entities are retained first, direct supporters second, and
        distance plus node id break ties within a role. The ee node is tracked
        separately.
        """
        ee = nodes.get("ee")
        ee_xy: Optional[np.ndarray] = None
        if ee is not None and ee.pose_world is not None:
            ee_xy = np.asarray(ee.pose_world[:2], dtype=float)

        candidates: List[Tuple[int, float, str]] = []
        for ent_id, n in nodes.items():
            if n.node_type == "ee":
                continue
            roles = set(n.attributes.get("whitelist_roles") or [])
            role_rank = (
                0 if "interacted" in roles else (1 if "support" in roles else 2)
            )
            d = _planar_dist(n, ee_xy)
            candidates.append((role_rank, d, ent_id))
        candidates.sort(key=lambda t: (t[0], t[1], t[2]))
        return [ent_id for _rank, _d, ent_id in candidates[: self.n_slots]]

    # ---------------------------------------------------------------- commit
    def commit(self, nodes: Dict[str, Node], frame: int) -> None:
        """Snapshot every visible whitelisted object for short persistence."""
        for ent_id, n in nodes.items():
            if n is None or n.node_type != "object":
                continue
            if n.visible:
                self._last_seen[ent_id] = frame
                self._history[ent_id] = _snapshot(n)

    def evict(self, evicted_ids: Iterable[str]) -> None:
        for ent_id in evicted_ids:
            self._history.pop(ent_id, None)
            self._last_seen.pop(ent_id, None)

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
        return expired
