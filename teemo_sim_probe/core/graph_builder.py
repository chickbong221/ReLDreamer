"""Per-frame orchestration (Track A: hard whitelist gate, no soft score).

    GraphBuilder(env, cfg).step(obs, frame, episode_boundary=False)
        -> (Graph, masks, camera, rgb)

Pipeline:

    build_nodes (background / robot / goals / static-scene / area filter)
    -> classify_pair_types          (relation vocabulary specificity)
    -> selector.merge_persistent    (k-frame identity-keyed retention)
    -> tau_i bookkeeping            (steps_since_seen)
    -> selector.expand_local_contact(V_{t-1} contact one hop, mask-gated)
    -> _dedup_by_live_entity        (collapse double-keyed entities)
    -> selector.apply_whitelist     (hard per-subtask eligibility gate)
    -> selector.overflow_truncate   (nearest-to-ee tiebroken by node_id)
    -> slot_manager.assign          (identity-keyed, reset_flag)
    -> build_absolute_edges + temporal edges (valid nodes only)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .affordance import canonical_affordance_key
from .schema import Graph, Node, padding_node
from .node_builder import build_nodes, classify_pair_types
from .relation_rules import build_absolute_edges
from .temporal_buffer import TemporalBuffer
from .mask_extractor import MaskAccumulator
from .selector import NodeSelector, _resolve_live_entity
from .slot_manager import SlotManager
from .whitelist import Whitelist, load_whitelist, resolve_whitelist_path
from ..adapters.privileged_state import get_privileged_state


def _dedup_by_live_entity(
    nodes: Dict[str, Node], state,
) -> Dict[str, Node]:
    """Collapse multiple nodes that resolve to the same live SAPIEN entity.

    Both the normal seg-id path and the local-contact path can produce nodes
    for the same physical entity. After Bug 1's namespace fix this should be
    rare, but it can still happen when MS-HAB persistence injects an
    ``actor:env-0_X-3`` target that later appears under a different SAPIEN
    wrapper. We collapse by live-entity identity, preferring the visible
    representative and merging segmentation_ids + mshab/local-contact flags.
    """
    by_entity: Dict[int, str] = {}
    keep: Dict[str, Node] = {}
    for nid, n in nodes.items():
        if n.node_type == "ee":
            keep[nid] = n
            continue
        live = _resolve_live_entity(n, state)
        if live is None:
            keep[nid] = n
            continue
        key = id(live)
        prior_nid = by_entity.get(key)
        if prior_nid is None:
            by_entity[key] = nid
            keep[nid] = n
            continue
        # Pick the preferred winner; merge the other into it.
        prior = keep[prior_nid]
        winner, loser = _pick_winner(prior, n)
        _merge_into(winner, loser)
        # Replace stored entry with the winner under its own node_id.
        if winner is not prior:
            keep.pop(prior_nid, None)
            keep[winner.node_id] = winner
            by_entity[key] = winner.node_id
    return keep


def _pick_winner(a: Node, b: Node):
    """Prefer (visible, non-persistent, has-pose) > rest. Stable for ties."""
    def rank(n: Node):
        return (
            1 if n.visible else 0,
            0 if n.persistent else 1,
            1 if n.pose_world is not None else 0,
            1 if n.segmentation_ids else 0,
        )
    if rank(a) >= rank(b):
        return a, b
    return b, a


def _merge_into(winner: Node, loser: Node) -> None:
    for sid in loser.segmentation_ids:
        if sid not in winner.segmentation_ids:
            winner.segmentation_ids.append(sid)
    for k, v in loser.attributes.items():
        # Don't overwrite the winner's own attributes; only fill gaps.
        winner.attributes.setdefault(k, v)
    if loser.pose_world is not None and winner.pose_world is None:
        winner.pose_world = list(loser.pose_world)
    if loser.visible and not winner.visible:
        winner.visible = True


class GraphBuilder:
    def __init__(
        self,
        env,
        cfg: dict,
        *,
        env_idx: int = 0,
        env_id: str = "env",
        camera: Optional[str] = None,
        include_goals: bool = False,
        include_background: bool = False,
        include_static_scene: bool = False,
        mshab_object_name: str = "actual",
    ):
        self.env = env
        self.cfg = cfg
        self.env_idx = env_idx
        self.env_id = env_id
        self.camera = camera
        self.include_goals = include_goals
        self.include_background = include_background
        self.include_static_scene = include_static_scene
        self.mshab_object_name = mshab_object_name

        self.temporal = TemporalBuffer(K=cfg["temporal"]["K"])
        self.selector = NodeSelector(cfg)
        self.slots = SlotManager(n_slots=int(cfg["selection"]["n_slots"]))

        # Track A: where the per-subtask whitelists live and which one is
        # currently bound. ``whitelist_dir`` may be absolute or relative to
        # the cfg directory; ``configs.loader`` resolves it.
        self._whitelist_dir: Optional[str] = cfg.get("whitelist_dir")
        self._whitelist_key: Optional[Tuple[str, str]] = None

        self._last_seen: Dict[str, int] = {}
        self._first_unseen: Dict[str, int] = {}

    # ---------------------------------------------------------------- reset
    def reset_episode(self) -> None:
        self.selector.reset_episode()
        self.slots.reset_episode()
        self.temporal = TemporalBuffer(K=self.cfg["temporal"]["K"])
        self._last_seen.clear()
        self._first_unseen.clear()
        # Force whitelist re-resolution next step.
        self._whitelist_key = None

    # ---------------------------------------------------------------- whitelist
    def _resolve_and_bind_whitelist(self, state) -> None:
        """Load the JSON for (active_subtask, canonical(active_obj_id)) and
        bind it on the selector. Fail loud if no matching file exists.

        Cached on ``self._whitelist_key`` so we don't re-read the file every
        frame; the cache is invalidated whenever the cached key differs from
        the live (subtask, target) tuple, which covers MS-HAB sub-task
        advancement within a single episode as well as fresh episodes.
        """
        subtask = state.active_subtask_type
        target = (
            canonical_affordance_key(state.active_obj_id)
            if state.active_obj_id else None
        )
        if subtask is None or target is None:
            raise RuntimeError(
                "Track A whitelist gate requires both an active subtask type "
                f"and an active_obj_id; got subtask={subtask!r}, "
                f"active_obj_id={state.active_obj_id!r}. Probe must run inside "
                "an MS-HAB-like env."
            )
        key = (subtask, target)
        if self._whitelist_key == key and self.selector.whitelist is not None:
            return
        path = resolve_whitelist_path(self._whitelist_dir, subtask, target)
        if path is None:
            raise FileNotFoundError(
                f"per-subtask whitelist not found for subtask={subtask!r}, "
                f"target={target!r} under whitelist_dir={self._whitelist_dir!r}. "
                "Mine assets with tools/build_subtask_whitelists.py."
            )
        wl = load_whitelist(path)
        self.selector.set_whitelist(wl)
        self._whitelist_key = key

    # ---------------------------------------------------------------- step
    def step(
        self, obs: dict, frame: int,
        *,
        episode_boundary: bool = False,
        seg_override=None, rgb_override=None, camera_override=None,
    ) -> Tuple[Graph, MaskAccumulator, str, np.ndarray]:
        if episode_boundary:
            self.reset_episode()

        state = get_privileged_state(
            self.env, self.env_idx, mshab_object_name=self.mshab_object_name,
        )

        # Track A: bind the active subtask's whitelist before any candidate
        # construction. Done every step so MS-HAB subtask advancement within
        # an episode swaps the gate cleanly (idempotent when (subtask, target)
        # is unchanged).
        self._resolve_and_bind_whitelist(state)

        nodes, masks, cam, rgb = build_nodes(
            obs, state,
            camera=self.camera,
            include_goals=self.include_goals,
            include_background=self.include_background,
            include_static_scene=self.include_static_scene,
            min_pixels=self.cfg.get("node", {}).get("min_pixels", 32),
            min_area_ratio=self.cfg.get("node", {}).get("min_area_ratio", 0.0005),
            seg_override=seg_override,
            rgb_override=rgb_override,
            camera_override=camera_override,
        )

        # Relation vocabulary classification (uses affordance set).
        classify_pair_types(nodes, self.cfg)

        # Identity-keyed persistence merge -- re-inject occluded entities
        # whose last_seen age is within k_persist.
        nodes = self.selector.merge_persistent(nodes, frame)

        # tau_i update for every object node.
        for nid, n in nodes.items():
            if n.node_type == "ee":
                continue
            if n.visible:
                self._last_seen[nid] = frame
                n.steps_since_seen = 0
            else:
                if nid not in self._last_seen:
                    first = self._first_unseen.setdefault(nid, frame)
                    n.steps_since_seen = max(1, frame - first + 1)
                else:
                    n.steps_since_seen = frame - self._last_seen[nid]

        # Local-contact exception (uses V_{t-1} -- one-frame lag, acyclic).
        nodes = self.selector.expand_local_contact(
            nodes, state, self.selector.prev_selected, masks=masks,
        )

        # Bug 1 part 2: physical-entity dedup. After both the normal seg-id
        # path and the local-contact path have populated ``nodes``, collapse
        # any pair that resolves to the same live SAPIEN entity so a single
        # entity never claims two slots. Prefer the visible representative and
        # merge segmentation_ids + mshab flags forward.
        nodes = _dedup_by_live_entity(nodes, state)

        # Re-classify so newly added local-contact nodes get a pair_type.
        classify_pair_types(nodes, self.cfg)

        # Track A: hard per-subtask whitelist gate. Everything failing the
        # gate is dropped here; only the ee node and whitelisted entities
        # survive into the slot assignment stage.
        nodes = self.selector.apply_whitelist(nodes)

        # Track A: deterministic overflow truncation -- nearest n_slots to ee
        # by planar distance, tie-broken on node_id ascending. The only place
        # distance influences selection.
        selected_ids = self.selector.overflow_truncate(nodes)

        # Slot assignment (identity-keyed, reset_flag on identity change).
        assignments = self.slots.assign(selected_ids)

        # Bug P fix: do NOT evict an entity from persistence history merely
        # because it was unselected this frame. Persistence eviction is now
        # age-based -- ``merge_persistent`` re-injects within k_persist, and
        # ``selector.evict_expired`` drops entries past the window. Temporal
        # history is purged only for those truly-expired entries.
        expired = self.selector.evict_expired(frame)
        if expired:
            self.temporal.purge(expired)
        for nid in expired:
            self._last_seen.pop(nid, None)
            self._first_unseen.pop(nid, None)

        # Build final node list: ee + slots (ordered by slot_id) + padding.
        ee_node = nodes.get("ee")
        slot_to_node: Dict[int, Node] = {}
        for ent_id in selected_ids:
            n = nodes.get(ent_id)
            if n is None:
                continue
            sa = assignments[ent_id]
            n.slot_id = sa.slot_id
            n.entity_id = ent_id
            n.valid_mask = True
            n.reset_flag = sa.reset_flag
            slot_to_node[sa.slot_id] = n

        ordered: List[Node] = []
        if ee_node is not None:
            ordered.append(ee_node)
        for s in range(self.slots.n_slots):
            n = slot_to_node.get(s)
            ordered.append(n if n is not None else padding_node(s))

        graph = Graph(
            frame=frame,
            env_id=self.env_id,
            camera=cam,
            nodes=ordered,
            meta=dict(
                is_mshab=state.is_mshab,
                active_subtask=state.active_subtask_type,
                mshab_object_name=self.mshab_object_name,
                n_valid=sum(1 for n in ordered if n.valid_mask and n.node_type == "object"),
            ),
        )

        # R15: edges only for valid nodes (build_absolute_edges already iterates
        # graph.nodes; the relation functions guard against padding via the
        # valid_mask check added there).
        build_absolute_edges(graph, state, self.cfg)
        self.temporal.update(graph)
        graph.edges.extend(self.temporal.temporal_edges(graph, self.cfg))

        # Commit selection state for next frame.
        self.selector.commit(selected_ids, nodes, frame)
        return graph, masks, cam, rgb
