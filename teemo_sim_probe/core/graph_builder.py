"""Per-frame segmentation, whitelist selection, and relation orchestration.

    GraphBuilder(env, cfg).step(obs, frame, episode_boundary=False)
        -> (Graph, masks, camera, rgb)

Pipeline:

    build_nodes (current visible segmentation, non-robot entities)
    -> selector.merge_persistent    (k-frame identity-keyed retention)
    -> selector.apply_whitelist     (hard per-subtask eligibility gate)
    -> tau_i bookkeeping            (steps_since_seen)
    -> classify_pair_types          (whitelist-role / affordance vocabulary)
    -> selector.overflow_truncate   (role, distance, node_id)
    -> slot_manager.assign          (identity-keyed, reset_flag)
    -> build_absolute_edges + temporal edges (valid nodes only)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .affordance import canonical_affordance_key
from .entity_identity import stable_entity_key
from .schema import Edge, Graph, Node, padding_node
from .node_builder import build_nodes, classify_pair_types
from .relation_rules import build_absolute_edges
from .temporal_buffer import TemporalBuffer
from .mask_extractor import MaskAccumulator
from .selector import NodeSelector
from .slot_manager import SlotManager
from .whitelist import load_whitelist, resolve_whitelist_path
from ..adapters.privileged_state import get_privileged_state

class GraphBuilder:
    def __init__(
        self,
        env,
        cfg: dict,
        *,
        env_idx: int = 0,
        env_id: str = "env",
        camera: Optional[str] = None,
    ):
        self.env = env
        self.cfg = cfg
        self.env_idx = env_idx
        self.env_id = env_id
        self.camera = camera

        self.temporal = TemporalBuffer(K=cfg["temporal"]["K"])
        self.selector = NodeSelector(cfg)
        self.slots = SlotManager(n_slots=int(cfg["selection"]["n_slots"]))

        # ``whitelist_dir`` may be absolute or relative to the config file.
        self._whitelist_dir: Optional[str] = cfg.get("whitelist_dir")
        self._whitelist_key: Optional[Tuple[str, str]] = None

        self._last_seen: Dict[str, int] = {}
        self._first_unseen: Dict[str, int] = {}
        # Last fresh absolute relation per key. Frozen nodes reuse these edges
        # with explicit stale metadata instead of mixing current ee state with
        # an old object pose.
        self._edge_history: Dict[Tuple[str, str, str], Edge] = {}

    # ---------------------------------------------------------------- reset
    def reset_episode(self) -> None:
        self.selector.reset_episode()
        self.slots.reset_episode()
        self.temporal = TemporalBuffer(K=self.cfg["temporal"]["K"])
        self._last_seen.clear()
        self._first_unseen.clear()
        self._edge_history.clear()
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
        if state.active_handle_link is not None:
            target = stable_entity_key(state.active_handle_link)
        else:
            canonical = (
                canonical_affordance_key(state.active_obj_id)
                if state.active_obj_id else None
            )
            target = f"actor:{canonical}" if canonical else None
        if subtask is None or target is None:
            raise RuntimeError(
                "whitelist selection requires an active subtask type and "
                f"target key; got subtask={subtask!r}, "
                f"active_obj_id={state.active_obj_id!r}, "
                f"active_handle_link={state.active_handle_link!r}. Probe must "
                "run inside an MS-HAB-like env."
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

        state = get_privileged_state(self.env, self.env_idx)

        # Done every step so MS-HAB subtask advancement within
        # an episode swaps the gate cleanly (idempotent when (subtask, target)
        # is unchanged).
        self._resolve_and_bind_whitelist(state)

        nodes, masks, cam, rgb = build_nodes(
            obs, state,
            camera=self.camera,
            seg_override=seg_override,
            rgb_override=rgb_override,
            camera_override=camera_override,
        )

        # Identity-keyed persistence merge -- re-inject occluded entities
        # whose last_seen age is within k_persist.
        nodes = self.selector.merge_persistent(nodes, frame)

        # Hard per-subtask whitelist gate. Everything failing the
        # gate is dropped here; only the ee node and whitelisted entities
        # survive into the slot assignment stage.
        nodes = self.selector.apply_whitelist(nodes)

        # Visibility age is tracked only for eligible semantic entities.
        for nid, n in nodes.items():
            if n.node_type == "ee":
                continue
            if n.visible:
                self._last_seen[nid] = frame
                n.steps_since_seen = 0
            elif nid not in self._last_seen:
                first = self._first_unseen.setdefault(nid, frame)
                n.steps_since_seen = max(1, frame - first + 1)
            else:
                n.steps_since_seen = frame - self._last_seen[nid]

        # Relation vocabulary now uses affordances plus whitelist roles. A
        # handle is an ordinary object and is interactive only when its role or
        # affordance says so.
        classify_pair_types(nodes, self.cfg)

        # Deterministic role-aware capacity; distance and node id break ties.
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
            for key in [k for k in self._edge_history if nid in k[:2]]:
                del self._edge_history[key]

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
                n_valid=sum(1 for n in ordered if n.valid_mask and n.node_type == "object"),
            ),
        )

        # R15: edges only for valid nodes (build_absolute_edges already iterates
        # graph.nodes; the relation functions guard against padding via the
        # valid_mask check added there).
        build_absolute_edges(graph, state, self.cfg)
        self._attach_stale_edges(graph, frame)
        self.temporal.update(graph)
        graph.edges.extend(self.temporal.temporal_edges(graph, self.cfg))

        # Commit selection state for next frame.
        self.selector.commit(nodes, frame)
        return graph, masks, cam, rgb

    def _attach_stale_edges(self, graph: Graph, frame: int) -> None:
        """Cache fresh relations and restore last observations for stale nodes."""
        by_id = {n.node_id: n for n in graph.nodes if n.valid_mask}
        fresh_ids = {nid for nid, n in by_id.items() if not n.frozen_pose}
        stale_ids = set(by_id) - fresh_ids

        # Replace cached relations only for pairs whose endpoints are both
        # fresh. Pairs touching stale nodes keep their last fully observed edge.
        for key in list(self._edge_history):
            if key[0] in fresh_ids and key[1] in fresh_ids:
                del self._edge_history[key]
        for edge in graph.edges:
            if edge.temporal or edge.stale:
                continue
            if edge.src in fresh_ids and edge.dst in fresh_ids:
                key = (edge.src, edge.dst, edge.relation)
                self._edge_history[key] = replace(
                    edge, stale=False, observed_frame=frame, age=0,
                )

        existing = {(e.src, e.dst, e.relation) for e in graph.edges}
        for key, cached in self._edge_history.items():
            if key in existing:
                continue
            if cached.src not in by_id or cached.dst not in by_id:
                continue
            if cached.src not in stale_ids and cached.dst not in stale_ids:
                continue
            observed = cached.observed_frame
            age = max(1, frame - observed) if observed is not None else 1
            graph.edges.append(replace(cached, stale=True, age=age))
