"""Per-frame node selection and relation orchestration.

Pipeline: build_nodes -> apply_whitelist -> classify_pair_types
-> overflow_truncate -> slot assign -> absolute + temporal edges.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .affordance import canonical_affordance_key
from .entity_identity import stable_entity_key, stable_node_id
from .schema import Edge, Graph, Node, padding_node
from .node_builder import build_nodes, classify_pair_types
from .relation_rules import build_absolute_edges
from .temporal_buffer import TemporalBuffer
from .mask_extractor import MaskAccumulator
from .selector import NodeSelector
from .slot_manager import SlotManager
from .whitelist import entity_match_key, load_whitelist, resolve_whitelist_path
from ..adapters.privileged_state import get_privileged_state

# Only physical-state relations survive on a frozen node. Spatial and
# compatibility edges would replay stale geometry, so we drop them.
_STALE_REPLAY_RELATIONS = frozenset({"contact", "grasp", "support", "contain"})


class GraphBuilder:
    def __init__(
        self,
        env,
        cfg: dict,
        *,
        env_idx: int = 0,
        env_id: str = "env",
        camera: Optional[str] = None,
        staleness_enabled: bool = True,
    ):
        self.env = env
        self.cfg = cfg
        self.env_idx = env_idx
        self.env_id = env_id
        self.camera = camera
        self.staleness_enabled = bool(staleness_enabled)

        self.temporal = TemporalBuffer(K=cfg["temporal"]["K"])
        self.selector = NodeSelector(cfg)
        self.slots = SlotManager(n_slots=int(cfg["selection"]["n_slots"]))
        self.cfg.setdefault("_affordance_selection_cache", {})

        self._whitelist_dir: Optional[str] = cfg.get("whitelist_dir")
        self._whitelist_key: Optional[Tuple[str, str]] = None

        self._last_seen: Dict[str, int] = {}
        self._first_unseen: Dict[str, int] = {}
        # Last fresh absolute edge per (src,dst,relation) -- replayed for frozen nodes.
        self._edge_history: Dict[Tuple[str, str, str], Edge] = {}
        # entity -> whitelist match key, identity-guarded (ids recycle).
        self._match_key_cache: Dict[int, Tuple[Any, Optional[str]]] = {}

    def reset_episode(self) -> None:
        self.selector.reset_episode()
        self.slots.reset_episode()
        self.temporal = TemporalBuffer(K=self.cfg["temporal"]["K"])
        self._last_seen.clear()
        self._first_unseen.clear()
        self._edge_history.clear()
        self._match_key_cache.clear()
        self.cfg.setdefault("_affordance_selection_cache", {}).clear()
        self._whitelist_key = None

    def _resolve_and_bind_whitelist(self, state) -> None:
        """Bind the whitelist for (subtask, target). Cached; rebinds on key change."""
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
        self.cfg.setdefault("_affordance_selection_cache", {}).clear()
        path = resolve_whitelist_path(self._whitelist_dir, subtask, target)
        if path is None:
            raise FileNotFoundError(
                f"per-subtask whitelist not found for subtask={subtask!r}, "
                f"target={target!r} under whitelist_dir={self._whitelist_dir!r}. "
                "Mine assets with tools/build_subtask_whitelists.py."
            )
        wl = load_whitelist(path)
        self.selector.set_whitelist(wl)
        # Push per-(subtask, target) bin edges and the per-member interaction-
        # type table into cfg so relation_rules and temporal_buffer pick them
        # up. cfg["profile"] remains the fallback for any relation the asset
        # omits; cfg["compat_norm"] (from thresholds.yaml or runtime defaults)
        # is untouched.
        self.cfg["bin_edges"] = dict(wl.bin_edges or {})
        self.cfg["interaction_types"] = {
            k: set(v) for k, v in (wl.interaction_types or {}).items()
        }
        self._whitelist_key = key

    def _entity_admitted(self, entity) -> bool:
        """Early whitelist gate for build_nodes: superset of apply_whitelist.

        Instance-level target filtering still happens in apply_whitelist, so
        this only skips entities whose match key is absent from the whitelist
        -- exactly the nodes apply_whitelist would drop unconditionally.
        """
        wl = self.selector.whitelist
        if wl is None:
            return True
        hit = self._match_key_cache.get(id(entity))
        if hit is not None and hit[0] is entity:
            key = hit[1]
        else:
            key = entity_match_key(entity)
            self._match_key_cache[id(entity)] = (entity, key)
        return wl.contains(key)

    def step(
        self, obs: dict, frame: int,
        *,
        episode_boundary: bool = False,
        seg_override=None, seg_overrides=None,
        rgb_override=None, camera_override=None, primary_camera=None,
        need_masks: bool = True,
    ) -> Tuple[Graph, MaskAccumulator, str, np.ndarray]:
        if episode_boundary:
            self.reset_episode()

        state = get_privileged_state(self.env, self.env_idx)

        # Re-bind every step: MS-HAB advances subtasks mid-episode.
        self._resolve_and_bind_whitelist(state)

        nodes, masks, cam, rgb = build_nodes(
            obs, state,
            camera=self.camera,
            seg_override=seg_override,
            seg_overrides=seg_overrides,
            rgb_override=rgb_override,
            camera_override=camera_override,
            primary_camera=primary_camera,
            need_masks=need_masks,
            # Recording paths keep full masks/nodes for overlays; the training
            # hot path skips node construction for never-admissible entities.
            admit=None if need_masks else self._entity_admitted,
        )

        # Whitelist admission first, then episode-scoped persistence: a node
        # that was ever seen (post-whitelist) and is still admissible stays as
        # a frozen snapshot for the rest of the episode. Non-whitelisted
        # entities are never persisted.
        active_target_node_id: Optional[str] = None
        if state.active_obj is not None:
            # Fail open if active-object resolution fell back to the merged
            # MS-HAB handle itself. Its node id is like ``actor:obj_0``, which
            # matches no visible segmentation node and would drop every target
            # instance from the graph.
            active_obj_merged = getattr(state, "active_obj_merged", None)
            resolution_fell_back = (
                active_obj_merged is not None
                and state.active_obj is active_obj_merged
            )
            if not resolution_fell_back:
                try:
                    active_target_node_id = stable_node_id(state.active_obj)
                except Exception:
                    active_target_node_id = None
        nodes = self.selector.apply_whitelist(
            nodes, active_target_node_id=active_target_node_id,
        )
        if self.staleness_enabled:
            nodes = self.selector.merge_persistent(nodes, frame)

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

        classify_pair_types(nodes, self.cfg)

        selected_ids = self.selector.overflow_truncate(nodes)
        assignments = self.slots.assign(selected_ids)

        expired = self.selector.evict_expired(frame)
        if expired:
            self.temporal.purge(expired)
        for nid in expired:
            self._last_seen.pop(nid, None)
            self._first_unseen.pop(nid, None)
            for key in [k for k in self._edge_history if nid in k[:2]]:
                del self._edge_history[key]

        # Final order: ee, then slots (by slot_id), padded.
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

        build_absolute_edges(graph, state, self.cfg)
        if self.staleness_enabled:
            self._attach_stale_edges(graph, frame)
        self.temporal.update(graph)
        graph.edges.extend(self.temporal.temporal_edges(graph, self.cfg))

        self.selector.commit(nodes, frame)
        return graph, masks, cam, rgb

    def _attach_stale_edges(self, graph: Graph, frame: int) -> None:
        """Cache fresh edges; replay last observed edge (tagged stale) for frozen nodes."""
        by_id = {n.node_id: n for n in graph.nodes if n.valid_mask}
        fresh_ids = {nid for nid, n in by_id.items() if not n.frozen_pose}
        stale_ids = set(by_id) - fresh_ids

        # Only refresh cache for edges between two fresh nodes.
        for key in list(self._edge_history):
            if key[0] in fresh_ids and key[1] in fresh_ids:
                del self._edge_history[key]
        for edge in graph.edges:
            if edge.temporal or edge.stale:
                continue
            if edge.relation not in _STALE_REPLAY_RELATIONS:
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
