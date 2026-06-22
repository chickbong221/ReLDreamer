"""Per-frame orchestration tying the pipeline together.

    GraphBuilder(env, cfg).step(obs, frame) -> (Graph, masks, camera, rgb)

The builder owns the TemporalBuffer so temporal labels persist across frames,
and tracks ``steps_since_seen`` (tau_i) for persistent nodes.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .schema import Graph, Node
from .node_builder import build_nodes
from .relation_rules import build_absolute_edges
from .temporal_buffer import TemporalBuffer
from .mask_extractor import MaskAccumulator
from .persistence import PersistentNodeRegistry
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
        self._last_seen: Dict[str, int] = {}     # node_id -> frame last visible
        self._first_unseen: Dict[str, int] = {}  # node_id -> frame first appeared unseen
        # MS-HAB-only: persistence + N-cap registry.
        pcfg = cfg.get("persistence") or {}
        self.registry = PersistentNodeRegistry(
            n_max=pcfg.get("n_max", 6),
            w_keep=pcfg.get("w_keep", 3),
            w_manip=pcfg.get("w_manip", 5),
        )

    def step(
        self, obs: dict, frame: int,
        *,
        seg_override=None, rgb_override=None, camera_override=None,
    ) -> Tuple[Graph, MaskAccumulator, str, np.ndarray]:
        state = get_privileged_state(
            self.env,
            self.env_idx,
            mshab_object_name=self.mshab_object_name,
        )

        nodes, masks, cam, rgb = build_nodes(
            obs,
            state,
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

        # MS-HAB: inject retained-but-invisible objects (frozen pose). Non-MS-HAB
        # envs are fixed-camera / fixed-object-set and skip this entirely.
        if state.is_mshab:
            nodes = self.registry.merge_retained(nodes, frame)

        # Update tau_i (steps_since_seen) for every node.
        for node_id, node in nodes.items():
            if node.visible:
                self._last_seen[node_id] = frame
                node.steps_since_seen = 0
            else:
                # If never seen, anchor age to when the node first appeared.
                # Use frame+1 (>0) to distinguish from "currently visible".
                if node_id not in self._last_seen:
                    first = self._first_unseen.setdefault(node_id, frame)
                    node.steps_since_seen = max(1, frame - first + 1)
                else:
                    node.steps_since_seen = frame - self._last_seen[node_id]

        # MS-HAB only: snapshot newly visible objects, rank tiers, drop the
        # overflow under N_max.
        evicted: set = set()
        if state.is_mshab:
            self.registry.snapshot_visible(nodes, frame)
            evicted = self.registry.rank_and_cap(nodes, state, self.cfg)
            for nid in evicted:
                nodes.pop(nid, None)
                self._last_seen.pop(nid, None)
                self._first_unseen.pop(nid, None)
            self.registry.drop(evicted)
            self.temporal.purge(evicted)

        graph = Graph(
            frame=frame,
            env_id=self.env_id,
            camera=cam,
            nodes=list(nodes.values()),
            meta=dict(
                is_mshab=state.is_mshab,
                active_subtask=state.active_subtask_type,
                mshab_object_name=self.mshab_object_name,
            ),
        )

        # Absolute then temporal edges.
        build_absolute_edges(graph, state, self.cfg)
        if state.is_mshab:
            self.registry.record_recency(graph)
        self.temporal.update(graph)
        graph.edges.extend(self.temporal.temporal_edges(graph, self.cfg))

        return graph, masks, cam, rgb
