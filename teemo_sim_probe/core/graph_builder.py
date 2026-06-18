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
    ):
        self.env = env
        self.cfg = cfg
        self.env_idx = env_idx
        self.env_id = env_id
        self.camera = camera
        self.include_goals = include_goals
        self.temporal = TemporalBuffer(K=cfg["temporal"]["K"])
        self._last_seen: Dict[str, int] = {}     # node_id -> frame last visible
        self._first_unseen: Dict[str, int] = {}  # node_id -> frame first appeared unseen

    def step(
        self, obs: dict, frame: int
    ) -> Tuple[Graph, MaskAccumulator, str, np.ndarray]:
        state = get_privileged_state(self.env, self.env_idx)

        nodes, masks, cam, rgb = build_nodes(
            obs,
            state,
            camera=self.camera,
            include_goals=self.include_goals,
            min_pixels=self.cfg.get("node", {}).get("min_pixels", 32),
            min_area_ratio=self.cfg.get("node", {}).get("min_area_ratio", 0.0005),
        )

        # Update tau_i (steps_since_seen) for every node.
        for node_id, node in nodes.items():
            if node.visible:
                self._last_seen[node_id] = frame
                node.steps_since_seen = 0
            else:
                # If never seen, anchor age to when the node first appeared.
                if node_id not in self._last_seen:
                    self._first_unseen.setdefault(node_id, frame)
                    node.steps_since_seen = frame - self._first_unseen[node_id]
                else:
                    node.steps_since_seen = frame - self._last_seen[node_id]

        graph = Graph(
            frame=frame,
            env_id=self.env_id,
            camera=cam,
            nodes=list(nodes.values()),
            meta=dict(
                is_mshab=state.is_mshab,
                active_subtask=state.active_subtask_type,
            ),
        )

        # Absolute then temporal edges.
        build_absolute_edges(graph, state, self.cfg)
        self.temporal.update(graph)
        graph.edges.extend(self.temporal.temporal_edges(graph, self.cfg))

        return graph, masks, cam, rgb
