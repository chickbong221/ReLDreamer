"""Per-frame orchestration (R1-R16).

    GraphBuilder(env, cfg).step(obs, frame, episode_boundary=False)
        -> (Graph, masks, camera, rgb)

Pipeline:

    R1-R6 build_nodes (background / robot / goals / static-scene / area filter)
    R4    E_domain hard filter on visible candidates (drops out-of-vocab)
    R6    classify_pair_types  (relation vocabulary specificity)
    R8    selector.merge_persistent (k-frame persistence)
    R7    selector.expand_local_contact (V_{t-1} contact one hop)
    R10   selector.score (continuous rule score)
    R11   selector.topk_with_refresh (refresh quota)
    R12   slot_manager.assign (identity-keyed, reset_flag)
    R11   pad to n_slots
    R15   build_absolute_edges + temporal edges (valid nodes only)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .schema import Graph, Node, padding_node
from .node_builder import build_nodes, classify_pair_types
from .relation_rules import build_absolute_edges
from .temporal_buffer import TemporalBuffer
from .mask_extractor import MaskAccumulator
from .selector import NodeSelector
from .slot_manager import SlotManager
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
        self.selector = NodeSelector(
            cfg, cfg["e_domain_set"], cfg.get("profile_name", "tabletop")
        )
        self.slots = SlotManager(n_slots=int(cfg["selection"]["n_slots"]))

        self._last_seen: Dict[str, int] = {}
        self._first_unseen: Dict[str, int] = {}

    # ---------------------------------------------------------------- reset
    def reset_episode(self) -> None:
        self.selector.reset_episode()
        self.slots.reset_episode()
        self.temporal = TemporalBuffer(K=self.cfg["temporal"]["K"])
        self._last_seen.clear()
        self._first_unseen.clear()

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

        # R4: E_domain hard filter for visible candidates. The ee node is always
        # kept. Empty domain (missing asset) accepts everything.
        e_domain = self.cfg["e_domain_set"]
        nodes = {
            nid: n for nid, n in nodes.items()
            if n.node_type == "ee" or e_domain.contains(n)
        }

        # R6: relation vocabulary classification (uses affordance set).
        classify_pair_types(nodes, self.cfg)

        # R8: identity-keyed persistence merge.
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

        # R7: local-contact exception (uses V_{t-1} -- one-frame lag, acyclic).
        nodes = self.selector.expand_local_contact(
            nodes, state, self.selector.prev_selected,
        )

        # Re-classify so newly added local-contact nodes get a pair_type.
        classify_pair_types(nodes, self.cfg)

        # R10 + R11: score and select.
        scores = self.selector.score(nodes, state)
        selected_ids = self.selector.topk_with_refresh(
            scores, self.selector.prev_selected
        )

        # R12: slot assignment.
        assignments = self.slots.assign(selected_ids)

        # Forget snapshots / housekeeping for un-selected entities.
        unselected = set(nodes.keys()) - set(selected_ids) - {"ee"}
        self.selector.evict(unselected)
        self.temporal.purge(unselected)
        for nid in unselected:
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
