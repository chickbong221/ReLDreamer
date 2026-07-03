"""Online oracle-graph plumbing for SAC.

Owns one ``GraphBuilder`` per parallel env and turns each per-env ``Graph``
into a fixed-shape batched tensor dict that slots straight into the replay
buffer next to ``rgb`` / ``state``.

Segmentation is read via ``read_unwrapped_sensor`` so the policy obs can stay
as flat RGB while the graph pipeline still gets the segmentation map.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from teemo_sim_probe.adapters.graph_pack import GRAPH_KEYS, pack_graph
from teemo_sim_probe.adapters.graph_vocab import (
    EdgeVocab,
    NodeVocab,
    build_edge_vocab,
    build_node_vocab,
)
from teemo_sim_probe.configs.loader import load_config as load_teemo_config
from teemo_sim_probe.core.graph_builder import GraphBuilder
from teemo_sim_probe.core.mask_extractor import read_unwrapped_sensor


class GraphObsBuilder:
    """One GraphBuilder per env. Emits packed batched tensors per frame."""

    def __init__(
        self,
        env,
        *,
        num_envs: int,
        teemo_cfg: dict,
        node_vocab: NodeVocab,
        edge_vocab: EdgeVocab,
        n_max: int,
        e_max: int,
        k_soft: float,
        camera: str,
    ):
        self.env = env
        self.num_envs = int(num_envs)
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab
        self.n_max = int(n_max)
        self.e_max = int(e_max)
        self.k_soft = float(k_soft)
        self.camera = camera
        self.builders = [
            GraphBuilder(env, teemo_cfg, env_idx=i, env_id=f"env{i}", camera=camera)
            for i in range(self.num_envs)
        ]
        self._frames = np.zeros(self.num_envs, dtype=np.int64)

    @property
    def obs_spec_shapes(self) -> Dict[str, tuple]:
        """Per-env shapes for build_obs_spec. Includes dtypes via dummy zeros."""
        return {
            "graph_node_ids":     (self.n_max,),
            "graph_node_valid":   (self.n_max,),
            "graph_node_ee_mask": (self.n_max,),
            "graph_node_conf":    (self.n_max,),
            "graph_edge_src":     (self.e_max,),
            "graph_edge_dst":     (self.e_max,),
            "graph_edge_pred":    (self.e_max,),
            "graph_edge_valid":   (self.e_max,),
        }

    def _pack_one(self, env_idx: int, episode_boundary: bool) -> Dict[str, np.ndarray]:
        seg, _depth, rgb = read_unwrapped_sensor(
            self.env, self.camera, env_idx=env_idx,
        )
        graph, _, _, _ = self.builders[env_idx].step(
            {},
            int(self._frames[env_idx]),
            episode_boundary=episode_boundary,
            seg_override=seg,
            rgb_override=rgb,
            camera_override=self.camera,
        )
        self._frames[env_idx] += 1
        return pack_graph(
            graph, self.node_vocab, self.edge_vocab,
            n_max=self.n_max, e_max=self.e_max, k_soft=self.k_soft,
        )

    def step(
        self, done_mask: Optional[torch.Tensor], device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if done_mask is not None:
            done_np = done_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        else:
            done_np = np.zeros(self.num_envs, dtype=bool)
        packed = [self._pack_one(i, bool(done_np[i])) for i in range(self.num_envs)]
        out: Dict[str, torch.Tensor] = {}
        for k in GRAPH_KEYS:
            arr = np.stack([p[k] for p in packed], axis=0)
            out[k] = torch.as_tensor(arr, device=device)
        return out

    def reset(self, device: torch.device) -> Dict[str, torch.Tensor]:
        self._frames[:] = 0
        return self.step(
            done_mask=torch.ones(self.num_envs, dtype=torch.bool), device=device,
        )


def build_graph_obs(
    env,
    graph_cfg: dict,
    *,
    num_envs: int,
) -> Optional[GraphObsBuilder]:
    """Return a GraphObsBuilder or None when graph obs is disabled."""
    if not bool(graph_cfg.get("enabled", False)):
        return None

    teemo_cfg = load_teemo_config(
        graph_cfg.get("profile", "tabletop"),
        path=graph_cfg.get("thresholds_path"),
    )
    if "n_slots" in graph_cfg:
        teemo_cfg["selection"]["n_slots"] = int(graph_cfg["n_slots"])
    if "k_persist" in graph_cfg:
        teemo_cfg["selection"]["k_persist"] = int(graph_cfg["k_persist"])
    if graph_cfg.get("whitelist_dir"):
        teemo_cfg["whitelist_dir"] = graph_cfg["whitelist_dir"]
    if teemo_cfg.get("whitelist_dir") is None:
        raise ValueError(
            "graph: whitelist_dir is not set in the loaded teemo config; "
            "set graph.whitelist_dir or configure teemo_sim_probe/configs/"
            "thresholds.yaml."
        )

    node_vocab = build_node_vocab(teemo_cfg["whitelist_dir"])
    edge_vocab = build_edge_vocab()

    n_slots = int(teemo_cfg["selection"]["n_slots"])
    n_max = n_slots + 1
    e_max = int(graph_cfg.get("e_max", 256))
    k_soft = float(
        graph_cfg.get("k_soft", teemo_cfg["selection"].get("k_persist", 5))
    )

    return GraphObsBuilder(
        env,
        num_envs=num_envs,
        teemo_cfg=teemo_cfg,
        node_vocab=node_vocab,
        edge_vocab=edge_vocab,
        n_max=n_max,
        e_max=e_max,
        k_soft=k_soft,
        camera=graph_cfg.get("camera", "fetch_head"),
    )
