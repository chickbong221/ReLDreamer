"""Online oracle-graph plumbing for SAC.

Owns one ``GraphBuilder`` per parallel env and turns each per-env ``Graph``
into a fixed-shape batched tensor dict that slots straight into the replay
buffer next to ``rgb`` / ``state``.

Segmentation is sliced from ``env.unwrapped._last_obs`` (set by the underlying
env's ``step`` / ``reset``) rather than re-fetched per env. Calling
``env.unwrapped.get_obs()`` would rerun ``get_info`` -> MS-HAB ``evaluate``,
which mutates ``subtask_pointer`` / ``subtask_steps_left`` / cumulative force,
and would also re-render + CUDA-sync once per env.
"""

from __future__ import annotations

from copy import copy as _shallow_copy
from typing import Dict, List, Optional

import numpy as np
import torch

from teemo_sim_probe.adapters.privileged_state import (
    begin_frame_cache,
    end_frame_cache,
)
from teemo_sim_probe.adapters.graph_pack import GRAPH_KEYS, pack_graph
from teemo_sim_probe.adapters.graph_vocab import (
    EdgeVocab,
    NodeVocab,
    build_edge_vocab,
    build_node_vocab,
)
from teemo_sim_probe.configs.loader import load_config as load_teemo_config
from teemo_sim_probe.core.graph_builder import GraphBuilder


def _verify_whitelist_coverage(env, whitelist_dir: str) -> None:
    """Fail at startup if any object-target plan lacks a mined whitelist.

    This catches the common split mismatch early, for example training with
    train-mined whitelists while eval uses val task plans. Only pick/place are
    checked here because their runtime target is exactly actor:<obj>. Open and
    close bind through live handle links and should fail loudly at runtime if
    the corresponding link whitelist is absent.
    """
    from teemo_sim_probe.core.affordance import canonical_affordance_key
    from teemo_sim_probe.core.whitelist import resolve_whitelist_path

    base = getattr(env, "unwrapped", env)
    plans_by_bci = getattr(base, "build_config_idx_to_task_plans", None)
    if plans_by_bci is None:
        return

    groups = (
        plans_by_bci.values() if hasattr(plans_by_bci, "values") else plans_by_bci
    )
    missing = set()
    checked = set()
    for plans in groups:
        for plan in plans:
            for subtask in getattr(plan, "subtasks", []) or []:
                st_type = getattr(subtask, "type", None)
                if st_type not in {"pick", "place"}:
                    continue
                obj_id = getattr(subtask, "obj_id", None)
                if not obj_id:
                    continue
                key = canonical_affordance_key(str(obj_id))
                if not key:
                    continue
                pair = (str(st_type), key)
                if pair in checked:
                    continue
                checked.add(pair)
                target = f"actor:{key}"
                if resolve_whitelist_path(whitelist_dir, str(st_type), target) is None:
                    missing.add(pair)

    if missing:
        listing = ", ".join(f"{st}:{key}" for st, key in sorted(missing))
        raise FileNotFoundError(
            f"graph: {len(missing)} object-target whitelist(s) missing under "
            f"{whitelist_dir!r}: {listing}. Mine them with "
            "tools/build_subtask_whitelists.py for the active mshab_split/"
            "mshab_eval_split before training."
        )


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
        cameras: List[str],
        primary_camera: str,
    ):
        self.env = env
        self.num_envs = int(num_envs)
        self.node_vocab = node_vocab
        self.edge_vocab = edge_vocab
        self.n_max = int(n_max)
        self.e_max = int(e_max)
        self.k_soft = float(k_soft)
        self.cameras = list(cameras)
        if primary_camera not in self.cameras:
            raise ValueError(
                f"primary_camera={primary_camera!r} not in cameras={self.cameras}"
            )
        self.primary_camera = primary_camera
        self.builders = []
        for i in range(self.num_envs):
            cfg_i = _shallow_copy(teemo_cfg)
            cfg_i["_affordance_selection_cache"] = {}
            self.builders.append(
                GraphBuilder(env, cfg_i, env_idx=i, env_id=f"env{i}",
                             camera=primary_camera)
            )
        self._frames = np.zeros(self.num_envs, dtype=np.int64)
        self.record_env0 = False
        self.last_env0_graph = None
        self.last_env0_masks = None
        self._cams_checked = False

    @property
    def obs_spec_shapes(self) -> Dict[str, tuple]:
        """Per-env shapes for each graph key, consumed by the replay buffer."""
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

    def _pack_one(
        self, env_idx: int, episode_boundary: bool,
        seg_by_cam: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        need_masks = env_idx == 0 and self.record_env0
        graph, masks, _, _ = self.builders[env_idx].step(
            {},
            int(self._frames[env_idx]),
            episode_boundary=episode_boundary,
            seg_overrides=seg_by_cam,
            rgb_override=None,
            primary_camera=self.primary_camera,
            need_masks=need_masks,
        )
        if env_idx == 0 and self.record_env0:
            self.last_env0_graph = graph
            self.last_env0_masks = masks
        self._frames[env_idx] += 1
        return pack_graph(
            graph, self.node_vocab, self.edge_vocab,
            n_max=self.n_max, e_max=self.e_max, k_soft=self.k_soft,
        )

    def read_rgb_env0(self) -> np.ndarray:
        rgb = self.env.unwrapped._last_obs["sensor_data"][self.primary_camera]["rgb"][0]
        return rgb.detach().cpu().numpy().astype(np.uint8)

    def _read_batched_segs(self) -> Dict[str, np.ndarray]:
        """Return ``{cam: [N, H, W]}`` for every configured camera."""
        sensor_data = self.env.unwrapped._last_obs["sensor_data"]
        if not self._cams_checked:
            for cam in self.cameras:
                if cam not in sensor_data:
                    raise KeyError(
                        f"graph: camera {cam!r} not in sensor_data "
                        f"(available: {list(sensor_data)}). Check obs_mode and "
                        "sensor configs render this camera."
                    )
                if "segmentation" not in sensor_data[cam]:
                    raise KeyError(
                        f"graph: camera {cam!r} has no 'segmentation' in "
                        f"_last_obs; obs_mode must include segmentation."
                    )
            self._cams_checked = True
        out: Dict[str, np.ndarray] = {}
        for cam in self.cameras:
            seg = sensor_data[cam]["segmentation"].squeeze(-1)
            out[cam] = seg.detach().cpu().numpy()
        return out

    def step(
        self, done_mask: Optional[torch.Tensor], device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if done_mask is not None:
            done_np = done_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        else:
            done_np = np.zeros(self.num_envs, dtype=bool)
        segs_by_cam = self._read_batched_segs()
        begin_frame_cache()
        try:
            packed = [
                self._pack_one(
                    i, bool(done_np[i]),
                    {cam: segs_by_cam[cam][i] for cam in self.cameras},
                )
                for i in range(self.num_envs)
            ]
        finally:
            end_frame_cache()
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

    _verify_whitelist_coverage(env, teemo_cfg["whitelist_dir"])
    node_vocab = build_node_vocab(teemo_cfg["whitelist_dir"])
    edge_vocab = build_edge_vocab()

    n_slots = int(teemo_cfg["selection"]["n_slots"])
    n_max = n_slots + 1
    e_max = int(graph_cfg.get("e_max", 256))
    k_soft = float(
        graph_cfg.get("k_soft", teemo_cfg["selection"].get("k_persist", 5))
    )

    cameras = graph_cfg.get("cameras")
    if not cameras:
        cameras = [graph_cfg.get("camera", "fetch_head")]
    primary_camera = graph_cfg.get("primary_camera") or cameras[0]

    return GraphObsBuilder(
        env,
        num_envs=num_envs,
        teemo_cfg=teemo_cfg,
        node_vocab=node_vocab,
        edge_vocab=edge_vocab,
        n_max=n_max,
        e_max=e_max,
        k_soft=k_soft,
        cameras=list(cameras),
        primary_camera=primary_camera,
    )
