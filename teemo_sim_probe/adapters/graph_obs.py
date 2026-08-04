"""Online graph plumbing shared by every trainer.

Owns one ``GraphBuilder`` per parallel env and turns each per-env ``Graph`` into
a fixed-shape batched tensor dict.

Segmentation and depth are sliced from ``env.unwrapped._last_obs`` (set by the
underlying env's ``step`` / ``reset``) rather than re-fetched per env. Calling
``env.unwrapped.get_obs()`` would rerun ``get_info`` -> MS-HAB ``evaluate``,
which mutates ``subtask_pointer`` / ``subtask_steps_left`` / cumulative force,
and would also re-render + CUDA-sync once per env.
"""

from __future__ import annotations

from copy import copy as _shallow_copy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from .privileged_state import (
    begin_frame_cache,
    clear_privileged_state_caches,
    end_frame_cache,
)
from .graph_pack import GRAPH_KEYS, pack_graph
from .graph_vocab import GraphVocab, build_graph_vocab
from ..configs.loader import load_config as load_teemo_config
from ..core.graph_builder import GraphBuilder

# ManiSkill caches one native PhysX GPU query per (obj1.name + obj2.name) pair
# and never evicts. MS-HAB partial resets recreate merged actors under fresh
# names, so stale entries accumulate (~122 KB each). Queries rebuild lazily on
# the next contact lookup, so clearing past this cap is always safe.
_CONTACT_QUERY_CAP = 2048

_DTYPES: Dict[str, np.dtype] = {
    "graph_node_ent": np.uint16,
    "graph_node_vis": np.uint8,
    "graph_node_valid": np.uint8,
    "graph_node_feat": np.float32,
    "graph_edge_src": np.uint8,
    "graph_edge_dst": np.uint8,
    "graph_edge_rel": np.uint8,
    "graph_edge_abs": np.uint8,
    "graph_edge_temp": np.uint8,
    "graph_edge_temp_mask": np.uint8,
    "graph_edge_valid": np.uint8,
    "graph_n_nodes": np.int32,
    "graph_n_edges": np.int32,
    "graph_target_ent": np.uint16,
}


def _verify_whitelist_coverage(env, whitelist_dir: str) -> None:
    """Fail at startup if any object-target plan lacks a mined whitelist.

    Catches the common split mismatch early, for example training with
    train-mined whitelists while eval uses val task plans. Only pick/place are
    checked because their runtime target is exactly actor:<obj>; open and close
    bind through live handle links and fail loudly at runtime instead.
    """
    from ..core.affordance import canonical_affordance_key
    from ..core.whitelist import resolve_whitelist_path

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
    """One GraphBuilder per env. Emits packed batched arrays per frame."""

    def __init__(
        self,
        env,
        *,
        num_envs: int,
        teemo_cfg: dict,
        vocab: GraphVocab,
        n_max: int,
        e_max: int,
        cameras: List[str],
        primary_camera: str,
        bypass_teemo: bool = False,
        staleness_enabled: bool = True,
    ):
        self.env = env
        self.num_envs = int(num_envs)
        self.vocab = vocab
        self.n_max = int(n_max)
        self.e_max = int(e_max)
        self.bypass_teemo = bool(bypass_teemo)
        self.staleness_enabled = bool(staleness_enabled)
        self.cameras = list(cameras)
        if primary_camera not in self.cameras:
            raise ValueError(
                f"primary_camera={primary_camera!r} not in cameras={self.cameras}"
            )
        self.primary_camera = primary_camera
        self.n_feat = len(self.cameras)
        self.builders = []
        for i in range(self.num_envs):
            cfg_i = _shallow_copy(teemo_cfg)
            cfg_i["_affordance_selection_cache"] = {}
            self.builders.append(
                GraphBuilder(env, cfg_i, env_idx=i, env_id=f"env{i}",
                             camera=primary_camera,
                             camera_order=self.cameras,
                             staleness_enabled=self.staleness_enabled)
            )
        self._frames = np.zeros(self.num_envs, dtype=np.int64)
        # Last packed arrays per env, re-emitted on terminal frames whose
        # sensors already belong to the next episode.
        self._last_packed: List[Optional[Dict[str, np.ndarray]]] = [
            None for _ in range(self.num_envs)
        ]
        # Env indices whose latest graph + primary-cam masks are cached for
        # offline rendering. Empty in the hot path.
        self.record_env_indices: Set[int] = set()
        self.last_graph_by_env: Dict[int, Any] = {}
        self.last_masks_by_env: Dict[int, Any] = {}
        self._cams_checked = False
        self._scene_cache_signature = None
        self._cpu_buffers: Dict[Tuple[str, str], torch.Tensor] = {}

    @property
    def obs_spec_shapes(self) -> Dict[str, tuple]:
        """Per-env shapes for each graph key."""
        return {
            "graph_node_ent":       (self.n_max,),
            "graph_node_vis":       (self.n_max,),
            "graph_node_valid":     (self.n_max,),
            "graph_node_feat":      (self.n_max, self.n_feat),
            "graph_edge_src":       (self.e_max,),
            "graph_edge_dst":       (self.e_max,),
            "graph_edge_rel":       (self.e_max,),
            "graph_edge_abs":       (self.e_max,),
            "graph_edge_temp":      (self.e_max,),
            "graph_edge_temp_mask": (self.e_max,),
            "graph_edge_valid":     (self.e_max,),
            "graph_n_nodes":        (),
            "graph_n_edges":        (),
            "graph_target_ent":     (),
        }

    @property
    def obs_spec_dtypes(self) -> Dict[str, np.dtype]:
        return dict(_DTYPES)

    def cache_stats(self) -> Dict[str, int]:
        """Sizes of every container that could grow without bound, for leak
        triage: a linear counter here names the leak directly."""
        stats: Dict[str, int] = {}
        scene = getattr(self.env.unwrapped, "scene", None)
        if scene is not None:
            d = getattr(scene, "__dict__", {})
            for key in (
                "_teemo_sidxs_cache", "_teemo_sliced_views",
                "_teemo_row_sliced_views", "_teemo_resolve_cache",
                "_teemo_per_env_seg_maps",
            ):
                v = d.get(key)
                if v is not None:
                    stats[key.replace("_teemo_", "")] = len(v)
            for key in ("pairwise_contact_queries", "actor_views"):
                v = getattr(scene, key, None)
                if v is not None:
                    stats[key] = len(v)
        stats["match_key"] = sum(len(b._match_key_cache) for b in self.builders)
        stats["edge_history"] = sum(len(b._edge_history) for b in self.builders)
        stats["registry"] = sum(len(b.registry) for b in self.builders)
        stats["temporal_values"] = sum(
            len(b.temporal._values) for b in self.builders
        )
        stats["overflow_drops"] = sum(
            b.registry.overflow_drops for b in self.builders
        )
        return stats

    def _zero_pack(self) -> Dict[str, np.ndarray]:
        return {
            k: np.zeros(shape, dtype=_DTYPES[k])
            for k, shape in self.obs_spec_shapes.items()
        }

    def _pack_one(
        self, env_idx: int, episode_boundary: bool,
        seg_by_cam: Dict[str, np.ndarray], depth_by_cam: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        need_masks = env_idx in self.record_env_indices
        if episode_boundary:
            self._frames[env_idx] = 0
        graph, masks, _, _ = self.builders[env_idx].step(
            {},
            int(self._frames[env_idx]),
            episode_boundary=episode_boundary,
            seg_overrides=seg_by_cam,
            depth_overrides=depth_by_cam,
            rgb_override=None,
            primary_camera=self.primary_camera,
            need_masks=need_masks,
        )
        if need_masks:
            self.last_graph_by_env[env_idx] = graph
            self.last_masks_by_env[env_idx] = masks
        self._frames[env_idx] += 1
        return pack_graph(
            graph, self.vocab,
            n_max=self.n_max, e_max=self.e_max, n_feat=self.n_feat,
        )

    def read_rgb(self, env_idx: int) -> np.ndarray:
        """Primary-cam RGB (uint8, [H, W, 3]) for ``env_idx``."""
        rgb = self.env.unwrapped._last_obs["sensor_data"][self.primary_camera]["rgb"][env_idx]
        return rgb.detach().cpu().numpy().astype(np.uint8)

    def read_view(self, camera: str, env_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """RGB (uint8, [H, W, 3]) + segmentation (int64, [H, W]) for one env."""
        sensor = self.env.unwrapped._last_obs["sensor_data"][camera]
        rgb = sensor["rgb"][env_idx].detach().cpu().numpy().astype(np.uint8)
        seg = sensor["segmentation"][env_idx].squeeze(-1).detach().cpu().numpy().astype(np.int64)
        return rgb, seg

    @property
    def secondary_camera(self) -> Optional[str]:
        for cam in self.cameras:
            if cam != self.primary_camera:
                return cam
        return None

    def _read_batched(self, field: str) -> Dict[str, np.ndarray]:
        """Return ``{cam: [N, H, W]}`` for one sensor field on every camera."""
        sensor_data = self.env.unwrapped._last_obs["sensor_data"]
        if not self._cams_checked:
            for cam in self.cameras:
                if cam not in sensor_data:
                    raise KeyError(
                        f"graph: camera {cam!r} not in sensor_data "
                        f"(available: {list(sensor_data)}). Check obs_mode and "
                        "sensor configs render this camera."
                    )
                for required in ("segmentation", "depth"):
                    if required not in sensor_data[cam]:
                        raise KeyError(
                            f"graph: camera {cam!r} has no {required!r} in "
                            f"_last_obs; obs_mode must include it."
                        )
            self._cams_checked = True
        out: Dict[str, np.ndarray] = {}
        for cam in self.cameras:
            value = sensor_data[cam][field].squeeze(-1).detach()
            key = (cam, field)
            buf = self._cpu_buffers.get(key)
            if (
                buf is None
                or tuple(buf.shape) != tuple(value.shape)
                or buf.dtype != value.dtype
            ):
                buf = torch.empty(tuple(value.shape), dtype=value.dtype, device="cpu")
                self._cpu_buffers[key] = buf
            buf.copy_(value, non_blocking=False)
            out[cam] = buf.numpy()
        return out

    def _current_scene_signature(self):
        base = self.env.unwrapped
        scene = getattr(base, "scene", None)
        if scene is None:
            return None
        actors = getattr(scene, "actors", {}) or {}
        articulations = getattr(scene, "articulations", {}) or {}
        actor_ids = tuple(sorted(id(a) for a in actors.values()))
        link_ids = []
        for art in articulations.values():
            link_ids.extend(id(link) for link in getattr(art, "links", []) or [])
        return (id(scene), actor_ids, tuple(sorted(link_ids)))

    def _refresh_scene_caches_if_needed(self) -> None:
        sig = self._current_scene_signature()
        if sig is None:
            return
        if self._scene_cache_signature is None:
            self._scene_cache_signature = sig
            return
        if sig == self._scene_cache_signature:
            return
        clear_privileged_state_caches(self.env)
        self._scene_cache_signature = sig

    def _purge_contact_queries(self) -> None:
        scene = getattr(self.env.unwrapped, "scene", None)
        queries = getattr(scene, "pairwise_contact_queries", None)
        if queries is None or len(queries) <= _CONTACT_QUERY_CAP:
            return
        queries.clear()
        hashes = getattr(scene, "_pairwise_contact_query_unique_hashes", None)
        if hashes is not None:
            hashes.clear()

    def step(
        self,
        *,
        is_first: Optional[Sequence[bool]] = None,
        is_last: Optional[Sequence[bool]] = None,
    ) -> Dict[str, np.ndarray]:
        """Pack one frame for every env.

        ``is_first`` drives the per-env episode reset. ``is_last`` marks envs
        whose sensors already belong to the next episode because the vector env
        auto-reset inside ``step``; those re-emit the previous frame's arrays
        rather than a graph built from the wrong episode.
        """
        first = (
            np.asarray(is_first, dtype=bool).reshape(-1)
            if is_first is not None else np.zeros(self.num_envs, dtype=bool)
        )
        last = (
            np.asarray(is_last, dtype=bool).reshape(-1)
            if is_last is not None else np.zeros(self.num_envs, dtype=bool)
        )
        if first.any():
            self._refresh_scene_caches_if_needed()

        if self.bypass_teemo:
            packed = [self._zero_pack() for _ in range(self.num_envs)]
            return self._stack(packed)

        segs = self._read_batched("segmentation")
        depths = self._read_batched("depth")
        self._purge_contact_queries()
        begin_frame_cache(getattr(self.env.unwrapped, "scene", None))
        try:
            packed: List[Dict[str, np.ndarray]] = []
            for i in range(self.num_envs):
                cached = self._last_packed[i]
                if last[i] and cached is not None:
                    packed.append(cached)
                    continue
                out = self._pack_one(
                    i, bool(first[i]),
                    {cam: segs[cam][i] for cam in self.cameras},
                    {cam: depths[cam][i] for cam in self.cameras},
                )
                self._last_packed[i] = out
                packed.append(out)
        finally:
            end_frame_cache()
        return self._stack(packed)

    def _stack(self, packed: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            k: np.stack([p[k] for p in packed], axis=0).astype(_DTYPES[k], copy=False)
            for k in GRAPH_KEYS
        }

    def reset(self) -> Dict[str, np.ndarray]:
        self._frames[:] = 0
        self._last_packed = [None for _ in range(self.num_envs)]
        return self.step(is_first=np.ones(self.num_envs, dtype=bool))


def build_graph_obs(
    env,
    graph_cfg: dict,
    *,
    num_envs: int,
    builder_cls: type = GraphObsBuilder,
) -> Optional[GraphObsBuilder]:
    """Return a builder or None when graph obs is disabled."""
    if not bool(graph_cfg.get("enabled", False)):
        return None

    # Unset paths arrive from elements.Config as "", which load_config reads as
    # "use the packaged thresholds".
    teemo_cfg = load_teemo_config(
        graph_cfg.get("profile") or "tabletop",
        path=graph_cfg.get("thresholds_path"),
    )
    if "n_max" in graph_cfg:
        teemo_cfg["selection"]["n_max"] = int(graph_cfg["n_max"])
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
    vocab = build_graph_vocab(teemo_cfg["whitelist_dir"])

    n_max = int(teemo_cfg["selection"]["n_max"])
    e_max = int(graph_cfg.get("e_max", 256))

    cameras = graph_cfg.get("cameras")
    if not cameras:
        cameras = [graph_cfg.get("camera", "fetch_head")]
    primary_camera = graph_cfg.get("primary_camera") or cameras[0]

    return builder_cls(
        env,
        num_envs=num_envs,
        teemo_cfg=teemo_cfg,
        vocab=vocab,
        n_max=n_max,
        e_max=e_max,
        cameras=list(cameras),
        primary_camera=primary_camera,
        bypass_teemo=bool(graph_cfg.get("bypass_teemo", False)),
        staleness_enabled=bool(graph_cfg.get("staleness_enabled", True)),
    )
