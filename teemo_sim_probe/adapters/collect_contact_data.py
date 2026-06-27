"""Local successful-rollout interaction collector (schema version 5).

MS-HAB is treated as read-only.  This wrapper observes the simulator through
public environment state, buffers each vector environment independently, and
commits a rollout only when that environment succeeds.

For each successful rollout it records:

* every non-robot entity contacted by an **ee link** (tcp, finger1, finger2);
  contact by any other robot link does not count as evidence -- the runtime
  graph only cares about ee-driven interactions.
* the task target key, used to name/select offline assets;
* direct supporters of contacted entities (one hop only).
* a per-rollout ``bin_stats`` block: running maxes of ee->object planar
  distance, |height offset|, and their K-window absolute changes, used by the
  whitelist miner to derive per-(subtask, target) bin edges.

The active target is not injected as a whitelist member unless an ee link
actually contacts it. Robot links other than tcp/finger1/finger2 are evidence
of nothing.  Pose arrays are stored for affordance mining, including
schema-v4 ``tcp_pose_wrt_base`` to avoid rebuilding TCP poses with a separate
FK chain.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, TYPE_CHECKING

import gymnasium as gym
import numpy as np
import sapien.physx as physx

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch

from teemo_sim_probe.core.affordance import canonical_affordance_key
from teemo_sim_probe.core.entity_identity import (
    entity_kind,
    entity_name,
    stable_entity_key,
)
from teemo_sim_probe.adapters.privileged_state import get_privileged_state

if TYPE_CHECKING:
    from mshab.envs.sequential_task import SequentialTaskEnv


_SCHEMA_VERSION = 5
_EPS_FORCE_DEFAULT = 0.05
_MIN_VERTICAL_FORCE_RATIO_DEFAULT = 0.5
_OBSERVE_STRIDE_DEFAULT = 5
_DEFAULT_TEMPORAL_K = 5


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _entity_xyz(ent, env_idx: int) -> Optional[np.ndarray]:
    pose = getattr(ent, "pose", None)
    if pose is None:
        return None
    try:
        p = _to_np(pose.p)
    except Exception:
        return None
    if p.ndim == 2:
        if env_idx >= p.shape[0]:
            return None
        p = p[env_idx]
    p = np.asarray(p, dtype=float).reshape(-1)
    return p[:3] if p.size >= 3 and np.all(np.isfinite(p[:3])) else None


def _record(ent, *, key: Optional[str] = None) -> Optional[Dict[str, str]]:
    stable = key or stable_entity_key(ent)
    if not stable:
        return None
    return {
        "key": stable,
        "name": entity_name(ent),
        "kind": entity_kind(ent),
    }


class FetchCollectContactDataWrapper(gym.Wrapper):
    """Collect successful-rollout interactions without modifying MS-HAB."""

    def __init__(
        self,
        env,
        *,
        eps_force: float = _EPS_FORCE_DEFAULT,
        min_vertical_force_ratio: float = _MIN_VERTICAL_FORCE_RATIO_DEFAULT,
        observe_stride: int = _OBSERVE_STRIDE_DEFAULT,
        temporal_k: int = _DEFAULT_TEMPORAL_K,
        out_root: Optional[str] = None,
    ) -> None:
        super().__init__(env)
        base: "SequentialTaskEnv" = env.unwrapped
        self._base_env = base
        self.agent: Fetch = base.agent
        if not isinstance(self.agent, Fetch):
            raise TypeError(f"{self.__class__.__name__} currently supports Fetch only")

        plans = [tp for values in base.bc_to_task_plans.values() for tp in values]
        if not plans or not all(len(tp.subtasks) == 1 for tp in plans):
            raise ValueError("interaction collection requires single-subtask plans")
        canonical_ids = {
            canonical_affordance_key(tp.subtasks[0].obj_id)
            for tp in plans
        }
        if len(canonical_ids) != 1:
            raise ValueError(
                "all task plans must reference the same canonical target; "
                f"got {sorted(canonical_ids)}"
            )
        self.obj_id = next(iter(canonical_ids))
        self.subtask_type = plans[0].subtasks[0].type
        self.num_envs = int(getattr(base, "num_envs", 1))

        self.eps_force = float(eps_force)
        self.min_vertical_force_ratio = float(min_vertical_force_ratio)
        self.observe_stride = max(1, int(observe_stride))
        self.temporal_k = max(1, int(temporal_k))
        self._step_count = 0
        self._last_observed_step = -1
        self._force_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}

        self.success_robot_qpos: List[List[float]] = []
        self.success_obj_pose_wrt_base: List[List[float]] = []
        self.success_tcp_pose_wrt_base: List[List[float]] = []
        self.interaction_rollouts: List[Dict[str, Any]] = []

        # Per-env transient buffers.
        self._episode_interacted: List[Dict[str, Dict[str, Any]]] = []
        self._episode_entities: List[Dict[str, Any]] = []
        self._episode_supports: List[Dict[Tuple[str, str], Dict[str, Any]]] = []
        self._episode_bin_max: List[Dict[str, float]] = []
        # (entity_key, relation_name) -> deque of last K+1 values
        self._episode_history: List[Dict[Tuple[str, str], Deque[float]]] = []
        self._reset_buffers()

        # Aggregated across every committed success, used as a quick top-level
        # summary in the pickle. The miner still re-aggregates per rollout.
        self.aggregated_bin_max: Dict[str, float] = defaultdict(float)

        root = Path(out_root) if out_root else Path(ASSET_DIR) / "robot_success_states"
        save_dir = root / self.agent.uid / self.subtask_type
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = save_dir / f"{self.obj_id}.pkl"

    def _reset_buffers(self, env_indices=None) -> None:
        if not self._episode_interacted:
            self._episode_interacted = [dict() for _ in range(self.num_envs)]
            self._episode_entities = [dict() for _ in range(self.num_envs)]
            self._episode_supports = [dict() for _ in range(self.num_envs)]
            self._episode_bin_max = [defaultdict(float) for _ in range(self.num_envs)]
            self._episode_history = [dict() for _ in range(self.num_envs)]
        indices = range(self.num_envs) if env_indices is None else env_indices
        for idx in indices:
            i = int(idx)
            self._episode_interacted[i].clear()
            self._episode_entities[i].clear()
            self._episode_supports[i].clear()
            self._episode_bin_max[i].clear()
            self._episode_history[i].clear()

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self._reset_buffers()
        return result

    def step(self, action, *args, **kwargs):
        obs, rew, term, trunc, info = super().step(action, *args, **kwargs)
        if not physx.is_gpu_enabled():
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support CPU simulation"
            )

        # Forces are valid only for the just-stepped tick. Drop the cache
        # before any observation so we never read stale GPU contact data.
        self._force_cache.clear()
        self._step_count += 1
        if self._step_count % self.observe_stride == 0:
            self._observe_all_envs()
            self._last_observed_step = self._step_count

        done = np.logical_or(
            _to_np(term).astype(bool).reshape(-1),
            _to_np(trunc).astype(bool).reshape(-1),
        )
        self._reset_buffers(np.where(done[:self.num_envs])[0].tolist())
        return obs, rew, term, trunc, info

    def _observe_all_envs(self) -> None:
        for env_idx in range(self.num_envs):
            self._observe_step(env_idx)

    def _pairwise_force(self, scene, a, b, env_idx: int) -> Optional[np.ndarray]:
        key = (id(a), id(b))
        if key not in self._force_cache:
            try:
                self._force_cache[key] = _to_np(
                    scene.get_pairwise_contact_forces(a, b)
                )
            except Exception:
                self._force_cache[key] = None
        force = self._force_cache[key]
        if force is None:
            return None
        if force.ndim == 1:
            return np.asarray(force, dtype=float)
        if env_idx >= force.shape[0]:
            return None
        return np.asarray(force[env_idx], dtype=float)

    def _robot_links(self) -> Tuple[List[Any], set]:
        try:
            links = list(self.agent.robot.get_links())
        except Exception:
            links = []
        names = {entity_name(link) for link in links}
        return links, names

    def _scene_entities(self) -> List[Any]:
        seg_map = dict(getattr(self._base_env, "segmentation_id_map", {}))
        robot_links, robot_names = self._robot_links()
        robot_ids = {id(link) for link in robot_links}
        entities: List[Any] = []
        seen = set()
        for seg_id, ent in seg_map.items():
            if not seg_id or ent is None:
                continue
            if id(ent) in robot_ids or entity_name(ent) in robot_names:
                continue
            key = stable_entity_key(ent)
            if not key or key in seen:
                continue
            seen.add(key)
            entities.append(ent)
        return entities

    def _target(self, env_idx: int, state=None) -> Tuple[Optional[Any], str]:
        state = state or get_privileged_state(
            self, env_idx, mshab_object_name="actual"
        )
        if self.subtask_type == "pick":
            # MS-HAB may expose pick targets through a merged runtime actor
            # named ``obj_0``.  The offline assets and runtime whitelist are
            # keyed by the original task object id instead.
            return state.active_obj, f"actor:{self.obj_id}"

        target = state.active_handle_link or state.active_obj
        if target is not None:
            key = stable_entity_key(target)
            if key:
                return target, key
        return target, f"actor:{self.obj_id}"

    def _mark_interacted(
        self,
        env_idx: int,
        key: str,
        ent: Any,
        *,
        max_ee_force: float = 0.0,
        grasped: bool = False,
    ) -> None:
        rec = _record(ent, key=key)
        if rec is None:
            return
        rec["max_ee_force"] = float(max_ee_force)
        if grasped:
            rec["grasped"] = True

        interacted = self._episode_interacted[env_idx]
        prior = interacted.get(key)
        prior_force = 0.0 if prior is None else float(
            prior.get("max_ee_force", 0.0)
        )
        if prior is not None and prior.get("grasped"):
            rec["grasped"] = True
        if prior is None or grasped or max_ee_force > prior_force:
            interacted[key] = rec
        self._episode_entities[env_idx][key] = ent

    def _update_bin_stats(
        self,
        env_idx: int,
        entity_by_key: Dict[str, Any],
        ee_xyz: Optional[np.ndarray],
    ) -> None:
        """Accumulate per-rollout running maxes for ee<->object spatial values
        and their K-window absolute changes. Compatibility metrics are not
        sampled here (they require the affordance asset) -- defaults are
        applied in the miner."""
        if ee_xyz is None:
            return
        maxes = self._episode_bin_max[env_idx]
        history = self._episode_history[env_idx]
        ee_xy = ee_xyz[:2]
        ee_z = float(ee_xyz[2])
        for key, ent in entity_by_key.items():
            obj_xyz = _entity_xyz(ent, env_idx)
            if obj_xyz is None:
                continue
            pd = float(np.linalg.norm(ee_xy - obj_xyz[:2]))
            ho_signed = ee_z - float(obj_xyz[2])
            if pd > maxes["planar_distance"]:
                maxes["planar_distance"] = pd
            ho_abs = abs(ho_signed)
            if ho_abs > maxes["height_offset"]:
                maxes["height_offset"] = ho_abs
            for relation, value in (
                ("planar_distance", pd),
                ("height_offset", ho_signed),
            ):
                hk = (key, relation)
                buf = history.get(hk)
                if buf is None:
                    buf = deque(maxlen=self.temporal_k + 1)
                    history[hk] = buf
                buf.append(value)
                if len(buf) > self.temporal_k:
                    change = abs(buf[-1] - buf[0])
                    change_key = f"{relation}_change"
                    if change > maxes[change_key]:
                        maxes[change_key] = change

    def _observe_step(self, env_idx: int) -> None:
        scene = self._base_env.scene
        entities = self._scene_entities()
        actual_state = get_privileged_state(
            self, env_idx, mshab_object_name="actual"
        )
        target_ent, target_key = self._target(env_idx, actual_state)
        physics_target_ent = target_ent
        ee_links = list(actual_state.ee_links)
        ee_xyz = (
            np.asarray(actual_state.tcp_pose_world[:3], dtype=float)
            if actual_state.tcp_pose_world is not None
            else None
        )

        # For pick, MS-HAB's grasp predicate and contact forces are most
        # reliable on the merged runtime actor.  We still record the canonical
        # task key, so the whitelist never learns ``actor:obj_0``.
        merged_state = None
        if self.subtask_type == "pick":
            merged_state = get_privileged_state(
                self, env_idx, mshab_object_name="merged"
            )
            if getattr(merged_state, "active_obj", None) is not None:
                physics_target_ent = merged_state.active_obj

        entity_by_key: Dict[str, Any] = {}
        for ent in entities:
            key = stable_entity_key(ent)
            if key:
                entity_by_key[key] = ent
        if target_ent is not None:
            raw_key = stable_entity_key(target_ent)
            if raw_key and raw_key != target_key:
                entity_by_key.pop(raw_key, None)
        if physics_target_ent is not None:
            raw_key = stable_entity_key(physics_target_ent)
            if raw_key and raw_key != target_key:
                entity_by_key.pop(raw_key, None)
            entity_by_key[target_key] = physics_target_ent

        # Spatial running-max sampling spans every candidate, not just those
        # the ee currently touches: bin edges learn the range the ee operates
        # over during the rollout.
        self._update_bin_stats(env_idx, entity_by_key, ee_xyz)

        # Contact evidence is restricted to the ee link set so the whitelist
        # ignores incidental robot-body bumps.
        for key, ent in entity_by_key.items():
            total_force = 0.0
            for ee_link in ee_links:
                vector = self._pairwise_force(scene, ee_link, ent, env_idx)
                if vector is not None:
                    total_force += float(np.linalg.norm(vector))
            if total_force <= self.eps_force:
                continue
            self._mark_interacted(
                env_idx, key, ent, max_ee_force=total_force
            )

        if self.subtask_type == "pick" and physics_target_ent is not None:
            grasped = False
            if merged_state is not None:
                grasped = merged_state.is_grasping(physics_target_ent)
            if not grasped:
                grasped = actual_state.is_grasping(target_ent)
            if grasped:
                force = actual_state.ee_object_contact_force(target_ent)
                if merged_state is not None:
                    force = max(
                        force,
                        merged_state.ee_object_contact_force(physics_target_ent),
                    )
                self._mark_interacted(
                    env_idx, target_key, physics_target_ent,
                    max_ee_force=force, grasped=True,
                )

        # Support evidence may be visible before robot-object contact. Buffer
        # interacted entities plus the active target; the whitelist miner later
        # admits only supporters of interacted roots.
        roots = dict(self._episode_entities[env_idx])
        if physics_target_ent is not None:
            roots[target_key] = physics_target_ent
        for supported_key, supported in roots.items():
            self._observe_direct_supporters(
                env_idx, supported_key, supported, entity_by_key,
            )

    def _observe_direct_supporters(
        self,
        env_idx: int,
        supported_key: str,
        supported: Any,
        candidates: Dict[str, Any],
    ) -> None:
        supported_xyz = _entity_xyz(supported, env_idx)
        scene = self._base_env.scene
        for supporter_key, supporter in candidates.items():
            if supporter_key == supported_key:
                continue
            vector = self._pairwise_force(scene, supporter, supported, env_idx)
            if vector is None:
                continue
            force = float(np.linalg.norm(vector))
            if force <= self.eps_force:
                continue
            vertical_ratio = abs(float(vector[2])) / force if force > 0 else 0.0
            supporter_xyz = _entity_xyz(supporter, env_idx)
            vertical_support = bool(
                vertical_ratio >= self.min_vertical_force_ratio
            )
            dz = (
                float(supported_xyz[2] - supporter_xyz[2])
                if (supporter_xyz is not None and supported_xyz is not None)
                else None
            )
            rec = {
                "supporter": _record(supporter, key=supporter_key),
                "supported_key": supported_key,
                "force": force,
                "vertical_force_ratio": vertical_ratio,
                "dz": dz,
                "vertical_support": vertical_support,
            }
            pair = (supporter_key, supported_key)
            prior = self._episode_supports[env_idx].get(pair)
            if prior is None:
                self._episode_supports[env_idx][pair] = rec
                continue
            prior_rank = (
                bool(prior.get("vertical_support", False)),
                float(prior.get("force", 0.0)),
            )
            if (vertical_support, force) > prior_rank:
                self._episode_supports[env_idx][pair] = rec

    def commit_success(
        self,
        env_idx: int,
        qpos: np.ndarray,
        obj_pose_wrt_base: np.ndarray,
        tcp_pose_wrt_base: np.ndarray,
    ) -> None:
        """Append one success sample. Called by the collection script after
        venv.step returns, with state read at the script level.
        """
        # Guarantee the grasp moment is captured: if stride skipped this tick,
        # observe this single env now while sim forces are still valid.
        if self._last_observed_step != self._step_count:
            self._observe_step(env_idx)

        self.success_robot_qpos.append(np.asarray(qpos, dtype=float).tolist())
        self.success_obj_pose_wrt_base.append(
            np.asarray(obj_pose_wrt_base, dtype=float).tolist()
        )
        self.success_tcp_pose_wrt_base.append(
            np.asarray(tcp_pose_wrt_base, dtype=float).tolist()
        )

        _target_ent, target_key = self._target(env_idx)
        interacted = list(self._episode_interacted[env_idx].values())
        root_keys = {item["key"] for item in interacted}
        root_keys.add(target_key)
        supports = [
            value for (_supporter, supported), value
            in self._episode_supports[env_idx].items()
            if supported in root_keys
            and value.get("supporter") is not None
        ]
        rollout_bin_stats = dict(self._episode_bin_max[env_idx])
        for k, v in rollout_bin_stats.items():
            if v > self.aggregated_bin_max[k]:
                self.aggregated_bin_max[k] = v
        self.interaction_rollouts.append({
            "target_key": target_key,
            "interacted": interacted,
            "supports": supports,
            "bin_stats": rollout_bin_stats,
        })

    def close(self):
        entity_key = (
            self.interaction_rollouts[0].get("target_key")
            if self.interaction_rollouts else f"actor:{self.obj_id}"
        )
        payload = {
            "_schema_version": _SCHEMA_VERSION,
            "obj_id": self.obj_id,
            "entity_key": entity_key,
            "subtask_type": self.subtask_type,
            "temporal_k": self.temporal_k,
            "robot_qpos": self.success_robot_qpos,
            "obj_pose_wrt_base": self.success_obj_pose_wrt_base,
            "tcp_pose_wrt_base": self.success_tcp_pose_wrt_base,
            "interaction_rollouts": self.interaction_rollouts,
            "bin_stats": dict(self.aggregated_bin_max),
        }
        with open(self.save_path, "wb") as stream:
            pickle.dump(payload, stream)
        return super().close()
