"""Local successful-rollout interaction collector (schema version 3).

MS-HAB is treated as read-only.  This wrapper observes the simulator through
public environment state, buffers each vector environment independently, and
commits a rollout only when that environment succeeds.

For each successful rollout it records:

* every non-robot entity contacted by any robot link;
* the task target key, used to name/select offline assets;
* direct supporters of contacted entities (one hop only).

The active target is not injected as a whitelist member unless the robot
actually contacts it. Robot links are evidence of interaction, never whitelist
members.  The legacy ``robot_qpos`` and ``obj_pose_wrt_base`` arrays remain in
the payload so the affordance miner continues to consume the same success
samples.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

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


_SCHEMA_VERSION = 3
_EPS_FORCE_DEFAULT = 0.05
_EPS_Z_DEFAULT = 0.01
_MIN_VERTICAL_FORCE_RATIO_DEFAULT = 0.5


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


def _pairwise_force(scene, a, b, env_idx: int) -> Optional[np.ndarray]:
    try:
        force = _to_np(scene.get_pairwise_contact_forces(a, b))
    except Exception:
        return None
    if force.ndim == 1:
        return np.asarray(force, dtype=float)
    if env_idx >= force.shape[0]:
        return None
    return np.asarray(force[env_idx], dtype=float)


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
        eps_z: float = _EPS_Z_DEFAULT,
        min_vertical_force_ratio: float = _MIN_VERTICAL_FORCE_RATIO_DEFAULT,
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
        self.eps_z = float(eps_z)
        self.min_vertical_force_ratio = float(min_vertical_force_ratio)

        self.success_robot_qpos: List[List[float]] = []
        self.success_obj_pose_wrt_base: List[List[float]] = []
        self.interaction_rollouts: List[Dict[str, Any]] = []
        self._reset_buffers()

        root = Path(out_root) if out_root else Path(ASSET_DIR) / "robot_success_states"
        save_dir = root / self.agent.uid / self.subtask_type
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = save_dir / f"{self.obj_id}.pkl"

    def _reset_buffers(self, env_indices=None) -> None:
        if not hasattr(self, "_episode_interacted"):
            self._episode_interacted = [dict() for _ in range(self.num_envs)]
            self._episode_entities = [dict() for _ in range(self.num_envs)]
            self._episode_supports = [dict() for _ in range(self.num_envs)]
            self._success_latched = np.zeros(self.num_envs, dtype=bool)
        indices = range(self.num_envs) if env_indices is None else env_indices
        for idx in indices:
            self._episode_interacted[int(idx)].clear()
            self._episode_entities[int(idx)].clear()
            self._episode_supports[int(idx)].clear()
            self._success_latched[int(idx)] = False

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

        for env_idx in range(self.num_envs):
            self._observe_step(env_idx)

        success = info.get("success")
        success_np = (
            np.zeros(self.num_envs, dtype=bool)
            if success is None else _to_np(success).astype(bool).reshape(-1)
        )
        if success_np.size < self.num_envs:
            success_np = np.pad(success_np, (0, self.num_envs - success_np.size))
        newly_successful = success_np[:self.num_envs] & ~self._success_latched
        for env_idx in np.where(newly_successful)[0].tolist():
            self._commit_success(int(env_idx))
            self._success_latched[int(env_idx)] = True

        done = np.logical_or(
            _to_np(term).astype(bool).reshape(-1),
            _to_np(trunc).astype(bool).reshape(-1),
        )
        self._reset_buffers(np.where(done[:self.num_envs])[0].tolist())
        return obs, rew, term, trunc, info

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

    def _target(self, env_idx: int) -> Tuple[Optional[Any], str]:
        state = get_privileged_state(self, env_idx, mshab_object_name="actual")
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

    def _observe_step(self, env_idx: int) -> None:
        scene = self._base_env.scene
        entities = self._scene_entities()
        robot_links, _ = self._robot_links()
        target_ent, target_key = self._target(env_idx)

        entity_by_key: Dict[str, Any] = {}
        for ent in entities:
            key = stable_entity_key(ent)
            if key:
                entity_by_key[key] = ent
        if target_ent is not None:
            # Alias the actual target to the task key when appropriate.  For
            # pick tasks this removes the generic ``actor:obj_0`` key so one
            # physical object is not recorded twice.
            raw_key = stable_entity_key(target_ent)
            if raw_key and raw_key != target_key:
                entity_by_key.pop(raw_key, None)
            entity_by_key[target_key] = target_ent

        interacted = self._episode_interacted[env_idx]
        episode_entities = self._episode_entities[env_idx]
        for key, ent in entity_by_key.items():
            total_force = 0.0
            for robot_link in robot_links:
                vector = _pairwise_force(scene, robot_link, ent, env_idx)
                if vector is not None:
                    total_force += float(np.linalg.norm(vector))
            if total_force <= self.eps_force:
                continue
            rec = _record(ent, key=key)
            if rec is not None:
                rec["max_robot_force"] = float(total_force)
                prior = interacted.get(key)
                if prior is None or total_force > prior.get("max_robot_force", 0.0):
                    interacted[key] = rec
                episode_entities[key] = ent

        # Support observation is one hop from interacted entities only.  The
        # task target is not injected as a member just because it is active; if
        # it is contacted, it enters ``episode_entities`` like any other object.
        roots = dict(episode_entities)
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
        if supported_xyz is None:
            return
        scene = self._base_env.scene
        for supporter_key, supporter in candidates.items():
            if supporter_key == supported_key:
                continue
            vector = _pairwise_force(scene, supporter, supported, env_idx)
            if vector is None:
                continue
            force = float(np.linalg.norm(vector))
            if force <= self.eps_force:
                continue
            vertical_ratio = abs(float(vector[2])) / force if force > 0 else 0.0
            if vertical_ratio < self.min_vertical_force_ratio:
                continue
            supporter_xyz = _entity_xyz(supporter, env_idx)
            if supporter_xyz is None or supporter_xyz[2] + self.eps_z >= supported_xyz[2]:
                continue
            rec = {
                "supporter": _record(supporter, key=supporter_key),
                "supported_key": supported_key,
                "force": force,
                "vertical_force_ratio": vertical_ratio,
                "dz": float(supported_xyz[2] - supporter_xyz[2]),
            }
            pair = (supporter_key, supported_key)
            prior = self._episode_supports[env_idx].get(pair)
            if prior is None or force > prior.get("force", 0.0):
                self._episode_supports[env_idx][pair] = rec

    def _commit_success(self, env_idx: int) -> None:
        # Preserve the legacy affordance samples, but record only the success
        # transition rather than every frame while success remains true.
        qpos = _to_np(self.agent.robot.qpos)[env_idx]
        merged_state = get_privileged_state(
            self, env_idx, mshab_object_name="merged"
        )
        pose_target = (
            merged_state.active_obj if self.subtask_type == "pick"
            else (merged_state.active_handle_link or merged_state.active_obj)
        )
        if pose_target is None or getattr(pose_target, "pose", None) is None:
            raise RuntimeError("successful rollout has no resolvable target pose")
        relative_pose = _to_np(
            (self.agent.base_link.pose.inv() * pose_target.pose).raw_pose
        )
        obj_pose = relative_pose[env_idx] if relative_pose.ndim == 2 else relative_pose
        self.success_robot_qpos.append(np.asarray(qpos, dtype=float).tolist())
        self.success_obj_pose_wrt_base.append(np.asarray(obj_pose, dtype=float).tolist())

        _target_ent, target_key = self._target(env_idx)
        interacted = list(self._episode_interacted[env_idx].values())
        root_keys = {item["key"] for item in interacted}
        supports = [
            value for (_supporter, supported), value
            in self._episode_supports[env_idx].items()
            if supported in root_keys and value.get("supporter") is not None
        ]
        self.interaction_rollouts.append({
            "target_key": target_key,
            "interacted": interacted,
            "supports": supports,
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
            "robot_qpos": self.success_robot_qpos,
            "obj_pose_wrt_base": self.success_obj_pose_wrt_base,
            "interaction_rollouts": self.interaction_rollouts,
        }
        with open(self.save_path, "wb") as stream:
            pickle.dump(payload, stream)
        return super().close()
