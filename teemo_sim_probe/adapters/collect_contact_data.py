"""Track B collection wrapper: contact-graph capture at success frames.

Extends ``FetchCollectRobotInitWrapper`` to ALSO record, at each per-env
success transition, the pairwise contact graph in a neighborhood of the
manipulation target. The existing affordance + e_domain pipeline depended only
on ``robot_qpos`` and ``obj_pose_wrt_base``; this wrapper preserves both and
adds the data the new per-subtask whitelist miner needs.

Writes to the SAME path the upstream wrapper used --
``$MS_ASSET_DIR/robot_success_states/<robot_uid>/<subtask>/<obj_id>.pkl`` --
so ``build_affordances.py`` keeps reading it unchanged. The pkl is a strict
superset of the old shape: ``robot_qpos`` and ``obj_pose_wrt_base`` are
preserved; ``contact_graphs`` is added as a new top-level field that the
new ``tools/build_subtask_whitelists.py`` consumes.

On-disk pkl shape (schema_version=2)::

    {
      "_schema_version": 2,
      "obj_id":            "024_bowl",
      "subtask_type":      "pick",
      "robot_qpos":        [[...15...], ...],            # unchanged
      "obj_pose_wrt_base": [[x,y,z,qw,qx,qy,qz], ...],   # unchanged
      "contact_graphs":    [                              # NEW: one per success
        {
          "target":     "env-0_024_bowl-3",
          "target_canonical": "024_bowl",
          "pairs": [
            {"a": "env-0_024_bowl-3", "b": "kitchen_counter",
             "force": 3.4, "dz": 0.012,
             "a_kind": "actor", "b_kind": "link"},
            ...
          ]
        },
        ...
      ]
    }

Only pairs with ``force > eps_force`` are recorded. The miner then takes the
transitive closure: target ∪ contacts(target) ∪ contacts(contacts) ∪ ...
clipped to a small hop budget so unrelated scene clutter doesn't leak in.

Runtime cost: ``get_pairwise_contact_forces`` is O(1) per pair on GPU sim.
We restrict the scan to entities within a planar radius of the target, so the
per-step cost is bounded by O(n_nearby) -- well under one ms in practice.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import gymnasium as gym
import numpy as np
import sapien.physx as physx

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Fetch

from teemo_sim_probe.core.affordance import canonical_affordance_key

if TYPE_CHECKING:
    from mshab.envs.sequential_task import SequentialTaskEnv


_SCHEMA_VERSION = 2

# Pairs whose contact force exceeds this magnitude (Newtons) are recorded.
# Matches the runtime ``contact.eps_force`` in thresholds.yaml so the miner
# and runtime use the same "in contact" criterion.
_EPS_FORCE_DEFAULT = 0.05

# Planar radius (m) around the target within which pairs are scanned. Wider
# than n_slots' overflow radius so the miner sees both direct supports and
# any second-hop scene structure even when the first hop is small.
_NEIGHBORHOOD_RADIUS_M = 1.5


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _entity_name(ent) -> str:
    return getattr(ent, "name", str(ent))


def _entity_kind(ent) -> str:
    """``actor`` / ``link`` / ``other`` -- mirrors selector match_key kinds."""
    t = type(ent).__name__
    if t == "Actor":
        return "actor"
    if t == "Link":
        return "link"
    return "other"


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
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        return None
    return p.astype(float)


def _pairwise_force(scene, a, b, env_idx: int) -> Optional[np.ndarray]:
    """World-frame contact-force vector for one env, or None on failure."""
    try:
        f = _to_np(scene.get_pairwise_contact_forces(a, b))
    except Exception:
        return None
    if f.ndim == 1:
        return f.astype(float)
    if env_idx >= f.shape[0]:
        return None
    return f[env_idx].astype(float)


class FetchCollectContactDataWrapper(gym.Wrapper):
    """Drop-in replacement for ``FetchCollectRobotInitWrapper`` that also
    captures per-success contact graphs around the manipulation target.

    Single-subtask plans only (same constraint as the upstream wrapper); the
    output pkl is keyed by canonical YCB id, like the existing affordance and
    success-state pipelines.
    """

    def __init__(
        self,
        env,
        *,
        eps_force: float = _EPS_FORCE_DEFAULT,
        neighborhood_radius: float = _NEIGHBORHOOD_RADIUS_M,
        out_root: Optional[str] = None,
    ) -> None:
        super().__init__(env)
        uenv: "SequentialTaskEnv" = env.unwrapped
        self._base_env = uenv
        self.agent: Fetch = self._base_env.agent
        assert isinstance(self.agent, Fetch), (
            f"{self.__class__.__name__} currently only supports fetch"
        )

        all_task_plans = [
            tp for tps in self._base_env.bc_to_task_plans.values() for tp in tps
        ]
        assert all(len(tp.subtasks) == 1 for tp in all_task_plans), (
            "Must have only one subtask"
        )
        canonical_ids = {
            canonical_affordance_key(tp.subtasks[0].obj_id)
            for tp in all_task_plans
        }
        assert len(canonical_ids) == 1, (
            f"All task plans must reference the same canonical object; "
            f"got {sorted(canonical_ids)}"
        )
        self.obj_id = next(iter(canonical_ids))

        tp0 = all_task_plans[0].subtasks[0]
        self.subtask_type = tp0.type

        self.success_robot_qpos: List[List[float]] = []
        self.success_obj_pose_wrt_base: List[List[float]] = []
        self.contact_graphs: List[Dict[str, Any]] = []

        self.eps_force = float(eps_force)
        self.neighborhood_radius = float(neighborhood_radius)

        # Drop-in replacement for FetchCollectRobotInitWrapper: same output
        # path so build_affordances.py keeps working unchanged.
        root = Path(out_root) if out_root else Path(ASSET_DIR) / "robot_success_states"
        save_dir = root / self._base_env.agent.uid / self.subtask_type
        os.makedirs(save_dir, exist_ok=True)
        self.save_path = save_dir / f"{self.obj_id}.pkl"

    # ----- per-step --------------------------------------------------------- #
    def step(self, action, *args, **kwargs):
        obs, rew, term, trunc, info = super().step(action, *args, **kwargs)
        if not physx.is_gpu_enabled():
            raise NotImplementedError(
                f"{self.__class__.__name__} doesn't work on CPU sim yet"
            )

        success = info.get("success")
        if success is None or len(success) == 0:
            return obs, rew, term, trunc, info

        success_np = _to_np(success)
        # robot_qpos & obj_pose_wrt_base: unchanged from upstream wrapper.
        self.success_robot_qpos += (
            self.agent.robot.qpos[success].cpu().numpy().tolist()
        )
        self.success_obj_pose_wrt_base += (
            (
                self.agent.base_link.pose.inv()
                * self._base_env.subtask_objs[0].pose
            )
            .raw_pose[success].cpu().numpy().tolist()
        )

        # NEW: per-success-env contact-graph capture.
        for env_idx in np.where(success_np)[0].tolist():
            cg = self._capture_contact_graph(int(env_idx))
            if cg is not None:
                self.contact_graphs.append(cg)
        return obs, rew, term, trunc, info

    # ----- contact-graph snapshot ------------------------------------------ #
    def _capture_contact_graph(self, env_idx: int) -> Optional[Dict[str, Any]]:
        e = self._base_env
        scene = e.scene
        seg_id_map: Dict[int, Any] = dict(getattr(e, "segmentation_id_map", {}))
        if not seg_id_map:
            return None

        target = e.subtask_objs[0]
        # Per-env name disambiguation: the merged actor's name typically still
        # carries the env-prefixed instance string for the right env.
        target_name = _entity_name(target)
        target_canonical = canonical_affordance_key(target_name) or target_name

        target_xyz = _entity_xyz(target, env_idx)

        # Build the neighborhood entity list. Robot links excluded -- the
        # whitelist gate is about scene physics, not the manipulator.
        try:
            robot_links = set(self.agent.robot.get_links())
        except Exception:
            robot_links = set()

        nearby: List[Any] = []
        for seg_id, ent in seg_id_map.items():
            if ent is None or seg_id == 0 or ent in robot_links:
                continue
            if ent is target:
                continue
            if target_xyz is not None:
                xyz = _entity_xyz(ent, env_idx)
                if xyz is not None:
                    if np.linalg.norm(xyz[:2] - target_xyz[:2]) > self.neighborhood_radius:
                        continue
            nearby.append(ent)

        # Always include the target itself in the participating set.
        participants = [target] + nearby

        pairs: List[Dict[str, Any]] = []
        seen: set = set()

        def _add_pair(a, b, force_vec: np.ndarray) -> None:
            force = float(np.linalg.norm(force_vec))
            if force <= self.eps_force:
                return
            pa = _entity_xyz(a, env_idx)
            pb = _entity_xyz(b, env_idx)
            dz: Optional[float] = None
            if pa is not None and pb is not None:
                dz = float(pa[2] - pb[2])
            key = frozenset((id(a), id(b)))
            if key in seen:
                return
            seen.add(key)
            pairs.append({
                "a": _entity_name(a),
                "b": _entity_name(b),
                "a_kind": _entity_kind(a),
                "b_kind": _entity_kind(b),
                "force": force,
                "dz": dz,
            })

        # Target <-> every nearby entity.
        for ent in nearby:
            fv = _pairwise_force(scene, target, ent, env_idx)
            if fv is None:
                continue
            _add_pair(target, ent, fv)

        # Second hop: every (nearby_a, nearby_b) pair. Bounded by neighborhood
        # so this stays O(|nearby|^2) on a small set.
        n = len(nearby)
        for i in range(n):
            a = nearby[i]
            for j in range(i + 1, n):
                b = nearby[j]
                fv = _pairwise_force(scene, a, b, env_idx)
                if fv is None:
                    continue
                _add_pair(a, b, fv)

        return {
            "target": target_name,
            "target_canonical": target_canonical,
            "pairs": pairs,
        }

    # ----- shutdown --------------------------------------------------------- #
    def close(self):
        payload = dict(
            _schema_version=_SCHEMA_VERSION,
            obj_id=self.obj_id,
            subtask_type=self.subtask_type,
            robot_qpos=self.success_robot_qpos,
            obj_pose_wrt_base=self.success_obj_pose_wrt_base,
            contact_graphs=self.contact_graphs,
        )
        with open(self.save_path, "wb") as f:
            pickle.dump(payload, f)
        return super().close()
