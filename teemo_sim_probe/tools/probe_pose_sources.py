"""Compare three ways to record obj_pose_wrt_base at success time.

Runs a checkpointed Fetch pick policy until the first successes, and at each
success transition records the position of the target object in the robot
base frame using three sources:

  A) actual_state.active_obj.pose       (current collector)
  B) merged_state.active_obj.pose       (Option 1: use merged actor)
  C) per-env SAPIEN entity ._objs[row].pose, with single-env base inverse
     (Option 2: bypass batched wrappers entirely)

For each success, prints the TCP-to-OBJ distance computed from each source.
The correct source should give a few centimetres at success; the broken one
will give 0.4--1.5 m.

Usage:
    python -m teemo_sim_probe.tools.probe_pose_sources \\
        --ckpt-dir mshab_checkpoints/rl/tidy_house/pick/002_master_chef_can \\
        --obj-id 002_master_chef_can --num-envs 4 --max-successes 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch  # noqa: F401

import mshab.envs  # noqa: F401  registers PickSubtaskTrain-v0
from mani_skill import ASSET_DIR
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mshab.envs.planner import plan_data_from_file
from mshab.envs.wrappers import (
    FetchActionWrapper,
    FetchDepthObservationWrapper,
    FrameStack,
)

from teemo_sim_probe.adapters.policy_loader import load_policy
from teemo_sim_probe.adapters.privileged_state import (
    _entity_for_env,
    get_privileged_state,
)


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _quat_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _inv_then_apply_to_point(pose_wxyz, point):
    p, q = pose_wxyz[:3], pose_wxyz[3:7]
    R = _quat_R(q)
    return R.T @ (np.asarray(point) - p)


def _wrapper_pose_for_env(wrapper, env_idx):
    """(p, q) in world frame for env_idx, reading the wrapper's own scene_idxs."""
    if wrapper is None or getattr(wrapper, "pose", None) is None:
        return None
    p = _to_np(wrapper.pose.p)
    q = _to_np(wrapper.pose.q)
    scene_idxs = _to_np(getattr(wrapper, "_scene_idxs", np.array([0]))).reshape(-1).tolist()
    if env_idx not in scene_idxs:
        # Wrapper doesn't cover this env; fall back to row 0 (matches the
        # current collector's silent behaviour).
        row = 0
    else:
        row = scene_idxs.index(env_idx)
    if p.ndim == 2:
        p = p[row]
    if q.ndim == 2:
        q = q[row]
    return np.asarray(p, dtype=float), np.asarray(q, dtype=float)


def _sapien_entity_pose(active_obj, env_idx):
    """Read the per-env underlying SAPIEN entity's world pose directly."""
    ent = _entity_for_env(active_obj, env_idx)
    if ent is None or getattr(ent, "pose", None) is None:
        return None
    pose = ent.pose
    try:
        p = _to_np(pose.p).reshape(-1)[:3]
        q = _to_np(pose.q).reshape(-1)[:4]
    except Exception:
        return None
    return np.asarray(p, dtype=float), np.asarray(q, dtype=float)


def _base_world_for_env(agent, env_idx):
    p = _to_np(agent.base_link.pose.p)
    q = _to_np(agent.base_link.pose.q)
    if p.ndim == 2:
        p = p[env_idx]
    if q.ndim == 2:
        q = q[env_idx]
    return np.asarray(p, dtype=float), np.asarray(q, dtype=float)


def _to_base_frame(base_pq, obj_pq):
    """Return obj position expressed in the robot base frame."""
    base_p, base_q = base_pq
    obj_p, _ = obj_pq
    R_base = _quat_R(base_q)
    return R_base.T @ (obj_p - base_p)


def _tcp_in_base(agent, env_idx):
    tcp_pose = agent.tcp.pose
    p = _to_np(tcp_pose.p)
    q = _to_np(tcp_pose.q)
    if p.ndim == 2:
        p = p[env_idx]
    if q.ndim == 2:
        q = q[env_idx]
    base_p, base_q = _base_world_for_env(agent, env_idx)
    R_base = _quat_R(base_q)
    return R_base.T @ (np.asarray(p, dtype=float) - base_p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--task", default="tidy_house")
    parser.add_argument("--subtask", default="pick")
    parser.add_argument("--obj-id", default="002_master_chef_can")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-successes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    RD = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_fp = RD / "task_plans" / args.task / args.subtask / "train" / f"{args.obj_id}.json"
    spawn = RD / "spawn_data" / args.task / args.subtask / "train" / "spawn_data.pt"
    pd = plan_data_from_file(plan_fp)
    n_envs = max(1, args.num_envs)
    task_plans = [pd.plans[i % len(pd.plans)] for i in range(n_envs)]

    env = gym.make(
        f"{args.subtask.capitalize()}SubtaskTrain-v0",
        num_envs=n_envs,
        obs_mode="rgb+depth+segmentation",
        sim_backend="gpu",
        robot_uids="fetch",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        render_mode="all",
        shader_dir="minimal",
        max_episode_steps=200,
        task_plans=task_plans,
        scene_builder_cls=pd.dataset,
        spawn_data_fp=spawn,
        require_build_configs_repeated_equally_across_envs=False,
        add_event_tracker_info=True,
        sensor_configs=dict(width=128, height=128),
    )
    env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
    env = FrameStack(env, num_stack=3,
                     stacking_keys=["fetch_head_depth", "fetch_hand_depth"])
    env = FetchActionWrapper(env, stationary_base=False,
                             stationary_torso=False, stationary_head=True)
    venv = ManiSkillVectorEnv(env, ignore_terminations=True,
                              max_episode_steps=200)

    obs, _ = venv.reset(seed=args.seed, options=dict(reconfigure=True))
    policy = load_policy(args.ckpt_dir, venv, obs, device=args.device)
    if policy.kind == "random":
        print("[warn] policy loader fell back to random actions; "
              "success rate will be ~0.")

    base_env = venv.unwrapped
    agent = base_env.agent
    success_latched = np.zeros(n_envs, dtype=bool)

    print(f"\n{'env':>3s} {'A(actual) |tcp-obj|':>22s}  "
          f"{'B(merged) |tcp-obj|':>22s}  "
          f"{'C(sapien) |tcp-obj|':>22s}")
    n_committed = 0
    t0 = time.time()
    for step in range(args.max_steps):
        if n_committed >= args.max_successes:
            break
        action = policy.act(obs)
        obs, _r, _t, _tr, info = venv.step(action)
        success = info.get("success")
        if success is None:
            continue
        succ = _to_np(success).astype(bool).reshape(-1)
        newly = succ & ~success_latched
        for env_idx in np.where(newly)[0].tolist():
            success_latched[env_idx] = True
            actual_state = get_privileged_state(
                venv, env_idx, mshab_object_name="actual")
            merged_state = get_privileged_state(
                venv, env_idx, mshab_object_name="merged")

            tcp_in_base = _tcp_in_base(agent, env_idx)
            base_pq = _base_world_for_env(agent, env_idx)

            results = {}
            for label, source in (("A", actual_state.active_obj),
                                  ("B", merged_state.active_obj)):
                pq = _wrapper_pose_for_env(source, env_idx)
                if pq is None:
                    results[label] = None
                else:
                    obj_in_base = _to_base_frame(base_pq, pq)
                    results[label] = float(
                        np.linalg.norm(tcp_in_base - obj_in_base))

            ent_pq = _sapien_entity_pose(merged_state.active_obj, env_idx)
            if ent_pq is not None:
                obj_in_base_C = _to_base_frame(base_pq, ent_pq)
                results["C"] = float(
                    np.linalg.norm(tcp_in_base - obj_in_base_C))
            else:
                results["C"] = None

            def fmt(v):
                return f"{v:.4f}" if v is not None else "  N/A "
            print(f"{env_idx:>3d} {fmt(results['A']):>22s}  "
                  f"{fmt(results['B']):>22s}  "
                  f"{fmt(results['C']):>22s}")
            n_committed += 1
            if n_committed >= args.max_successes:
                break
    print(f"\n[summary] {n_committed} successes in {step+1} steps "
          f"({time.time()-t0:.1f}s)")
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
