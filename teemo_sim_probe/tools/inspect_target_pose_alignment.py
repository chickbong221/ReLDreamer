"""Diagnostic: print the TCP and active-target poses per env, side-by-side.

Spins up a PickSubtaskTrain env with N envs, resets, and prints for each env:
  - agent.base_link.pose
  - agent.tcp.pose
  - subtask_objs[ptr].pose (merged "obj_<ptr>")
  - per-env "actual" entity pose resolved via the collector's seg_id_map path
  - the actual scene actor named env-<i>_<obj_id>.pose

If the merged-actor pose differs from the per-env scene actor pose, that's the
collector bug. If TCP - obj distance is reasonable (centimeters) at success but
much larger right after reset (expected, robot hasn't approached yet), that's
not a bug; the question is what the collector sees AT success commit time.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch  # noqa: F401  required by mshab.envs registration

import mshab.envs  # noqa: F401  registers PickSubtaskTrain-v0
from mani_skill import ASSET_DIR
from mshab.envs.planner import plan_data_from_file


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="tidy_house")
    parser.add_argument("--subtask", default="pick", choices=["pick"])
    parser.add_argument("--obj-id", default="002_master_chef_can")
    parser.add_argument("--num-envs", type=int, default=2)
    args = parser.parse_args()

    RD = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_fp = RD / "task_plans" / args.task / args.subtask / "train" / f"{args.obj_id}.json"
    spawn = RD / "spawn_data" / args.task / args.subtask / "train" / "spawn_data.pt"
    pd = plan_data_from_file(plan_fp)
    if not pd.plans:
        print(f"ERROR: empty plan at {plan_fp}")
        return 2
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
    env.reset(seed=0, options=dict(reconfigure=True))
    e = env.unwrapped

    ptr = int(_to_np(e.subtask_pointer)[0])
    merged = e.subtask_objs[ptr]
    print(f"[plan] {plan_fp.name}  num_envs={n_envs}  ptr={ptr}")
    print(f"[merged] name={merged.name} type={type(merged).__name__} "
          f"scene_idxs={_to_np(merged._scene_idxs).tolist()}")

    base = e.agent.base_link
    tcp = e.agent.tcp
    print(f"[base ] name={base.name}  scene_idxs={_to_np(base._scene_idxs).tolist()}")
    print(f"[tcp  ] name={tcp.name}  scene_idxs={_to_np(tcp._scene_idxs).tolist()}")

    base_p = _to_np(base.pose.p)
    tcp_p = _to_np(tcp.pose.p)
    merged_p = _to_np(merged.pose.p)
    print("\nbatched .pose.p shapes:",
          f"base={base_p.shape} tcp={tcp_p.shape} merged={merged_p.shape}")

    seg = dict(getattr(e, "segmentation_id_map", {}))
    print(f"\nseg_id_map size: {len(seg)}")

    print(f"\n{'env':>3s}  {'base.p':>26s}  {'tcp.p':>26s}  "
          f"{'merged.p':>26s}  {'actual.p (scene name match)':>32s}")
    for i in range(n_envs):
        # Find the per-env scene actor whose name is env-<i>_<obj_id>.
        target_name = f"env-{i}_{args.obj_id}"
        actual_pose = None
        for actor in e.scene.actors.values():
            if getattr(actor, "name", None) == target_name:
                actual_pose = _to_np(actor.pose.p)[0] if _to_np(actor.pose.p).ndim == 2 else _to_np(actor.pose.p)
                break
        bp = base_p[i] if base_p.ndim == 2 else base_p
        tp = tcp_p[i] if tcp_p.ndim == 2 else tcp_p
        mp = merged_p[i] if merged_p.ndim == 2 else merged_p
        print(f"{i:>3d}  {str(bp.round(3)):>26s}  {str(tp.round(3)):>26s}  "
              f"{str(mp.round(3)):>26s}  "
              f"{(str(actual_pose.round(3)) if actual_pose is not None else 'NOT FOUND'):>32s}")

    print("\nKey check: does merged.p match actual.p per env?")
    for i in range(n_envs):
        target_name = f"env-{i}_{args.obj_id}"
        actual_pose = None
        for actor in e.scene.actors.values():
            if getattr(actor, "name", None) == target_name:
                ap = _to_np(actor.pose.p)
                actual_pose = ap[0] if ap.ndim == 2 else ap
                break
        mp = merged_p[i] if merged_p.ndim == 2 else merged_p
        if actual_pose is None:
            print(f"  env {i}: actual actor {target_name} NOT FOUND in scene.actors")
            continue
        d = float(np.linalg.norm(mp - actual_pose))
        flag = " <-- MISMATCH" if d > 1e-3 else ""
        print(f"  env {i}: ||merged - actual|| = {d:.4f} m{flag}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
