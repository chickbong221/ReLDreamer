"""Run the TEEMO sim-probe on MS-HAB (default PickSubtaskTrain-v0) with the
provided PPO checkpoint.

Task / subtask / split default to whatever the checkpoint's config.yml ships
with. The env is built in obs_mode="rgb+depth+segmentation" (probe reads seg);
the policy receives rgbd only via strip_segmentation. The vector wrapper does
NOT hide task internals -- the probe reads them through env.unwrapped.

Usage:
    python -m teemo_sim_probe.run_mshab_probe \
        --ckpt-dir mshab_checkpoints/rl/tidy_house/pick/all \
        --steps 60 --video
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .configs.loader import load_config
from .core.graph_builder import GraphBuilder
from .viz.overlay import render_overlay
from .viz.graph_draw import render_graph
from .viz.video_writer import write_video
from .adapters.policy_loader import (
    load_checkpoint_config,
    load_ppo_policy,
    strip_segmentation,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="mshab_checkpoints/rl/tidy_house/pick/all",
                   help="dir with config.yml + policy.pt")
    p.add_argument("--task", default=None, help="override; else from config.yml")
    p.add_argument("--subtask", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--camera", default=None)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--include-goals", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                 "outputs", "mshab"))
    p.add_argument("--video", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _resolve_task(args, ckpt_cfg):
    """Pull task/subtask/split from checkpoint config unless overridden."""
    def dig(*keys, default=None):
        cur = ckpt_cfg
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    task = args.task or dig("env", "task") or dig("task") or "tidy_house"
    subtask = args.subtask or dig("env", "subtask") or dig("subtask") or "pick"
    split = args.split or dig("env", "split") or dig("split") or "train"
    return task, subtask, split


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    import gymnasium as gym
    from mani_skill import ASSET_DIR
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    import mshab.envs  # noqa: F401  (registers MS-HAB envs)
    from mshab.envs.planner import plan_data_from_file

    # Checkpoint config -> task/subtask/split.
    try:
        ckpt_cfg = load_checkpoint_config(args.ckpt_dir)
    except FileNotFoundError as exc:
        print(exc)
        ckpt_cfg = {}
    task, subtask, split = _resolve_task(args, ckpt_cfg)
    print(f"[env] task={task} subtask={subtask} split={split}")

    REARRANGE_DIR = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_data = plan_data_from_file(
        REARRANGE_DIR / "task_plans" / task / subtask / split / "all.json"
    )
    spawn_data_fp = (
        REARRANGE_DIR / "spawn_data" / task / subtask / split / "spawn_data.pt"
    )

    env_id = f"{subtask.capitalize()}SubtaskTrain-v0"
    env = gym.make(
        env_id,
        num_envs=1,
        # Probe needs segmentation; policy obs are derived by dropping it.
        obs_mode="rgb+depth+segmentation",
        sim_backend="gpu",
        robot_uids="fetch",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        render_mode="rgb_array",
        shader_dir="minimal",
        max_episode_steps=200,
        task_plans=plan_data.plans[:1],
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn_data_fp,
        add_event_tracker_info=True,
        sensor_configs=dict(width=args.width, height=args.height),
    )

    # Mirror the evaluate wrapper stack. ManiSkillVectorEnv does not hide
    # subtask internals; the probe still reaches them via env.unwrapped.
    venv = ManiSkillVectorEnv(
        env,
        max_episode_steps=1000,
        ignore_terminations=True,
    )

    policy = load_ppo_policy(args.ckpt_dir, venv, device=args.device)
    print(f"[policy] kind={policy.kind}")

    cfg = load_config("room_scale")
    builder = GraphBuilder(
        venv, cfg, env_idx=0, env_id=env_id,
        camera=args.camera, include_goals=args.include_goals,
    )

    obs, info = venv.reset(seed=args.seed)

    overlay_paths, graph_paths = [], []
    for frame in range(args.steps):
        graph, masks, cam, rgb = builder.step(obs, frame)

        graph.save(os.path.join(args.out, f"graph_{frame:04d}.json"))
        op = render_overlay(
            rgb, graph, masks,
            os.path.join(args.out, f"overlay_{frame:04d}.png"),
        )
        gp = render_graph(
            graph, os.path.join(args.out, f"graph_{frame:04d}.png"),
        )
        overlay_paths.append(op)
        graph_paths.append(gp)

        if frame == 0:
            _print_mshab_summary(venv)

        action = policy.act(obs)
        obs, reward, terminated, truncated, info = venv.step(action)

    if args.video:
        vid = write_video(
            overlay_paths, graph_paths,
            os.path.join(args.out, "probe.mp4"), fps=5,
        )
        print("video:", vid)

    venv.close()
    print(f"wrote {args.steps} frames to {args.out}")


def _print_mshab_summary(venv):
    e = venv.unwrapped
    print("--- segmentation_id_map (excluding id 0) ---")
    for sid, ent in sorted(e.segmentation_id_map.items()):
        if sid == 0:
            continue
        print(f"  {sid:3d}  {type(ent).__name__:12s}  {getattr(ent, 'name', '?')}")
    print("--- subtask handles ---")
    import numpy as _np
    ptr = int(_np.asarray(e.subtask_pointer.detach().cpu()
              if hasattr(e.subtask_pointer, "detach") else e.subtask_pointer)[0])
    print(f"  subtask_pointer[0] = {ptr}")
    objs = getattr(e, "subtask_objs", [])
    arts = getattr(e, "subtask_articulations", [])
    if ptr < len(objs):
        o = objs[ptr]
        print(f"  active_obj = {getattr(o, 'name', None) if o is not None else None}")
    if ptr < len(arts):
        a = arts[ptr]
        print(f"  active_articulation = "
              f"{getattr(a, 'name', None) if a is not None else None}")


if __name__ == "__main__":
    main()
