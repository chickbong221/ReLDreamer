"""Run the TEEMO sim-probe on MS-HAB (default PickSubtaskTrain-v0) with the PPO
checkpoint, mirroring mshab.envs.make.make_env and mshab/evaluate.py.

Design (verified against installed mshab):
  * Env built in obs_mode="depth+segmentation": the depth wrapper + framestack
    feed the PPO policy exactly the obs it was trained on, while segmentation
    rides along in the unwrapped env's sensor_data for the probe.
  * Wrapper stack copied from make_env: FetchDepthObservationWrapper(cat_state)
    -> FrameStack(num_stack=3, keys=[fetch_head_depth, fetch_hand_depth])
    -> ManiSkillVectorEnv(ignore_terminations=True).
  * Policy: PPOAgent(eval_obs, single_action_space.shape),
    get_action(obs, deterministic=True).
  * Probe reads segmentation from env.unwrapped.get_obs()["sensor_data"]
    ["fetch_head"]["segmentation"] each step.

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
from .core.mask_extractor import read_unwrapped_sensor, depth_to_gray_rgb
from .viz.overlay import render_overlay
from .viz.graph_draw import render_graph
from .viz.video_writer import write_video
from .adapters.policy_loader import load_ppo_policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir",
                   default="mshab_checkpoints/rl/tidy_house/pick/all",
                   help="dir with the PPO .pt (policy.pt / latest.pt / best.pt)")
    p.add_argument("--task", default="tidy_house")
    p.add_argument("--subtask", default="pick")
    p.add_argument("--split", default="train")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--probe-camera", default="fetch_head",
                   help="camera the scene graph is built from")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--include-goals", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                 "outputs", "mshab"))
    p.add_argument("--video", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    import gymnasium as gym
    import torch
    from mani_skill import ASSET_DIR
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    import mshab.envs  # noqa: F401  registers MS-HAB envs
    from mshab.envs.planner import plan_data_from_file
    from mshab.envs.wrappers import FetchDepthObservationWrapper, FrameStack

    task, subtask, split = args.task, args.subtask, args.split
    print(f"[env] task={task} subtask={subtask} split={split}")

    RD = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_data = plan_data_from_file(
        RD / "task_plans" / task / subtask / split / "all.json"
    )
    spawn_data_fp = RD / "spawn_data" / task / subtask / split / "spawn_data.pt"

    env_id = f"{subtask.capitalize()}SubtaskTrain-v0"

    # Build the env in depth+segmentation. The depth wrapper ignores the extra
    # segmentation texture, so the policy obs is unchanged; the probe reads the
    # segmentation from the unwrapped env.
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="depth+segmentation",
        sim_backend="gpu",
        robot_uids="fetch",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        render_mode="all",
        shader_dir="minimal",
        max_episode_steps=200,
        task_plans=plan_data.plans[:1],
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn_data_fp,
        add_event_tracker_info=True,
        sensor_configs=dict(width=args.width, height=args.height),
    )

    # Wrapper stack copied from mshab.envs.make.make_env (PPO path).
    env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
    if args.frame_stack:
        env = FrameStack(
            env,
            num_stack=args.frame_stack,
            stacking_keys=["fetch_head_depth", "fetch_hand_depth"],
        )
    venv = ManiSkillVectorEnv(env, ignore_terminations=True)

    eval_obs, _ = venv.reset(seed=args.seed, options=dict(reconfigure=True))
    policy = load_ppo_policy(args.ckpt_dir, venv, eval_obs, device=args.device)
    print(f"[policy] kind={policy.kind}")

    cfg = load_config("room_scale")
    builder = GraphBuilder(
        venv, cfg, env_idx=0, env_id=env_id,
        camera=args.probe_camera, include_goals=args.include_goals,
    )

    obs = eval_obs
    overlay_paths, graph_paths = [], []
    for frame in range(args.steps):
        # Read segmentation + depth straight from the unwrapped env.
        seg, depth = read_unwrapped_sensor(venv, args.probe_camera, env_idx=0)
        backdrop = depth_to_gray_rgb(depth) if depth is not None else None

        graph, masks, cam, rgb = builder.step(
            obs, frame,
            seg_override=seg, rgb_override=backdrop,
            camera_override=args.probe_camera,
        )

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
            _print_mshab_summary(venv, args.probe_camera)

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


def _print_mshab_summary(venv, camera):
    e = venv.unwrapped
    print("--- segmentation_id_map (excluding id 0) ---")
    for sid, ent in sorted(e.segmentation_id_map.items()):
        if sid == 0:
            continue
        print(f"  {sid:3d}  {type(ent).__name__:12s}  {getattr(ent, 'name', '?')}")
    print(f"--- probe camera: {camera} ---")
    print("--- subtask handles ---")
    ptr = int(np.asarray(
        e.subtask_pointer.detach().cpu() if hasattr(e.subtask_pointer, "detach")
        else e.subtask_pointer)[0])
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
