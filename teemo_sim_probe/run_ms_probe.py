"""Run the TEEMO sim-probe on a plain ManiSkill task (default PickCube-v1).

Fully runnable without any MS-HAB checkpoint. Drives the env with random or
zero actions, builds a per-frame semantic graph, and saves JSON + mask overlay
+ node-graph render per frame, plus an optional stitched video.

Usage:
    python -m teemo_sim_probe.run_ms_probe \
        --env-id PickCube-v1 --steps 40 --actions random --video
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
from .viz.palette import ColorMap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="PickCube-v1")
    p.add_argument("--robot", default="panda")
    p.add_argument("--control-mode", default="pd_joint_delta_pos")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--actions", choices=["random", "zero"], default="random")
    p.add_argument("--camera", default=None, help="sensor name; auto if unset")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--overlay-size", type=float, default=6.0)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--include-goals", action="store_true")
    p.add_argument("--include-background", action="store_true")
    p.add_argument("--include-static-scene", action="store_true")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                 "outputs", "ms"))
    p.add_argument("--sim-backend", default="gpu")
    p.add_argument("--video", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers envs)

    env = gym.make(
        args.env_id,
        obs_mode="rgb+depth+segmentation",
        render_mode="rgb_array",
        robot_uids=args.robot,
        control_mode=args.control_mode,
        num_envs=1,
        sim_backend=args.sim_backend,
        sensor_configs=dict(width=args.width, height=args.height),
    )

    cfg = load_config("tabletop")
    builder = GraphBuilder(
        env, cfg, env_idx=0, env_id=args.env_id,
        camera=args.camera, include_goals=args.include_goals,
        include_background=args.include_background,
        include_static_scene=args.include_static_scene,
    )

    obs, info = env.reset(seed=args.seed)

    colormap = ColorMap()   # shared so colors stay stable across frames
    overlay_paths, graph_paths = [], []
    for frame in range(args.steps):
        graph, masks, cam, rgb = builder.step(obs, frame)

        graph.save(os.path.join(args.out, f"graph_{frame:04d}.json"))
        op = render_overlay(
            rgb, graph, masks,
            os.path.join(args.out, f"overlay_{frame:04d}.png"),
            colormap=colormap, target_inches=args.overlay_size,
        )
        gp = render_graph(
            graph, os.path.join(args.out, f"graph_{frame:04d}.png"),
            colormap=colormap,
        )
        overlay_paths.append(op)
        graph_paths.append(gp)

        if frame == 0:
            _print_seg_map_summary(env)

        if args.actions == "random":
            action = env.action_space.sample()
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

    if args.video:
        vid = write_video(
            overlay_paths, graph_paths,
            os.path.join(args.out, "probe.mp4"), fps=5,
        )
        print("video:", vid)

    env.close()
    print(f"wrote {args.steps} frames to {args.out}")


def _print_seg_map_summary(env):
    """Frame-0 diagnostic: print seg-id -> entity, excluding id 0."""
    from mani_skill.utils.structs import Actor, Link  # noqa
    e = env.unwrapped
    print("--- segmentation_id_map (excluding id 0) ---")
    for sid, ent in sorted(e.segmentation_id_map.items()):
        if sid == 0:
            continue
        kind = type(ent).__name__
        print(f"  {sid:3d}  {kind:8s}  {getattr(ent, 'name', '?')}")
    print("--- ee links ---")
    for attr in ("tcp", "finger1_link", "finger2_link"):
        link = getattr(e.agent, attr, None)
        if link is not None:
            print(f"  {attr}: {getattr(link, 'name', '?')}")


if __name__ == "__main__":
    main()
