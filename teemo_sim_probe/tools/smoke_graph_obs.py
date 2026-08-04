"""Live check of the graph observation path. Needs a GPU and mined assets.

The unit tests never touch a simulator, so this is the only thing that
exercises the sensor plumbing: that ``depth+segmentation`` still reaches the
MSHab wrappers, that ``sensor_data[cam]['depth']`` exists in the shape the
appearance pooler assumes, that the mined whitelists cover the split, and that
the packed facts fit ``e_max``.

    python -m teemo_sim_probe.tools.smoke_graph_obs \
        --task maniskill_PickSubtaskTrain-v0 \
        --mshab-task set_table --mshab-obj 024_bowl \
        --num-envs 4 --steps 20
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="maniskill_PickSubtaskTrain-v0")
    p.add_argument("--mshab-task", default="set_table")
    p.add_argument("--mshab-obj", default="024_bowl")
    p.add_argument("--mshab-split", default="train")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--n-max", type=int, default=11)
    p.add_argument("--e-max", type=int, default=256)
    p.add_argument("--whitelist-dir",
                   default="teemo_sim_probe/configs/subtask_whitelists")
    p.add_argument("--num-build-configs", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    from embodied.envs.maniskill import ManiSkill

    graph_cfg = dict(
        enabled=True,
        profile="room_scale",
        thresholds_path=None,
        whitelist_dir=args.whitelist_dir,
        cameras=["fetch_head", "fetch_hand"],
        primary_camera="fetch_head",
        n_max=args.n_max,
        e_max=args.e_max,
        staleness_enabled=True,
        bypass_teemo=False,
    )

    print("building env ...", flush=True)
    env = ManiSkill(
        args.task.split("_", 1)[1],
        num_envs=args.num_envs,
        obs_mode="depth+segmentation",
        image_size=args.image_size,
        control_mode="pd_joint_delta_pos",
        mshab_task=args.mshab_task,
        mshab_split=args.mshab_split,
        mshab_obj=args.mshab_obj,
        mshab_num_build_configs=args.num_build_configs,
        max_episode_steps=100,
        frame_stack=1,
        graph=graph_cfg,
        seed=0,
    )

    space = env.obs_space
    graph_keys = sorted(k for k in space if k.startswith("graph_"))
    if not graph_keys:
        print("FAIL: no graph keys in obs_space; graph.enabled did not take")
        return 1
    print(f"\nobs_space ({len(graph_keys)} graph keys)")
    for key in graph_keys:
        print(f"  {key:24s} {str(space[key].shape):12s} {space[key].dtype}")
    print(f"  entity vocab: {env.graph_vocab_sizes}")

    act = {
        "action": np.zeros((args.num_envs, env.act_space["action"].shape[0]),
                           np.float32),
        "reset": np.ones(args.num_envs, bool),
    }
    obs = env.step(act)
    act["reset"] = np.zeros(args.num_envs, bool)

    n_nodes, n_edges, n_visible, truncated = [], [], [], 0
    feat_seen = np.zeros(args.num_envs, bool)
    for step in range(args.steps):
        act["action"] = np.clip(
            np.random.randn(*act["action"].shape).astype(np.float32) * 0.2,
            -1, 1)
        obs = env.step(act)

        for key in graph_keys:
            value = obs[key]
            expected = (args.num_envs, *space[key].shape)
            if value.shape != expected:
                print(f"FAIL step {step}: {key} shape {value.shape} "
                      f"!= {expected}")
                return 1
            if value.dtype != space[key].dtype:
                print(f"FAIL step {step}: {key} dtype {value.dtype} "
                      f"!= {space[key].dtype}")
                return 1

        n_nodes.append(obs["graph_n_nodes"].astype(np.int64))
        n_edges.append(obs["graph_n_edges"].astype(np.int64))
        n_visible.append(obs["graph_node_vis"].sum(-1))
        truncated += int((obs["graph_n_edges"] >= args.e_max).sum())
        feat_seen |= (obs["graph_node_feat"] > 0).any(axis=(1, 2))

    nodes = np.concatenate(n_nodes)
    edges = np.concatenate(n_edges)
    visible = np.concatenate(n_visible)

    print(f"\nover {args.steps} steps x {args.num_envs} envs")
    print(f"  vertices   min {nodes.min():3d}  mean {nodes.mean():6.2f}  "
          f"max {nodes.max():3d}   (cap {args.n_max})")
    print(f"  visible    min {visible.min():3.0f}  mean {visible.mean():6.2f}  "
          f"max {visible.max():3.0f}")
    print(f"  facts      min {edges.min():3d}  mean {edges.mean():6.2f}  "
          f"max {edges.max():3d}   (cap {args.e_max})")
    print(f"  truncated frames: {truncated}")

    ok = True
    if nodes.max() <= 1:
        print("FAIL: no object vertices; the whitelist admitted nothing")
        ok = False
    if edges.max() == 0:
        print("FAIL: no facts emitted")
        ok = False
    if truncated:
        print(f"WARN: {truncated} frames hit e_max; raise it or facts are lost")
    if not feat_seen.all():
        print("WARN: some envs never produced a nonzero appearance; check that "
              "sensor_data[cam]['depth'] is populated")
    print("\nOK" if ok else "\nFAILED")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
