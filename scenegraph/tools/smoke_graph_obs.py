"""Live check of the graph observation path. Needs a GPU and mined assets.

The unit tests never touch a simulator, so this is the only thing that
exercises the sensor plumbing: that ``rgb+segmentation`` reaches both named
cameras, that DINO pools something nonzero, that terminal transitions carry the
true final frames rather than the next episode's, that the mined whitelists
cover the split, and that the packed facts fit ``e_max``.

    python -m scenegraph.tools.smoke_graph_obs \
        --task maniskill_PickSubtaskTrain-v0 \
        --mshab-task set_table --mshab-obj 024_bowl \
        --num-envs 4 --steps 120
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
    p.add_argument("--image-size", type=int, default=112)
    p.add_argument("--n-max", type=int, default=11)
    p.add_argument("--e-max", type=int, default=256)
    p.add_argument("--whitelist-dir",
                   default="teemo_sim_probe/configs/subtask_whitelists")
    p.add_argument("--num-build-configs", type=int, default=4)
    return p.parse_args()


def _report_state_parity(env):
    """The named-camera wrapper hands back upstream's ``state`` untouched.

    True by construction today, since the subclass returns what
    ``super().observation`` produced. Checked anyway: reimplementing the state
    layout is the one change here that would shift the observation silently.
    """
    from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper

    wrapper = env._named_wrapper
    raw = env._env.unwrapped._last_obs
    ours = wrapper.observation(raw)["state"].cpu().numpy()
    upstream = FlattenRGBDObservationWrapper.observation(
        wrapper, raw)["state"].cpu().numpy()
    same = np.array_equal(ours, upstream)
    print(f"  state width {ours.shape[-1]}, parity with upstream: {same}")
    return same


def main():
    args = parse_args()
    from embodied.envs.maniskill import ManiSkill

    graph_cfg = dict(
        enabled=True,
        profile="room_scale",
        thresholds_path=None,
        whitelist_dir=args.whitelist_dir,
        cameras=["fetch_head", "fetch_hand"],
        n_max=args.n_max,
        e_max=args.e_max,
        k_persist=-1,
        dino_model="dinov2_vits14_reg",
        dino_res=args.image_size,
        app_dim=384,
        staleness_enabled=True,
        bypass_teemo=False,
    )

    print("building env ...", flush=True)
    env = ManiSkill(
        args.task.split("_", 1)[1],
        num_envs=args.num_envs,
        obs_mode="rgb+segmentation",
        image_size=args.image_size,
        control_mode="pd_joint_delta_pos",
        mshab_task=args.mshab_task,
        mshab_split=args.mshab_split,
        mshab_obj=args.mshab_obj,
        mshab_num_build_configs=args.num_build_configs,
        max_episode_steps=100,
        num_frames=1,
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
    for key in graph_keys + ["image_head", "image_hand", "state"]:
        print(f"  {key:24s} {str(space[key].shape):20s} {space[key].dtype}")
    print(f"  entity vocab: {env.graph_vocab_sizes}")
    per_step = sum(
        int(np.prod(space[k].shape)) * np.dtype(space[k].dtype).itemsize
        for k in space if not k.startswith("log/"))
    print(f"  replay bytes per transition: {per_step / 1024:.1f} KiB")

    ok = _report_state_parity(env)

    act = {
        "action": np.zeros((args.num_envs, env.act_space["action"].shape[0]),
                           np.float32),
        "reset": np.ones(args.num_envs, bool),
    }
    obs = env.step(act)
    act["reset"] = np.zeros(args.num_envs, bool)

    n_nodes, n_edges, n_visible, truncated = [], [], [], 0
    app_seen = np.zeros((args.num_envs, 2), bool)
    terminal_frames = 0
    for step in range(args.steps):
        act["action"] = np.clip(
            np.random.randn(*act["action"].shape).astype(np.float32) * 0.2,
            -1, 1)
        prev = obs
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

        ent = obs["graph_node_ent"]
        bbox = obs["graph_node_bbox"].astype(np.float32)
        app = obs["graph_node_app"].astype(np.float32)
        valid = ent != 0
        seen = (bbox[..., 1] > bbox[..., 0]) & (bbox[..., 3] > bbox[..., 2])
        n_nodes.append(valid.sum(-1))
        n_edges.append((obs["graph_edge_rel"] != 0).sum(-1))
        n_visible.append(seen.any(-1).sum(-1))
        truncated += int(((obs["graph_edge_rel"] != 0).sum(-1) >= args.e_max).sum())
        app_seen |= (np.abs(app).sum(-1) > 0).any(1)

        if obs["is_last"].any():
            terminal_frames += int(obs["is_last"].sum())
            # A terminal frame re-emits the previous graph but keeps the true
            # final image; identical images would mean final_observation was
            # aliased to the post-reset render.
            for i in np.nonzero(obs["is_last"])[0]:
                if np.array_equal(obs["image_head"][i], prev["image_head"][i]):
                    print(f"WARN step {step} env {i}: terminal image_head is "
                          "identical to the previous frame")

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
    print(f"  terminal frames:  {terminal_frames}")
    print(f"  overflow drops:   {env._graph.overflow_drops}")

    if nodes.max() <= 1:
        print("FAIL: no object vertices; the whitelist admitted nothing")
        ok = False
    if edges.max() == 0:
        print("FAIL: no facts emitted")
        ok = False
    if not app_seen[:, 0].all():
        print("FAIL: some envs never pooled a head-camera appearance")
        ok = False
    if not app_seen[:, 1].any():
        print("WARN: no env ever pooled a hand-camera appearance; check the "
              "wrist view sees an admitted entity")
    if truncated:
        print(f"WARN: {truncated} frames hit e_max; raise it or facts are lost")
    print("\nOK" if ok else "\nFAILED")
    env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
