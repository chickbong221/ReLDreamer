"""Sanity-check: does the collector, run to an isolated scratch path, produce
a non-empty pkl for 024_bowl with drawer3 in ``supports``?

The wrapper's default save_path is the production pkl at
``$MS_ASSET_DIR/data/robot_success_states/fetch/pick/024_bowl.pkl``. The
regular diagnose tool clobbers that pkl with an empty payload on close(). This
script passes ``out_root`` to a scratch tempdir so it is safe to run alongside
your real dataset.

Usage::

    python -m teemo_sim_probe.tools.verify_collector_writes_supports \\
        --ckpt-dir mshab_checkpoints/rl/set_table/pick/024_bowl \\
        --task set_table --obj 024_bowl --n-success 3 --num-envs 4

Reports, per rollout in the resulting pkl:
    - ``actor:024_bowl`` in ``interacted``?
    - ``link:kitchen_counter-0/drawer3`` seen as supporter / obj_contact?
    - the raw ``supports`` and ``obj_contacts`` rows.
"""

from __future__ import annotations

import argparse
import pickle
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


DRAWER_KEY = "link:kitchen_counter-0/drawer3"
BOWL_KEY = "actor:024_bowl"


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _build_env(task: str, obj_id: str, subtask: str, num_envs: int,
               max_episode_steps: int, out_root: str):
    """Copy of collect_robot_success_states._build_env, except the wrapper's
    save path is redirected to ``out_root`` so we can never overwrite the
    production pkl."""
    import gymnasium as gym
    from mani_skill import ASSET_DIR
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    import mshab.envs  # noqa: F401
    from mshab.envs.planner import plan_data_from_file
    from mshab.envs.wrappers import (
        FetchActionWrapper,
        FetchDepthObservationWrapper,
        FrameStack,
    )
    from teemo_sim_probe.adapters.collect_contact_data import (
        FetchCollectContactDataWrapper,
    )

    RD = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_fp = RD / "task_plans" / task / subtask / "train" / f"{obj_id}.json"
    if not plan_fp.exists():
        raise FileNotFoundError(f"missing task plan: {plan_fp}")
    spawn_data_fp = RD / "spawn_data" / task / subtask / "train" / "spawn_data.pt"
    plan_data = plan_data_from_file(plan_fp)

    n_plans = len(plan_data.plans)
    task_plans = [plan_data.plans[i % n_plans] for i in range(num_envs)]

    env = gym.make(
        f"{subtask.capitalize()}SubtaskTrain-v0",
        num_envs=num_envs,
        obs_mode="rgb+depth+segmentation",
        sim_backend="gpu",
        robot_uids="fetch",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        render_mode="all",
        shader_dir="minimal",
        max_episode_steps=max_episode_steps,
        task_plans=task_plans,
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn_data_fp,
        require_build_configs_repeated_equally_across_envs=False,
        add_event_tracker_info=True,
        sensor_configs=dict(width=128, height=128),
    )

    collect = FetchCollectContactDataWrapper(env, out_root=out_root)
    env = collect
    env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
    env = FrameStack(env, num_stack=3,
                     stacking_keys=["fetch_head_depth", "fetch_hand_depth"])
    env = FetchActionWrapper(env, stationary_base=False,
                             stationary_torso=False, stationary_head=True)
    venv = ManiSkillVectorEnv(env, ignore_terminations=True,
                              max_episode_steps=max_episode_steps)
    return venv, collect


def _commit(venv, collect, info, committed_mask) -> None:
    """Copy of _commit_successes_at_script_level from
    collect_robot_success_states.py."""
    success = info.get("success")
    if success is None:
        return
    n_envs = collect.num_envs
    success_np = _to_np(success).astype(bool).reshape(-1)
    if success_np.size < n_envs:
        success_np = np.pad(success_np, (0, n_envs - success_np.size))
    newly = success_np[:n_envs] & ~committed_mask
    if not newly.any():
        return

    base_env = venv.unwrapped
    agent = base_env.agent
    ptrs = _to_np(base_env.subtask_pointer).astype(int).reshape(-1)
    base_inv = agent.base_link.pose.inv()
    tcp_rel_all = _to_np((base_inv * agent.tcp.pose).raw_pose)
    qpos_all = _to_np(agent.robot.qpos)

    for env_idx in np.where(newly)[0].tolist():
        ptr = int(min(ptrs[env_idx], len(base_env.task_plan) - 1))
        if ptr >= len(base_env.subtask_objs):
            continue
        target = base_env.subtask_objs[ptr]
        if target is None or getattr(target, "pose", None) is None:
            continue
        obj_rel_all = _to_np((base_inv * target.pose).raw_pose)
        obj_pose = (obj_rel_all[env_idx] if obj_rel_all.ndim == 2
                    else obj_rel_all)
        tcp_pose = (tcp_rel_all[env_idx] if tcp_rel_all.ndim == 2
                    else tcp_rel_all)
        qpos = qpos_all[env_idx] if qpos_all.ndim == 2 else qpos_all
        collect.commit_success(int(env_idx), qpos, obj_pose, tcp_pose)
        committed_mask[env_idx] = True
        print(f"    [commit] env={env_idx} committed a success "
              f"(total={len(collect.success_robot_qpos)})")


def run(args) -> Path:
    from teemo_sim_probe.adapters.policy_loader import load_policy

    scratch = Path(tempfile.mkdtemp(prefix="verify_collector_"))
    print(f"[env] scratch out_root = {scratch}")

    venv, collect = _build_env(
        args.task, args.obj, args.subtask, args.num_envs,
        args.max_episode_steps, str(scratch),
    )
    print(f"[env] wrapper save_path = {collect.save_path}")
    assert str(collect.save_path).startswith(str(scratch)), (
        "scratch redirect failed; refusing to run since we'd overwrite prod pkl"
    )

    obs, _ = venv.reset(seed=args.seed, options=dict(reconfigure=True))
    policy = load_policy(args.ckpt_dir, venv, obs, device=args.device)
    if policy.kind == "random":
        print("[warn] policy loader fell back to random actions; "
              "success rate will be ~0")

    n_envs = collect.num_envs
    committed_mask = np.zeros(n_envs, dtype=bool)
    step = 0
    t0 = time.time()
    while True:
        n_succ = len(collect.success_robot_qpos)
        if n_succ >= args.n_success:
            print(f"[ok] reached {n_succ}/{args.n_success} successes "
                  f"in {step} steps ({time.time() - t0:.1f}s)")
            break
        if step >= args.max_total_steps:
            print(f"[cap] hit --max-total-steps={args.max_total_steps} "
                  f"with {n_succ}/{args.n_success} successes")
            break
        action = policy.act(obs)
        obs, _rew, term, trunc, info = venv.step(action)
        _commit(venv, collect, info, committed_mask)
        done = (_to_np(term).astype(bool).reshape(-1)
                | _to_np(trunc).astype(bool).reshape(-1))
        if done.size < n_envs:
            done = np.pad(done, (0, n_envs - done.size))
        committed_mask[done[:n_envs]] = False
        step += 1
        if step % 200 == 0:
            print(f"    [step {step}] successes so far = {n_succ}")

    venv.close()
    return collect.save_path


def dump(pkl_path: Path) -> None:
    print("\n" + "=" * 78)
    print(f"pkl dump: {pkl_path}")
    print("=" * 78)
    if not pkl_path.exists():
        print("[error] pkl was not written")
        return
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"schema         : {data.get('_schema_version')}")
    print(f"entity_key     : {data.get('entity_key')}")
    print(f"subtask_type   : {data.get('subtask_type')}")
    print(f"n_success      : {len(data.get('robot_qpos', []))}")
    rollouts = data.get("interaction_rollouts") or []
    print(f"n_rollouts     : {len(rollouts)}")

    if not rollouts:
        print("\n[verdict] pkl has zero committed rollouts. Collector never "
              "commits a success. Fixing the diagnose overwrite would NOT be "
              "enough on its own -- something in the collector path is broken.")
        return

    saw_bowl_interacted = 0
    saw_drawer_supporter = 0
    saw_drawer_endpoint = 0
    for i, r in enumerate(rollouts):
        interacted_keys = [
            (it or {}).get("key") for it in (r.get("interacted") or [])
        ]
        support_pairs = [
            ((s.get("supporter") or {}).get("key"), s.get("supported_key"),
             float(s.get("force", 0.0)))
            for s in (r.get("supports") or [])
        ]
        contact_endpoints: Counter = Counter()
        for c in (r.get("obj_contacts") or []):
            for k in (c.get("a_key"), c.get("b_key")):
                if k:
                    contact_endpoints[k] += 1

        bowl_interacted = BOWL_KEY in interacted_keys
        drawer_as_supporter = any(
            sup_k == DRAWER_KEY and sd_k == BOWL_KEY
            for sup_k, sd_k, _ in support_pairs
        )
        drawer_endpoint = DRAWER_KEY in contact_endpoints

        saw_bowl_interacted += int(bowl_interacted)
        saw_drawer_supporter += int(drawer_as_supporter)
        saw_drawer_endpoint += int(drawer_endpoint)

        print(f"\n-- rollout {i} --")
        print(f"    interacted keys           : {interacted_keys}")
        print(f"    bowl interacted           : {bowl_interacted}")
        print(f"    supports rows             : {len(support_pairs)}")
        for sup_k, sd_k, f in support_pairs:
            marker = "  <-- drawer3->bowl" if (
                sup_k == DRAWER_KEY and sd_k == BOWL_KEY
            ) else ""
            print(f"        supporter={sup_k} supported={sd_k} "
                  f"|F|={f:.3f}{marker}")
        print(f"    drawer3 as supporter/bowl : {drawer_as_supporter}")
        print(f"    obj_contact endpoints     : "
              f"{sorted(contact_endpoints.keys())}")
        print(f"    drawer3 as obj_contact    : {drawer_endpoint}")

    print("\n" + "=" * 78)
    print("summary across rollouts")
    print("=" * 78)
    n = len(rollouts)
    print(f"    bowl in 'interacted'          : {saw_bowl_interacted}/{n}")
    print(f"    drawer3 supports bowl         : {saw_drawer_supporter}/{n}")
    print(f"    drawer3 obj_contact endpoint  : {saw_drawer_endpoint}/{n}")

    if saw_drawer_supporter > 0:
        print("\n[verdict] drawer3 IS being recorded as bowl's supporter in a "
              "fresh, isolated pkl. Your production pkl is empty solely "
              "because the diagnose tool overwrote it. Fix A + B are enough.")
    elif saw_bowl_interacted > 0:
        print("\n[verdict] bowl IS interacted, but drawer3 is NEVER recorded "
              "as supporter. The diagnose overwrite is only part of the "
              "story; there is a real collector-side supports bug too.")
    else:
        print("\n[verdict] bowl is not even in 'interacted' -- the grasp "
              "branch of _observe_step likely never fires at an observed "
              "tick. Deeper collector timing bug.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt-dir",
                   default="mshab_checkpoints/rl/set_table/pick/024_bowl")
    p.add_argument("--task", default="set_table")
    p.add_argument("--subtask", default="pick")
    p.add_argument("--obj", default="024_bowl")
    p.add_argument("--n-success", type=int, default=3)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--max-total-steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pkl_path = run(args)
    dump(pkl_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
