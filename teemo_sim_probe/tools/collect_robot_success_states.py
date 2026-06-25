"""Collect successful Fetch manipulation rollouts into robot_success_states.

Discovers checkpoints under
``<ckpt-root>/<task>/<subtask>/<target>/policy.pt`` and rolls each policy out
in a vectorised MS-HAB subtask environment wrapped in
the local ``FetchCollectContactDataWrapper``. MS-HAB remains untouched. The
wrapper commits one schema-v3 record per successful environment rollout:
success pose data, every robot-interacted entity, and direct supporters of the
target/interacted entities. On ``close()`` it writes to::

    $MS_ASSET_DIR/data/robot_success_states/fetch/pick/<obj_id>.pkl

``build_affordances.py`` consumes the pose arrays and
``build_subtask_whitelists.py`` consumes ``interaction_rollouts``.

Usage::

    export MS_ASSET_DIR=/root/.maniskill
    python -m teemo_sim_probe.tools.collect_robot_success_states \\
        --ckpt-root mshab_checkpoints/rl \\
        --n-success 30 --num-envs 8

Filters::

    --task tidy_house --task set_table     # only these tasks
    --obj 024_bowl --obj 003_cracker_box   # only these YCB ids
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _discover_work(
    ckpt_root: Path,
    subtask: str,
    task_filter: List[str],
    obj_filter: List[str],
) -> List[Tuple[str, str, Path]]:
    """Find per-object checkpoints, deduplicated by obj_id.

    The wrapper writes one file per obj_id regardless of task, so processing
    the same obj_id twice (e.g. once under ``tidy_house`` and once under
    ``set_table``) would just overwrite. We keep the first task encountered
    in alphabetical order.
    """
    seen: dict = {}
    for pt in sorted(ckpt_root.glob(f"*/{subtask}/*/policy.pt")):
        parts = pt.parts
        if len(parts) < 4:
            continue
        task = parts[-4]
        obj_id = parts[-2]
        if obj_id == "all":
            continue
        if task_filter and task not in task_filter:
            continue
        if obj_filter and obj_id not in obj_filter:
            continue
        seen.setdefault(obj_id, (task, obj_id, pt.parent))
    return [seen[k] for k in sorted(seen)]


def _already_done(
    asset_dir: Path, subtask: str, obj_id: str, min_samples: int,
) -> bool:
    pkl = asset_dir / "robot_success_states" / "fetch" / subtask / f"{obj_id}.pkl"
    if not pkl.exists():
        return False
    try:
        import pickle
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        return (
            int(d.get("_schema_version", 0)) >= 3
            and len(d.get("robot_qpos", [])) >= min_samples
            and len(d.get("interaction_rollouts", [])) >= min_samples
        )
    except Exception:
        return False


def _build_env(task: str, obj_id: str, args):
    """Recreate the wrapper stack the released SAC was trained on, plus the
    ``FetchCollectRobotInitWrapper`` on the inside (so it sees raw success
    info before policy-side obs transforms)."""
    import gymnasium as gym
    from mani_skill import ASSET_DIR
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    import mshab.envs  # noqa: F401  registers PickSubtaskTrain-v0 etc.
    from mshab.envs.planner import plan_data_from_file
    from mshab.envs.wrappers import (
        FetchActionWrapper,
        FetchDepthObservationWrapper,
        FrameStack,
    )
    from teemo_sim_probe.adapters.collect_contact_data import (
        FetchCollectContactDataWrapper as FetchCollectRobotInitWrapper,
    )

    RD = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_fp = RD / "task_plans" / task / args.subtask / "train" / f"{obj_id}.json"
    if not plan_fp.exists():
        raise FileNotFoundError(f"missing task plan: {plan_fp}")
    spawn_data_fp = RD / "spawn_data" / task / args.subtask / "train" / "spawn_data.pt"
    plan_data = plan_data_from_file(plan_fp)
    if not plan_data.plans:
        raise RuntimeError(f"{plan_fp} contained no plans")

    # Cycle plans across envs so we get init-config diversity even when a
    # single task plan file is shorter than --num-envs.
    n_plans = len(plan_data.plans)
    n_envs = max(1, args.num_envs)
    task_plans = [plan_data.plans[i % n_plans] for i in range(n_envs)]

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
        max_episode_steps=args.max_episode_steps,
        task_plans=task_plans,
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn_data_fp,
        require_build_configs_repeated_equally_across_envs=False,
        add_event_tracker_info=True,
        sensor_configs=dict(width=args.sensor_width, height=args.sensor_height),
    )

    # Collect on the INSIDE (closest to base env) so it reads raw
    # ``info["success"]`` and raw ``agent.robot.qpos`` before any policy-side
    # wrapper has a chance to mutate them.
    collect = FetchCollectRobotInitWrapper(env)
    env = collect
    env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
    env = FrameStack(
        env,
        num_stack=args.frame_stack,
        stacking_keys=["fetch_head_depth", "fetch_hand_depth"],
    )
    env = FetchActionWrapper(
        env,
        stationary_base=False,
        stationary_torso=False,
        stationary_head=True,
    )
    venv = ManiSkillVectorEnv(
        env,
        ignore_terminations=True,
        max_episode_steps=args.max_episode_steps,
    )
    return venv, collect, plan_fp


def _collect_one(task: str, obj_id: str, ckpt_dir: Path, args) -> int:
    from teemo_sim_probe.adapters.policy_loader import load_policy

    venv, collect, plan_fp = _build_env(task, obj_id, args)
    print(f"[env] plan={plan_fp.name} num_envs={venv.unwrapped.num_envs}")

    obs, _ = venv.reset(seed=args.seed, options=dict(reconfigure=True))
    policy = load_policy(str(ckpt_dir), venv, obs, device=args.device)
    if policy.kind == "random":
        print(f"[warn] {obj_id}: policy loader fell back to random actions; "
              "success rate will be ~0. Check ckpt-dir and config.yml.")

    n_target = args.n_success
    t0 = time.time()
    step = 0
    last_log = 0
    last_progress_step = 0
    last_progress_n = 0
    while True:
        n_succ = len(collect.success_robot_qpos)
        if n_succ >= n_target:
            print(f"[ok] {obj_id}: reached {n_succ}/{n_target} successes "
                  f"in {step} steps ({time.time() - t0:.1f}s)")
            break
        if step >= args.max_total_steps:
            print(f"[cap] {obj_id}: hit --max-total-steps={args.max_total_steps} "
                  f"with {n_succ}/{n_target} successes")
            break
        # Stall detector: if no new success in --stall-steps env steps, give up.
        if n_succ > last_progress_n:
            last_progress_n = n_succ
            last_progress_step = step
        if step - last_progress_step > args.stall_steps and n_succ > 0:
            print(f"[stall] {obj_id}: no new success in {args.stall_steps} steps; "
                  f"stopping with {n_succ}/{n_target}")
            break

        action = policy.act(obs)
        obs, _rew, _term, _trunc, _info = venv.step(action)
        step += 1
        if step - last_log >= args.log_every:
            last_log = step
            print(f"  [{obj_id}] step={step} successes={n_succ} "
                  f"({step / max(time.time() - t0, 1e-9):.1f} steps/s)")

    # close() flushes the pickle to disk via FetchCollectRobotInitWrapper.
    venv.close()
    n_final = len(collect.success_robot_qpos)
    print(f"[wrote] {collect.save_path}  ({n_final} samples)")
    return n_final


def parse_args(argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--ckpt-root", default="mshab_checkpoints/rl",
        help="Root containing <task>/<subtask>/<target>/policy.pt subtrees.",
    )
    p.add_argument(
        "--subtask", default="pick", choices=["pick", "open", "close"],
        help="Manipulation subtask checkpoint tree to collect.",
    )
    p.add_argument(
        "--n-success", type=int, default=30,
        help="Target successful picks per object before stopping (default 30).",
    )
    p.add_argument(
        "--max-total-steps", type=int, default=20000,
        help="Hard cap on env steps per object (default 20000).",
    )
    p.add_argument(
        "--stall-steps", type=int, default=4000,
        help="Stop if no new success in this many steps (default 4000).",
    )
    p.add_argument(
        "--max-episode-steps", type=int, default=200,
        help="Per-episode step cap (default 200, same as MS-HAB training).",
    )
    p.add_argument(
        "--num-envs", type=int, default=8,
        help="Parallel envs per object on GPU sim (default 8).",
    )
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--sensor-width", type=int, default=128)
    p.add_argument("--sensor-height", type=int, default=128)
    p.add_argument(
        "--task", action="append", default=[],
        help="Filter to specific task(s); repeatable. Default: all.",
    )
    p.add_argument(
        "--obj", action="append", default=[],
        help="Filter to specific YCB id(s); repeatable. Default: all.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--asset-dir",
        default=str(
            Path(os.environ.get("MS_ASSET_DIR", os.path.expanduser("~/.maniskill")))
            / "data"
        ),
        help="Data root (default: $MS_ASSET_DIR/data, then ~/.maniskill/data). "
             "This is the directory that contains robot_success_states/, "
             "scene_datasets/, etc -- the same dir mani_skill.ASSET_DIR points at.",
    )
    p.add_argument(
        "--no-skip-done", action="store_true",
        help="Re-collect even if a .pkl with >= n-success rows already exists.",
    )
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    ckpt_root = Path(args.ckpt_root).resolve()
    asset_dir = Path(args.asset_dir).resolve()

    if not ckpt_root.is_dir():
        print(f"ERROR: --ckpt-root not found: {ckpt_root}", file=sys.stderr)
        return 2

    work = _discover_work(ckpt_root, args.subtask, args.task, args.obj)
    if not work:
        print("ERROR: no per-object checkpoints matched the filters under "
              f"{ckpt_root}", file=sys.stderr)
        return 2

    print(f"[plan] {len(work)} (task, obj) units; n_success target={args.n_success}")
    print(
        f"[plan] writing under "
        f"{asset_dir}/robot_success_states/fetch/{args.subtask}/"
    )
    ok = 0
    failed: List[str] = []
    for task, obj_id, ckpt_dir in work:
        print(f"\n=== {task}/{obj_id}   ckpt={ckpt_dir.name} ===")
        if not args.no_skip_done and _already_done(
            asset_dir, args.subtask, obj_id, args.n_success,
        ):
            print(f"[skip] {obj_id}: existing .pkl already has "
                  f">= {args.n_success} samples (use --no-skip-done to redo)")
            ok += 1
            continue
        try:
            n = _collect_one(task, obj_id, ckpt_dir, args)
            if n > 0:
                ok += 1
            else:
                failed.append(obj_id)
        except Exception:
            print(f"[error] {task}/{obj_id}:")
            traceback.print_exc()
            failed.append(obj_id)

    print(f"\n[summary] {ok}/{len(work)} units produced .pkl files")
    if failed:
        print(f"[summary] failures: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
