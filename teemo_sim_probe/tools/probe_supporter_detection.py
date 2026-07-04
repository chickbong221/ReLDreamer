"""Diagnostic: instrument the exact code path in FetchCollectContactDataWrapper
that should detect ``link:kitchen_counter-0/drawer3`` as the bowl's supporter,
and print every intermediate value so we can see which line is silently failing.

Runs a single env with the released set_table/pick/024_bowl SAC and prints for
each observation tick:

  * whether drawer3 is present in ``_scene_entities()``
  * ``physics_target_ent.pose.p`` (merged actor) vs actual bowl pose
  * pairwise force between drawer3 and bowl
  * dz / xy_gap between them
  * whether ``_entity_xyz`` returns None for either
  * whether ``best_geom`` gets set at end of the loop

Usage:
    python -m teemo_sim_probe.tools.probe_supporter_detection \
        --ckpt-dir mshab_checkpoints/rl/set_table/pick/024_bowl \
        --steps 60
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _fmt(a, prec=3):
    if a is None:
        return "None"
    a = np.asarray(a, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{x:+.{prec}f}" for x in a) + "]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", default="mshab_checkpoints/rl/set_table/pick/024_bowl")
    ap.add_argument("--task", default="set_table")
    ap.add_argument("--subtask", default="pick")
    ap.add_argument("--obj", default="024_bowl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--plan-index", type=int, default=0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--observe-every", type=int, default=5,
                    help="Print a diagnostic block every N wrapper steps")
    args = ap.parse_args()

    import gymnasium as gym
    import mshab.envs  # noqa: F401
    from mani_skill import ASSET_DIR
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    from mshab.envs.planner import plan_data_from_file
    from mshab.envs.wrappers import (
        FetchActionWrapper,
        FetchDepthObservationWrapper,
        FrameStack,
    )
    from teemo_sim_probe.adapters.collect_contact_data import (
        FetchCollectContactDataWrapper,
        _entity_xyz,
    )
    from teemo_sim_probe.adapters.policy_loader import load_policy
    from teemo_sim_probe.adapters.privileged_state import get_privileged_state
    from teemo_sim_probe.core.entity_identity import stable_entity_key

    RD = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_fp = RD / "task_plans" / args.task / args.subtask / args.split / f"{args.obj}.json"
    if not plan_fp.exists():
        print(f"[ERROR] task plan not found: {plan_fp}")
        return 2
    plan_data = plan_data_from_file(plan_fp)
    spawn = RD / "spawn_data" / args.task / args.subtask / args.split / "spawn_data.pt"

    n_envs = max(1, args.num_envs)
    task_plans = [plan_data.plans[args.plan_index] for _ in range(n_envs)]

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
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn,
        require_build_configs_repeated_equally_across_envs=False,
        add_event_tracker_info=True,
        sensor_configs=dict(width=128, height=128),
    )
    collect = FetchCollectContactDataWrapper(env)
    env = collect
    env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
    env = FrameStack(env, num_stack=3,
                     stacking_keys=["fetch_head_depth", "fetch_hand_depth"])
    env = FetchActionWrapper(env, stationary_base=False, stationary_torso=False,
                             stationary_head=True)
    venv = ManiSkillVectorEnv(env, ignore_terminations=True, max_episode_steps=200)

    obs, _ = venv.reset(seed=args.seed, options=dict(reconfigure=True))
    policy = load_policy(args.ckpt_dir, venv, obs, device=args.device)
    print(f"[policy] kind={policy.kind}")

    env_idx = 0
    for step in range(args.steps):
        action = policy.act(obs)
        obs, _rew, term, trunc, info = venv.step(action)

        if step % args.observe_every != 0:
            continue

        # Mirror what _observe_step does internally.
        actual_state = get_privileged_state(collect, env_idx, mshab_object_name="actual")
        merged_state = get_privileged_state(collect, env_idx, mshab_object_name="merged")
        entities = collect._scene_entities(
            env_idx=env_idx, seg_id_map=actual_state.seg_id_map
        )

        target_ent = actual_state.active_obj
        physics_target_ent = merged_state.active_obj or target_ent
        target_key = f"actor:{collect.obj_id}"

        # Build entity_by_key the same way the wrapper does.
        entity_by_key = {}
        for ent in entities:
            k = stable_entity_key(ent)
            if k:
                entity_by_key[k] = ent
        if target_ent is not None:
            rk = stable_entity_key(target_ent)
            if rk and rk != target_key:
                entity_by_key.pop(rk, None)
        if physics_target_ent is not None:
            rk = stable_entity_key(physics_target_ent)
            if rk and rk != target_key:
                entity_by_key.pop(rk, None)
            entity_by_key[target_key] = physics_target_ent

        drawer_key = "link:kitchen_counter-0/drawer3"
        drawer_ent = entity_by_key.get(drawer_key)

        supported_xyz = _entity_xyz(physics_target_ent, env_idx)
        # Also compute actual bowl xyz as a comparison
        actual_bowl_xyz = _entity_xyz(target_ent, env_idx) if target_ent is not None else None

        print(f"\n=========== wrapper step {step} ===========")
        print(f"info.success[{env_idx}] = {_to_np(info.get('success', np.array([False])))[env_idx] if info.get('success') is not None else '?'}")
        print(f"target_ent (actual)      = {getattr(target_ent, 'name', None)}  "
              f"pose[{env_idx}]={_fmt(actual_bowl_xyz)}")
        print(f"physics_target_ent(merged)= {getattr(physics_target_ent, 'name', None)}  "
              f"pose[{env_idx}]={_fmt(supported_xyz)}")
        print(f"n entities in _scene_entities: {len(entities)}")
        print(f"drawer3 in entity_by_key : {drawer_ent is not None}")

        if drawer_ent is None:
            # Show every 'drawer' or 'kitchen_counter' key that IS present.
            candidates = [k for k in entity_by_key if
                          ("drawer" in k.lower() or "kitchen_counter" in k.lower())]
            print(f"  keys containing drawer/kitchen_counter: {candidates}")
        else:
            drawer_xyz = _entity_xyz(drawer_ent, env_idx)
            print(f"drawer3.pose[{env_idx}]     = {_fmt(drawer_xyz)}")
            if supported_xyz is not None and drawer_xyz is not None:
                dz = float(supported_xyz[2] - drawer_xyz[2])
                xy = float(np.linalg.norm(supported_xyz[:2] - drawer_xyz[:2]))
                print(f"  dz(bowl - drawer3) = {dz:+.4f}   xy_gap = {xy:.4f}")
                in_gate = (-0.15 <= dz <= 0.5) and (xy <= 0.4)
                print(f"  in geometric gate  : {in_gate}")

            # Pairwise force between drawer3 and bowl (both merged and actual).
            try:
                f_merged = _to_np(
                    actual_state.scene.get_pairwise_contact_forces(
                        drawer_ent, physics_target_ent
                    )
                )
                f_merged_env = (
                    f_merged[env_idx] if f_merged.ndim == 2 else f_merged
                )
                print(f"  |F| drawer3-merged_bowl[{env_idx}] = {np.linalg.norm(f_merged_env):.4f} N   "
                      f"vector={_fmt(f_merged_env)}")
            except Exception as e:
                print(f"  |F| drawer3-merged_bowl: EXCEPTION {e!r}")

            try:
                f_actual = _to_np(
                    actual_state.scene.get_pairwise_contact_forces(
                        drawer_ent, target_ent
                    )
                )
                f_actual_env = (
                    f_actual[env_idx] if f_actual.ndim == 2 else f_actual
                )
                print(f"  |F| drawer3-actual_bowl[{env_idx}] = {np.linalg.norm(f_actual_env):.4f} N   "
                      f"vector={_fmt(f_actual_env)}")
            except Exception as e:
                print(f"  |F| drawer3-actual_bowl: EXCEPTION {e!r}")

        # Also report the state of _episode_supports and _episode_obj_contacts
        # for this env so far.
        supports = collect._episode_supports[env_idx]
        contacts = collect._episode_obj_contacts[env_idx]
        print(f"  _episode_supports so far: {list(supports.keys())}")
        print(f"  _episode_obj_contacts so far: {len(contacts)} entries")

    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
