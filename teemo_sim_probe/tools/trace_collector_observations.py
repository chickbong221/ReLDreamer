"""Instrument the collector's own ``_observe_step`` to see, per observed tick,
whether drawer3 is actually in entity_by_key AS THE COLLECTOR SEES IT and what
force the collector's own contact query returns.

The regular diagnose tool builds a *replica* of the wrapper's state after
``venv.step`` returns and queries forces through ``actual.scene``. That is
NOT the same code path as ``_observe_step`` running inside ``super().step()``,
which is where the actual pkl data comes from. This script hooks the real
``_observe_step``.

Runs the collector to a scratch dir (never touches production pkl). For each
successful env, prints its per-tick trace so we can tell whether:
    (a) drawer3 was missing from entity_by_key at every observed pre-grasp
        tick (seg_map / _scene_entities bug in multi-env sim), or
    (b) drawer3 was in entity_by_key but the collector's contact query
        returned ~0 N (contact-query timing bug), or
    (c) the successful env's episode ticks were all warmup-skipped.

Usage::

    python -m teemo_sim_probe.tools.trace_collector_observations \\
        --ckpt-dir mshab_checkpoints/rl/set_table/pick/024_bowl \\
        --n-success 2 --num-envs 4 --max-total-steps 3000
"""

from __future__ import annotations

import argparse
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


DRAWER_KEY = "link:kitchen_counter-0/drawer3"
BOWL_KEY = "actor:024_bowl"
COUNTER_BODY_KEY = "link:kitchen_counter-0/body"


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _make_traced_wrapper_cls():
    """Return a subclass of FetchCollectContactDataWrapper that records what
    ``_observe_step`` sees on every call, keyed by (env_idx, wrapper_step)."""
    from teemo_sim_probe.adapters.collect_contact_data import (
        FetchCollectContactDataWrapper,
        _entity_xyz,
    )
    from teemo_sim_probe.core.entity_identity import stable_entity_key
    from teemo_sim_probe.adapters.privileged_state import get_privileged_state

    class Traced(FetchCollectContactDataWrapper):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            # Per-env list of trace records. Cleared on that env's reset.
            self._trace: List[List[Dict[str, Any]]] = [
                [] for _ in range(self.num_envs)
            ]

        def _reset_buffers(self, env_indices=None):
            super()._reset_buffers(env_indices)
            if hasattr(self, "_trace"):
                indices = (
                    range(self.num_envs) if env_indices is None else env_indices
                )
                for idx in indices:
                    self._trace[int(idx)].clear()

        def _observe_step(self, env_idx: int) -> None:
            # Re-read the same state the real _observe_step uses, then run
            # the real logic and record what happened.
            scene = self._base_env.scene
            entities = self._scene_entities()
            entity_keys = [stable_entity_key(e) for e in entities]
            entity_keys = [k for k in entity_keys if k]

            actual_state = get_privileged_state(
                self, env_idx, mshab_object_name="actual"
            )
            target_ent, target_key = self._target(env_idx, actual_state)
            physics_target_ent = target_ent

            merged_state = None
            if self.subtask_type == "pick":
                merged_state = get_privileged_state(
                    self, env_idx, mshab_object_name="merged"
                )
                if getattr(merged_state, "active_obj", None) is not None:
                    physics_target_ent = merged_state.active_obj

            entity_by_key: Dict[str, Any] = {}
            for ent in entities:
                key = stable_entity_key(ent)
                if key:
                    entity_by_key[key] = ent
            # Snapshot the seg-map entry at target_key BEFORE the swap.
            seg_target_ent = entity_by_key.get(target_key)
            seg_target_is_merged = (
                seg_target_ent is physics_target_ent
                if seg_target_ent is not None else None
            )
            drawer_force_via_seg = None
            if seg_target_ent is not None and entity_by_key.get(DRAWER_KEY) is not None:
                try:
                    vec = self._pairwise_force(
                        scene, entity_by_key[DRAWER_KEY], seg_target_ent, env_idx,
                    )
                    if vec is not None:
                        drawer_force_via_seg = float(np.linalg.norm(vec))
                except Exception as e:
                    drawer_force_via_seg = f"exc:{e!r}"
            if target_ent is not None:
                raw_key = stable_entity_key(target_ent)
                if raw_key and raw_key != target_key:
                    entity_by_key.pop(raw_key, None)
            if physics_target_ent is not None:
                raw_key = stable_entity_key(physics_target_ent)
                if raw_key and raw_key != target_key:
                    entity_by_key.pop(raw_key, None)
                entity_by_key[target_key] = physics_target_ent

            drawer = entity_by_key.get(DRAWER_KEY)
            drawer_present = drawer is not None
            body = entity_by_key.get(COUNTER_BODY_KEY)
            body_present = body is not None
            bowl_xyz = _entity_xyz(physics_target_ent, env_idx)
            drawer_xyz = _entity_xyz(drawer, env_idx) if drawer_present else None

            drawer_force = None
            if drawer_present and physics_target_ent is not None:
                try:
                    vec = self._pairwise_force(
                        scene, drawer, physics_target_ent, env_idx,
                    )
                    if vec is not None:
                        drawer_force = float(np.linalg.norm(vec))
                except Exception as e:
                    drawer_force = f"exc:{e!r}"

            # Same query, but against ``actual.active_obj`` (the original,
            # non-merged bowl). If this returns 5 N while ``drawer_force``
            # against ``merged.active_obj`` returns 0, the merged runtime
            # actor is the wrong physics target for contact queries.
            drawer_force_actual = None
            if drawer_present and target_ent is not None:
                try:
                    vec = self._pairwise_force(
                        scene, drawer, target_ent, env_idx,
                    )
                    if vec is not None:
                        drawer_force_actual = float(np.linalg.norm(vec))
                except Exception as e:
                    drawer_force_actual = f"exc:{e!r}"

            body_force = None
            if body_present and physics_target_ent is not None:
                try:
                    vec = self._pairwise_force(
                        scene, body, physics_target_ent, env_idx,
                    )
                    if vec is not None:
                        body_force = float(np.linalg.norm(vec))
                except Exception as e:
                    body_force = f"exc:{e!r}"

            body_force_actual = None
            if body_present and target_ent is not None:
                try:
                    vec = self._pairwise_force(
                        scene, body, target_ent, env_idx,
                    )
                    if vec is not None:
                        body_force_actual = float(np.linalg.norm(vec))
                except Exception as e:
                    body_force_actual = f"exc:{e!r}"

            merged_is_target = (
                physics_target_ent is target_ent
            )
            merged_key = (
                stable_entity_key(physics_target_ent)
                if physics_target_ent is not None else None
            )
            target_raw_key = (
                stable_entity_key(target_ent) if target_ent is not None else None
            )

            # Now invoke the real observation logic (which will populate
            # _episode_supports / _episode_obj_contacts).
            super()._observe_step(env_idx)

            supports_snapshot = list(self._episode_supports[env_idx].keys())
            interacted_snapshot = list(self._episode_interacted[env_idx].keys())

            self._trace[env_idx].append({
                "wrapper_step": self._step_count,
                "episode_tick": self._episode_ticks[env_idx],
                "n_entities": len(entity_keys),
                "drawer_in_ebk": drawer_present,
                "body_in_ebk": body_present,
                "bowl_z": None if bowl_xyz is None else float(bowl_xyz[2]),
                "drawer_z": None if drawer_xyz is None else float(drawer_xyz[2]),
                "drawer_force": drawer_force,
                "drawer_force_actual": drawer_force_actual,
                "drawer_force_via_seg": drawer_force_via_seg,
                "seg_target_present": seg_target_ent is not None,
                "seg_target_is_merged": seg_target_is_merged,
                "body_force": body_force,
                "body_force_actual": body_force_actual,
                "merged_is_target": merged_is_target,
                "merged_key": merged_key,
                "target_raw_key": target_raw_key,
                "supports_pairs": supports_snapshot,
                "interacted_keys": interacted_snapshot,
            })

    return Traced


def _build_env(task, obj_id, subtask, num_envs, max_episode_steps, out_root):
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

    Traced = _make_traced_wrapper_cls()

    RD = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_fp = RD / "task_plans" / task / subtask / "train" / f"{obj_id}.json"
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
    collect = Traced(env, out_root=out_root)
    env = collect
    env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
    env = FrameStack(env, num_stack=3,
                     stacking_keys=["fetch_head_depth", "fetch_hand_depth"])
    env = FetchActionWrapper(env, stationary_base=False,
                             stationary_torso=False, stationary_head=True)
    venv = ManiSkillVectorEnv(env, ignore_terminations=True,
                              max_episode_steps=max_episode_steps)
    return venv, collect


def _commit(venv, collect, info, committed_mask, success_envs):
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
        # Snapshot the collector's trace for this env AT THE MOMENT of commit,
        # BEFORE commit_success re-runs _observe_step (which would append one
        # more trace record and mutate state).
        trace_snapshot = [dict(r) for r in collect._trace[int(env_idx)]]
        collect.commit_success(int(env_idx), qpos, obj_pose, tcp_pose)
        committed_mask[env_idx] = True
        success_envs.append({
            "env_idx": int(env_idx),
            "wrapper_step_at_commit": collect._step_count,
            "trace_before_commit": trace_snapshot,
            "trace_after_commit": [
                dict(r) for r in collect._trace[int(env_idx)]
            ],
            "supports_committed": list(
                collect.interaction_rollouts[-1].get("supports", [])
            ),
            "interacted_committed": list(
                collect.interaction_rollouts[-1].get("interacted", [])
            ),
            "obj_contacts_committed": list(
                collect.interaction_rollouts[-1].get("obj_contacts", [])
            ),
        })


def run(args) -> List[Dict[str, Any]]:
    from teemo_sim_probe.adapters.policy_loader import load_policy

    scratch = Path(tempfile.mkdtemp(prefix="trace_collector_"))
    print(f"[env] scratch out_root = {scratch}")

    venv, collect = _build_env(
        args.task, args.obj, args.subtask, args.num_envs,
        args.max_episode_steps, str(scratch),
    )
    assert str(collect.save_path).startswith(str(scratch))

    obs, _ = venv.reset(seed=args.seed, options=dict(reconfigure=True))
    policy = load_policy(args.ckpt_dir, venv, obs, device=args.device)
    if policy.kind == "random":
        print("[warn] random-action fallback")

    n_envs = collect.num_envs
    committed_mask = np.zeros(n_envs, dtype=bool)
    success_envs: List[Dict[str, Any]] = []
    step = 0
    t0 = time.time()
    while True:
        n_succ = len(collect.success_robot_qpos)
        if n_succ >= args.n_success:
            print(f"[ok] {n_succ}/{args.n_success} in {step} steps "
                  f"({time.time() - t0:.1f}s)")
            break
        if step >= args.max_total_steps:
            print(f"[cap] max steps hit; {n_succ}/{args.n_success}")
            break
        action = policy.act(obs)
        obs, _rew, term, trunc, info = venv.step(action)
        _commit(venv, collect, info, committed_mask, success_envs)
        done = (_to_np(term).astype(bool).reshape(-1)
                | _to_np(trunc).astype(bool).reshape(-1))
        if done.size < n_envs:
            done = np.pad(done, (0, n_envs - done.size))
        committed_mask[done[:n_envs]] = False
        step += 1

    venv.close()
    return success_envs


def _fmt_force(v):
    if v is None:
        return "None"
    if isinstance(v, str):
        return v
    return f"{v:.4f}"


def report(success_envs: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print(f"n_success_traced = {len(success_envs)}")
    print("=" * 78)

    for meta in success_envs:
        env_idx = meta["env_idx"]
        wstep = meta["wrapper_step_at_commit"]
        trace = meta["trace_before_commit"]
        supports = meta["supports_committed"]
        interacted = meta["interacted_committed"]
        obj_contacts = meta["obj_contacts_committed"]

        print(f"\n---- env {env_idx}: success at wrapper_step={wstep} ----")
        print(f"    trace has {len(trace)} observed ticks before commit")
        if not trace:
            print("    [!] no observed ticks recorded before commit -- "
                  "warmup skip covered the entire pre-success window")
        drawer_ever_present = any(r["drawer_in_ebk"] for r in trace)
        drawer_force_merged = any(
            isinstance(r["drawer_force"], float) and r["drawer_force"] > 0.05
            for r in trace
        )
        drawer_force_actual = any(
            isinstance(r["drawer_force_actual"], float)
            and r["drawer_force_actual"] > 0.05
            for r in trace
        )
        body_force_merged = any(
            isinstance(r["body_force"], float) and r["body_force"] > 0.05
            for r in trace
        )
        body_force_actual = any(
            isinstance(r["body_force_actual"], float)
            and r["body_force_actual"] > 0.05
            for r in trace
        )
        keys_seen = {r["merged_key"] for r in trace}
        raw_keys_seen = {r["target_raw_key"] for r in trace}
        merged_eq_target = all(r["merged_is_target"] for r in trace)
        print(f"    drawer3 in entity_by_key at ANY tick : {drawer_ever_present}")
        print(f"    drawer3 force>0.05 via MERGED bowl   : {drawer_force_merged}")
        print(f"    drawer3 force>0.05 via ACTUAL bowl   : {drawer_force_actual}")
        print(f"    body    force>0.05 via MERGED bowl   : {body_force_merged}")
        print(f"    body    force>0.05 via ACTUAL bowl   : {body_force_actual}")
        print(f"    merged_key seen across ticks         : {keys_seen}")
        print(f"    actual raw_key seen across ticks     : {raw_keys_seen}")
        print(f"    merged is target at ALL ticks        : {merged_eq_target}")
        seg_present_any = any(r["seg_target_present"] for r in trace)
        seg_is_merged_any = any(
            r["seg_target_is_merged"] is True for r in trace
        )
        seg_is_distinct_any = any(
            r["seg_target_is_merged"] is False for r in trace
        )
        seg_force_gt = any(
            isinstance(r["drawer_force_via_seg"], float)
            and r["drawer_force_via_seg"] > 0.05
            for r in trace
        )
        print(f"    seg-map entity at target_key at ANY  : {seg_present_any}")
        print(f"    seg-map entity IS merged handle      : {seg_is_merged_any}")
        print(f"    seg-map entity DISTINCT from merged  : {seg_is_distinct_any}")
        print(f"    |F|drawer-SEG > 0.05 at ANY tick     : {seg_force_gt}")

        # Print each observed tick.
        for r in trace:
            print(
                f"      tick={r['episode_tick']:3d} "
                f"drawer_in_ebk={r['drawer_in_ebk']!s:5s} "
                f"body_in_ebk={r['body_in_ebk']!s:5s} "
                f"bowl_z={r['bowl_z']} drawer_z={r['drawer_z']} "
                f"|F|drawer-M={_fmt_force(r['drawer_force'])} "
                f"|F|drawer-A={_fmt_force(r['drawer_force_actual'])} "
                f"|F|body-M={_fmt_force(r['body_force'])} "
                f"|F|body-A={_fmt_force(r['body_force_actual'])} "
                f"|F|drawer-SEG={_fmt_force(r['drawer_force_via_seg'])} "
                f"seg_is_merged={r['seg_target_is_merged']}"
            )

        print(f"    supports_committed  ({len(supports)}):")
        for s in supports:
            sup_k = (s.get("supporter") or {}).get("key")
            sd_k = s.get("supported_key")
            f = s.get("force", 0.0)
            print(f"        supporter={sup_k} supported={sd_k} |F|={f:.3f}")
        print(f"    interacted_committed ({len(interacted)}):")
        for it in interacted:
            print(f"        {it.get('key')} "
                  f"max_ee_force={it.get('max_ee_force')} "
                  f"grasped={it.get('grasped', False)}")
        print(f"    obj_contacts_committed ({len(obj_contacts)}) endpoint keys "
              f"= {sorted({c.get('a_key') for c in obj_contacts} | {c.get('b_key') for c in obj_contacts})}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt-dir",
                   default="mshab_checkpoints/rl/set_table/pick/024_bowl")
    p.add_argument("--task", default="set_table")
    p.add_argument("--subtask", default="pick")
    p.add_argument("--obj", default="024_bowl")
    p.add_argument("--n-success", type=int, default=2)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--max-total-steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    success_envs = run(args)
    report(success_envs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
