"""Full pipeline diagnostic for the pick/024_bowl supporter regression.

Splits the "no supporter recorded" failure into three checkable parts:

1. **Dump** the current ``024_bowl.pkl``:
   - Per-rollout: ``interacted`` keys, ``supports`` records (evidence, force),
     ``obj_contacts`` endpoint keys.
   - Union across rollouts: every entity key the collector ever saw as a
     candidate. If ``link:kitchen_counter-0/drawer3`` is not in this union,
     it never entered ``entity_by_key`` for any observed tick.

2. **Live-probe every plan_index** the collector cycles through. For each plan
   it runs a single env for ``--steps`` ticks and prints, per observation:
   - whether ``drawer3`` is in ``_scene_entities()``
   - drawer3's pose vs the merged bowl's pose
   - pairwise contact force between drawer3 and the merged bowl
   - whether the geometric criteria would have passed (not used by the
     collector anymore, but useful to see how close we are)

3. **Summarize** across plans which ones would have caught drawer3 via force
   and which never would have, so you can tell whether the miss is
   plan-specific (some plans never spawn bowl on drawer3) or systemic
   (collector cannot see drawer3 at all).

Usage::

    python -m teemo_sim_probe.tools.diagnose_bowl_supporter \\
        --ckpt-dir mshab_checkpoints/rl/set_table/pick/024_bowl \\
        --steps 40

Add ``--pkl /path/to/024_bowl.pkl`` to point at a non-default pickle.
Add ``--plan-index N`` to run only one plan instead of all.
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DRAWER_KEY_DEFAULT = "link:kitchen_counter-0/drawer3"


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _fmt(a, prec: int = 3) -> str:
    if a is None:
        return "None"
    a = np.asarray(a, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{x:+.{prec}f}" for x in a) + "]"


# --------------------------------------------------------------------------- #
# Part 1: dump the pkl
# --------------------------------------------------------------------------- #
def _resolve_pkl(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    asset_dir = os.environ.get("MS_ASSET_DIR")
    candidates: List[Path] = []
    if asset_dir:
        candidates.append(
            Path(asset_dir) / "data/robot_success_states/fetch/pick/024_bowl.pkl"
        )
    candidates.extend([
        Path("/mnt/data/tuannl/data/robot_success_states/fetch/pick/024_bowl.pkl"),
        Path.home() / ".maniskill/data/robot_success_states/fetch/pick/024_bowl.pkl",
    ])
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def dump_pkl(pkl_path: Path, drawer_key: str, n_detail: int) -> None:
    print("=" * 80)
    print("PART 1  --  pkl dump")
    print("=" * 80)
    print(f"pkl: {pkl_path}")
    if not pkl_path.exists():
        print("[error] pkl not found")
        return
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"schema         : {data.get('_schema_version')}")
    print(f"entity_key     : {data.get('entity_key')}")
    print(f"subtask_type   : {data.get('subtask_type')}")
    print(f"n_success      : {len(data.get('robot_qpos', []))}")

    rollouts = data.get("interaction_rollouts") or []
    print(f"n_rollouts     : {len(rollouts)}")

    # Aggregates.
    interacted_counter: Counter = Counter()
    supporter_counter: Counter = Counter()
    contact_endpoint_counter: Counter = Counter()
    rollouts_with_supports = 0
    rollouts_with_contacts = 0
    supported_seen: Counter = Counter()

    for r in rollouts:
        for it in r.get("interacted", []) or []:
            k = (it or {}).get("key")
            if k:
                interacted_counter[k] += 1
        supports = r.get("supports", []) or []
        if supports:
            rollouts_with_supports += 1
        for s in supports:
            sup = (s or {}).get("supporter") or {}
            sk = sup.get("key")
            if sk:
                supporter_counter[sk] += 1
            sd = (s or {}).get("supported_key")
            if sd:
                supported_seen[sd] += 1
        contacts = r.get("obj_contacts", []) or []
        if contacts:
            rollouts_with_contacts += 1
        for c in contacts:
            for endpoint_key in (c.get("a_key"), c.get("b_key")):
                if endpoint_key:
                    contact_endpoint_counter[endpoint_key] += 1

    print(f"rollouts with any support record : {rollouts_with_supports}/{len(rollouts)}")
    print(f"rollouts with any obj_contact    : {rollouts_with_contacts}/{len(rollouts)}")

    print("\nunique 'interacted' keys across rollouts:")
    for k, n in sorted(interacted_counter.items(), key=lambda kv: -kv[1]):
        print(f"    {n:3d}x  {k}")
    print("\nunique 'supporter' keys across rollouts:")
    if supporter_counter:
        for k, n in sorted(supporter_counter.items(), key=lambda kv: -kv[1]):
            print(f"    {n:3d}x  {k}")
    else:
        print("    (none)")
    print("\nunique 'supported_key' values across rollouts:")
    if supported_seen:
        for k, n in sorted(supported_seen.items(), key=lambda kv: -kv[1]):
            print(f"    {n:3d}x  {k}")
    else:
        print("    (none)")
    print("\nunique obj_contact endpoint keys across rollouts:")
    if contact_endpoint_counter:
        for k, n in sorted(
            contact_endpoint_counter.items(), key=lambda kv: -kv[1]
        )[:40]:
            print(f"    {n:4d}x  {k}")
        if len(contact_endpoint_counter) > 40:
            print(f"    ... (+{len(contact_endpoint_counter) - 40} more)")
    else:
        print("    (none)")

    # Was drawer3 ever seen at all in this pkl?
    saw_drawer_as_interacted = drawer_key in interacted_counter
    saw_drawer_as_supporter = drawer_key in supporter_counter
    saw_drawer_as_contact_endpoint = drawer_key in contact_endpoint_counter
    print(f"\n*** '{drawer_key}' seen as ***")
    print(f"    interacted key         : {saw_drawer_as_interacted}")
    print(f"    supporter key          : {saw_drawer_as_supporter}")
    print(f"    obj_contact endpoint   : {saw_drawer_as_contact_endpoint}")

    if not (saw_drawer_as_interacted or saw_drawer_as_supporter
            or saw_drawer_as_contact_endpoint):
        print("\n[hypothesis] drawer3 never entered entity_by_key or its "
              "force with the bowl was always <= eps_force during observed "
              "ticks. Part 2 (live probe) will disambiguate.")

    # Detail dump of the first n_detail rollouts.
    if n_detail > 0:
        print(f"\n---- first {min(n_detail, len(rollouts))} rollouts in detail ----")
    for i, r in enumerate(rollouts[:n_detail]):
        print(f"\n=== rollout {i} ===")
        interacted = r.get("interacted", []) or []
        print(f"  interacted keys : {[x.get('key') for x in interacted]}")
        supports = r.get("supports", []) or []
        print(f"  supports        : {len(supports)}")
        for s in supports:
            sup = (s or {}).get("supporter") or {}
            print(
                f"    supporter={sup.get('key')} supported={s.get('supported_key')} "
                f"evidence={s.get('evidence')} force={s.get('force', 0):.3f} "
                f"vertical={s.get('vertical_support')}"
            )
        contacts = r.get("obj_contacts", []) or []
        print(f"  obj_contacts    : {len(contacts)}")
        first_pairs = Counter()
        for c in contacts:
            a, b = c.get("a_key"), c.get("b_key")
            first_pairs[(a, b)] += 1
        for (a, b), n in sorted(first_pairs.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    {n:3d}x  a={a}  b={b}")


# --------------------------------------------------------------------------- #
# Part 2: live probe per plan_index
# --------------------------------------------------------------------------- #
def live_probe(
    ckpt_dir: str,
    task: str,
    subtask: str,
    obj: str,
    split: str,
    plan_indices: List[int],
    steps: int,
    observe_every: int,
    warmup_ticks: int,
    eps_force: float,
    drawer_key: str,
    device: str,
) -> Dict[int, Dict[str, Any]]:
    """Run the collector wrapper on each plan_index and print observation
    diagnostics. Returns a per-plan summary useful for Part 3."""
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

    print("\n" + "=" * 80)
    print("PART 2  --  live probe per plan_index")
    print("=" * 80)

    RD = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_fp = RD / "task_plans" / task / subtask / split / f"{obj}.json"
    if not plan_fp.exists():
        print(f"[error] task plan not found: {plan_fp}")
        return {}
    plan_data = plan_data_from_file(plan_fp)
    spawn = RD / "spawn_data" / task / subtask / split / "spawn_data.pt"

    all_plan_ids = list(range(len(plan_data.plans)))
    if plan_indices:
        plan_indices = [i for i in plan_indices if i in all_plan_ids]
    else:
        plan_indices = all_plan_ids
    print(f"[plan] file={plan_fp.name} total_plans={len(all_plan_ids)} "
          f"running={plan_indices}")

    summary: Dict[int, Dict[str, Any]] = {}

    for plan_idx in plan_indices:
        print(f"\n---- plan_index {plan_idx} ----")
        env = gym.make(
            f"{subtask.capitalize()}SubtaskTrain-v0",
            num_envs=1,
            obs_mode="rgb+depth+segmentation",
            sim_backend="gpu",
            robot_uids="fetch",
            control_mode="pd_joint_delta_pos",
            reward_mode="normalized_dense",
            render_mode="all",
            shader_dir="minimal",
            max_episode_steps=200,
            task_plans=[plan_data.plans[plan_idx]],
            scene_builder_cls=plan_data.dataset,
            spawn_data_fp=spawn,
            require_build_configs_repeated_equally_across_envs=False,
            add_event_tracker_info=True,
            sensor_configs=dict(width=128, height=128),
        )
        collect = FetchCollectContactDataWrapper(env)
        env = collect
        env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
        env = FrameStack(
            env, num_stack=3,
            stacking_keys=["fetch_head_depth", "fetch_hand_depth"],
        )
        env = FetchActionWrapper(
            env, stationary_base=False, stationary_torso=False,
            stationary_head=True,
        )
        venv = ManiSkillVectorEnv(env, ignore_terminations=True,
                                  max_episode_steps=200)
        obs, _ = venv.reset(seed=0, options=dict(reconfigure=True))
        policy = load_policy(ckpt_dir, venv, obs, device=device)

        max_force_this_plan = 0.0
        first_positive_force_step: Optional[int] = None
        drawer_seen_any_step = False
        env_idx = 0

        for step in range(steps):
            action = policy.act(obs)
            obs, _rew, term, trunc, info = venv.step(action)

            # Mirror _observe_step's setup to see exactly what the collector
            # sees, but do NOT modify collect's episode buffers.
            actual = get_privileged_state(collect, env_idx,
                                          mshab_object_name="actual")
            merged = get_privileged_state(collect, env_idx,
                                          mshab_object_name="merged")
            target_ent = actual.active_obj
            physics_target = merged.active_obj or target_ent

            # Build entity_by_key exactly like the wrapper does.
            entities = collect._scene_entities()
            entity_by_key: Dict[str, Any] = {}
            for e in entities:
                k = stable_entity_key(e)
                if k:
                    entity_by_key[k] = e
            if target_ent is not None:
                rk = stable_entity_key(target_ent)
                if rk and rk != "actor:024_bowl":
                    entity_by_key.pop(rk, None)
            if physics_target is not None:
                rk = stable_entity_key(physics_target)
                if rk and rk != "actor:024_bowl":
                    entity_by_key.pop(rk, None)
                entity_by_key["actor:024_bowl"] = physics_target

            drawer = entity_by_key.get(drawer_key)
            drawer_present = drawer is not None
            drawer_seen_any_step = drawer_seen_any_step or drawer_present

            if step % observe_every != 0:
                continue

            print(f"\n  step={step:3d}  success={bool(_to_np(info.get('success', [False]))[env_idx])}"
                  f"  post_warmup={step >= warmup_ticks}")
            bowl_xyz = _entity_xyz(physics_target, env_idx)
            print(f"    merged bowl pose         = {_fmt(bowl_xyz)}")
            print(f"    drawer3 in entity_by_key = {drawer_present}")

            if not drawer_present:
                # Show any kitchen_counter or drawer keys that ARE present.
                near_keys = sorted(
                    k for k in entity_by_key
                    if "kitchen_counter" in k.lower() or "drawer" in k.lower()
                )
                if near_keys:
                    print(f"    kitchen_counter/drawer keys present: {near_keys}")
                else:
                    print(f"    (no kitchen_counter/drawer keys present)")
                continue

            drawer_xyz = _entity_xyz(drawer, env_idx)
            print(f"    drawer3 pose             = {_fmt(drawer_xyz)}")
            if bowl_xyz is not None and drawer_xyz is not None:
                dz = float(bowl_xyz[2] - drawer_xyz[2])
                xy = float(np.linalg.norm(bowl_xyz[:2] - drawer_xyz[:2]))
                print(f"    dz={dz:+.4f}  xy_gap={xy:.4f}")

            try:
                fv_m = _to_np(
                    actual.scene.get_pairwise_contact_forces(drawer, physics_target)
                )
                fv_m_env = fv_m[env_idx] if fv_m.ndim == 2 else fv_m
                f_m = float(np.linalg.norm(fv_m_env))
                print(f"    |F| drawer3-merged_bowl  = {f_m:.4f} N  vec={_fmt(fv_m_env)}")
                max_force_this_plan = max(max_force_this_plan, f_m)
                if (
                    f_m > eps_force
                    and step >= warmup_ticks
                    and first_positive_force_step is None
                ):
                    first_positive_force_step = step
            except Exception as e:
                print(f"    [warn] merged force query failed: {e!r}")

            try:
                fv_a = _to_np(
                    actual.scene.get_pairwise_contact_forces(drawer, target_ent)
                )
                fv_a_env = fv_a[env_idx] if fv_a.ndim == 2 else fv_a
                f_a = float(np.linalg.norm(fv_a_env))
                print(f"    |F| drawer3-actual_bowl  = {f_a:.4f} N  vec={_fmt(fv_a_env)}")
            except Exception as e:
                print(f"    [warn] actual force query failed: {e!r}")

        summary[plan_idx] = dict(
            drawer_seen_any_step=drawer_seen_any_step,
            max_force=max_force_this_plan,
            first_positive_force_step=first_positive_force_step,
        )
        venv.close()

    return summary


# --------------------------------------------------------------------------- #
# Part 3: cross-plan summary
# --------------------------------------------------------------------------- #
def summarize(summary: Dict[int, Dict[str, Any]], eps_force: float,
              warmup_ticks: int) -> None:
    if not summary:
        return
    print("\n" + "=" * 80)
    print("PART 3  --  cross-plan summary")
    print("=" * 80)
    header = (
        f"{'plan_idx':>8s}  {'drawer_present':>15s}  {'max_force_N':>12s}  "
        f"{'first_hit_step':>15s}  verdict"
    )
    print(header)
    print("-" * len(header))
    would_catch = 0
    for plan_idx in sorted(summary):
        s = summary[plan_idx]
        max_f = s["max_force"]
        first = s["first_positive_force_step"]
        seen = s["drawer_seen_any_step"]
        if not seen:
            verdict = "drawer NEVER in seg_map for this plan"
        elif max_f <= eps_force:
            verdict = "drawer visible but force <= eps at all observed ticks"
        elif first is None or first < warmup_ticks:
            verdict = "force > eps only in warmup ticks (skipped)"
        else:
            verdict = "SHOULD be caught by force detector"
            would_catch += 1
        first_str = "-" if first is None else str(first)
        print(
            f"  {plan_idx:>6d}  {str(seen):>15s}  {max_f:>12.3f}  "
            f"{first_str:>15s}  {verdict}"
        )
    print(f"\n{would_catch}/{len(summary)} plans should have caught drawer3 "
          f"via force in the current collector.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pkl", type=Path, default=None,
                   help="Path to 024_bowl.pkl. Defaults to $MS_ASSET_DIR/... .")
    p.add_argument("--n-detail", type=int, default=3,
                   help="How many rollouts to dump in detail (default 3).")
    p.add_argument("--skip-live", action="store_true",
                   help="Only run the pkl dump; skip the live probe.")
    p.add_argument("--ckpt-dir",
                   default="mshab_checkpoints/rl/set_table/pick/024_bowl")
    p.add_argument("--task", default="set_table")
    p.add_argument("--subtask", default="pick")
    p.add_argument("--obj", default="024_bowl")
    p.add_argument("--split", default="train")
    p.add_argument("--plan-index", type=int, default=None,
                   help="Run only this plan_index. Default: all plans in the "
                        "task file.")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--observe-every", type=int, default=2)
    p.add_argument("--warmup-ticks", type=int, default=3,
                   help="Match _RESET_WARMUP_TICKS in the collector (3).")
    p.add_argument("--eps-force", type=float, default=0.05,
                   help="Match _EPS_FORCE_DEFAULT (0.05).")
    p.add_argument("--drawer-key", default=DRAWER_KEY_DEFAULT)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pkl_path = _resolve_pkl(args.pkl)
    dump_pkl(pkl_path, args.drawer_key, args.n_detail)

    if args.skip_live:
        return 0

    plan_indices: List[int] = (
        [args.plan_index] if args.plan_index is not None else []
    )
    summary = live_probe(
        ckpt_dir=args.ckpt_dir,
        task=args.task,
        subtask=args.subtask,
        obj=args.obj,
        split=args.split,
        plan_indices=plan_indices,
        steps=args.steps,
        observe_every=args.observe_every,
        warmup_ticks=args.warmup_ticks,
        eps_force=args.eps_force,
        drawer_key=args.drawer_key,
        device=args.device,
    )
    summarize(summary, args.eps_force, args.warmup_ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
