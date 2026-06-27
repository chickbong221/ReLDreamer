"""Print every scene entity near the pick-bowl target, plus the full
articulation breakdown for whatever the offline miner labelled as the
supporter (``scs-[2,3]_fridge-0`` for ``024_bowl``).

This is a diagnostic: it does not write any asset. It tells you whether the
mined supporter is the right entity, or whether a different articulation link
(a real drawer) is actually under the bowl.

Usage:
    python -m teemo_sim_probe.tools.inspect_bowl_supporters \\
        --task set_table --subtask pick --obj 024_bowl --plan-index 0
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

import mshab.envs  # noqa: F401  registers PickSubtaskTrain-v0
from mani_skill import ASSET_DIR
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mshab.envs.planner import plan_data_from_file
from mshab.envs.wrappers import (
    FetchActionWrapper,
    FetchDepthObservationWrapper,
    FrameStack,
)

from teemo_sim_probe.adapters.privileged_state import (
    _entity_for_env,
    get_privileged_state,
)
from teemo_sim_probe.core.entity_identity import (
    entity_kind,
    entity_name,
    stable_entity_key,
)


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _entity_xyz(ent, env_idx: int) -> Optional[np.ndarray]:
    pose = getattr(ent, "pose", None)
    if pose is None:
        return None
    try:
        p = _to_np(pose.p)
    except Exception:
        return None
    if p.ndim == 2:
        if env_idx >= p.shape[0]:
            return None
        p = p[env_idx]
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.size < 3 or not np.all(np.isfinite(p[:3])):
        return None
    return p[:3]


def _articulation(entity) -> Optional[Any]:
    for attr in ("articulation", "parent_articulation"):
        v = getattr(entity, attr, None)
        if v is not None:
            return v
    for method in ("get_articulation", "get_parent_articulation"):
        fn = getattr(entity, method, None)
        if callable(fn):
            try:
                v = fn()
            except Exception:
                continue
            if v is not None:
                return v
    return None


def _has_visual_mesh(entity, env_idx: int) -> Optional[bool]:
    """Best-effort: does this entity have a rendered shape? Returns None if we
    can't tell from the wrappers we have. Used purely to flag silent supporters
    that the runtime can never see in a camera."""
    target = _entity_for_env(entity, env_idx)
    if target is None:
        return None
    for attr in ("render_bodies", "_render_bodies", "visual_bodies"):
        bodies = getattr(target, attr, None)
        if bodies is not None:
            try:
                return len(bodies) > 0
            except TypeError:
                continue
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="set_table")
    p.add_argument("--subtask", default="pick")
    p.add_argument("--obj", default="024_bowl")
    p.add_argument("--split", default="train")
    p.add_argument("--plan-index", type=int, default=0)
    p.add_argument("--env-idx", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-xy", type=float, default=1.5,
        help="report entities within this xy distance of the bowl",
    )
    p.add_argument(
        "--articulation-substring", default="fridge",
        help="case-insensitive substring; every articulation in seg_id_map "
             "whose name contains this is fully dumped",
    )
    return p.parse_args()


def _make_env(args):
    RD = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_fp = (
        RD / "task_plans" / args.task / args.subtask / args.split
        / f"{args.obj}.json"
    )
    if not plan_fp.exists():
        raise FileNotFoundError(plan_fp)
    plan_data = plan_data_from_file(plan_fp)
    if not (0 <= args.plan_index < len(plan_data.plans)):
        raise IndexError(
            f"--plan-index {args.plan_index} outside [0, {len(plan_data.plans)-1}]"
        )
    spawn = RD / "spawn_data" / args.task / args.subtask / args.split / "spawn_data.pt"

    env = gym.make(
        f"{args.subtask.capitalize()}SubtaskTrain-v0",
        num_envs=max(1, args.num_envs),
        obs_mode="rgb+depth+segmentation",
        sim_backend="gpu",
        robot_uids="fetch",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        render_mode="all",
        shader_dir="minimal",
        max_episode_steps=200,
        task_plans=[plan_data.plans[args.plan_index]],
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn,
        require_build_configs_repeated_equally_across_envs=False,
        add_event_tracker_info=True,
        sensor_configs=dict(width=128, height=128),
    )
    env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
    env = FrameStack(
        env, num_stack=3,
        stacking_keys=["fetch_head_depth", "fetch_hand_depth"],
    )
    env = FetchActionWrapper(
        env, stationary_base=False, stationary_torso=False, stationary_head=True,
    )
    venv = ManiSkillVectorEnv(env, ignore_terminations=True, max_episode_steps=200)
    venv.reset(seed=args.seed, options=dict(reconfigure=True))
    return venv


def _fmt_xyz(p: Optional[np.ndarray]) -> str:
    if p is None:
        return "      ?            ?            ?    "
    return f"{p[0]:>+8.3f} {p[1]:>+8.3f} {p[2]:>+8.3f}"


def main() -> int:
    args = parse_args()
    venv = _make_env(args)
    state = get_privileged_state(venv, args.env_idx, mshab_object_name="actual")
    env_idx = args.env_idx

    bowl = state.active_obj
    bowl_xyz = _entity_xyz(bowl, env_idx) if bowl is not None else None
    bowl_key = stable_entity_key(bowl) if bowl is not None else None
    print("=" * 80)
    print(
        f"target: name={entity_name(bowl)}  key={bowl_key}  "
        f"xyz={_fmt_xyz(bowl_xyz)}"
    )
    print(f"active_subtask_type={state.active_subtask_type}")
    print(f"active_obj_id={state.active_obj_id}")

    # Build the same entity universe the offline miner uses: deduped by
    # stable_entity_key, skipping robot links.
    robot_link_ids = {id(l) for l in state.robot_links}
    robot_names = {entity_name(l) for l in state.robot_links}
    by_key: Dict[str, Any] = {}
    for seg_id, ent in state.seg_id_map.items():
        if not seg_id or ent is None:
            continue
        if id(ent) in robot_link_ids or entity_name(ent) in robot_names:
            continue
        k = stable_entity_key(ent)
        if k and k not in by_key:
            by_key[k] = ent
    print(f"non-robot entities in seg_id_map: {len(by_key)}")

    # ------------------------------------------------------------------ #
    # All entities near the bowl, sorted by dz (lowest below bowl first).
    # ------------------------------------------------------------------ #
    if bowl_xyz is None:
        print("[abort] bowl pose unavailable")
        venv.close()
        return 1
    near: List[Tuple[float, float, str, Any]] = []
    for k, ent in by_key.items():
        p = _entity_xyz(ent, env_idx)
        if p is None:
            continue
        xy_gap = float(np.linalg.norm(bowl_xyz[:2] - p[:2]))
        dz = float(bowl_xyz[2] - p[2])  # positive => candidate is below bowl
        if xy_gap <= args.max_xy:
            near.append((dz, xy_gap, k, ent))
    near.sort(key=lambda t: (abs(t[0]) if t[0] >= 0 else 999, t[1]))

    print()
    print("=" * 80)
    print(f"entities within xy<= {args.max_xy:.2f} m of bowl, sorted")
    print(f"  (positive dz = candidate below bowl)")
    print()
    print(
        f"{'dz':>7s} {'xy_gap':>7s} {'kind':>6s} {'visual?':>8s}  "
        f"{'stable_key'}"
    )
    for dz, xy_gap, k, ent in near:
        kind = entity_kind(ent)
        vis = _has_visual_mesh(ent, env_idx)
        vis_s = "y" if vis is True else ("n" if vis is False else "?")
        marker = " *" if k == bowl_key else "  "
        print(
            f"{dz:>+7.3f} {xy_gap:>7.3f} {kind:>6s} {vis_s:>8s}  {k}{marker}"
        )

    # ------------------------------------------------------------------ #
    # Per-articulation breakdown for any articulation whose name contains
    # the substring (default: "fridge"). Lists every link + pose so you can
    # see if there's a drawer link we should be picking instead of body.
    # ------------------------------------------------------------------ #
    print()
    print("=" * 80)
    sub = args.articulation_substring.lower()
    seen_arts: Dict[int, Any] = {}
    for k, ent in by_key.items():
        if entity_kind(ent) != "link":
            continue
        art = _articulation(ent)
        if art is None:
            continue
        an = entity_name(art).lower() if entity_name(art) else ""
        if sub not in an:
            continue
        seen_arts.setdefault(id(art), art)
    if not seen_arts:
        print(f"no articulations matched substring {args.articulation_substring!r}")
    for art in seen_arts.values():
        print(f"\narticulation: {entity_name(art)}")
        try:
            links = list(art.links)
        except Exception:
            links = []
        for link in links:
            p = _entity_xyz(link, env_idx)
            sk = stable_entity_key(link)
            vis = _has_visual_mesh(link, env_idx)
            vis_s = "y" if vis is True else ("n" if vis is False else "?")
            in_seg = sk in by_key
            print(
                f"  link={entity_name(link):<24s} xyz={_fmt_xyz(p)}  "
                f"visual={vis_s}  in_seg_map={'y' if in_seg else 'n'}  "
                f"key={sk}"
            )

    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
