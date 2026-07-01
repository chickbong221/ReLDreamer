"""Diagnostic: dump the raw ``024_bowl.pkl`` support/contact records.

Answers: is the collector actually writing ``link:kitchen_counter-0/drawer3``
into the per-rollout ``supports`` list, or is it never getting recorded?

Usage:
    python -m teemo_sim_probe.tools.dump_bowl_supports
    # or point at a custom pkl:
    python -m teemo_sim_probe.tools.dump_bowl_supports --pkl /path/to/024_bowl.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def _resolve_default_pkl() -> Path:
    asset_dir = os.environ.get("MS_ASSET_DIR")
    candidates = []
    if asset_dir:
        candidates.append(Path(asset_dir) / "data/robot_success_states/fetch/pick/024_bowl.pkl")
    candidates.extend([
        Path("/mnt/data/tuannl/data/robot_success_states/fetch/pick/024_bowl.pkl"),
        Path.home() / ".maniskill/data/robot_success_states/fetch/pick/024_bowl.pkl",
    ])
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkl", type=Path, default=None,
                    help="Path to <obj>.pkl. Defaults to $MS_ASSET_DIR/data/...")
    ap.add_argument("--n-rollouts", type=int, default=5,
                    help="How many rollouts to dump in detail (default 5)")
    args = ap.parse_args()

    pkl = args.pkl or _resolve_default_pkl()
    print(f"Reading: {pkl}")
    if not pkl.exists():
        print(f"[ERROR] file not found: {pkl}")
        return 2

    with open(pkl, "rb") as f:
        d = pickle.load(f)

    print(f"schema        : {d.get('_schema_version')}")
    print(f"entity_key    : {d.get('entity_key')}")
    print(f"subtask_type  : {d.get('subtask_type')}")
    print(f"n_success     : {len(d.get('robot_qpos', []))}")

    rollouts = d.get("interaction_rollouts", [])
    print(f"n_rollouts    : {len(rollouts)}")

    n_with_supports = sum(1 for r in rollouts if r.get("supports"))
    n_with_contacts = sum(1 for r in rollouts if r.get("obj_contacts"))
    print(f"rollouts with any support record : {n_with_supports}/{len(rollouts)}")
    print(f"rollouts with any obj_contact    : {n_with_contacts}/{len(rollouts)}")

    supporter_keys_seen: dict = {}
    for r in rollouts:
        for s in r.get("supports", []) or []:
            k = (s.get("supporter") or {}).get("key")
            if k:
                supporter_keys_seen[k] = supporter_keys_seen.get(k, 0) + 1
    print(f"unique supporter keys across all rollouts:")
    if supporter_keys_seen:
        for k, n in sorted(supporter_keys_seen.items(), key=lambda kv: -kv[1]):
            print(f"    {n:3d}x  {k}")
    else:
        print("    (none)")
    print()

    for i, r in enumerate(rollouts[: args.n_rollouts]):
        print(f"=== rollout {i} ===")
        print(f"  target_key       : {r.get('target_key')}")
        interacted = r.get("interacted", []) or []
        print(f"  interacted keys  : {[x.get('key') for x in interacted]}")
        for it in interacted:
            print(f"      key={it.get('key')}  grasped={it.get('grasped', False)}  "
                  f"max_ee_force={it.get('max_ee_force')}")
        supports = r.get("supports", []) or []
        print(f"  supports ({len(supports)}):")
        for s in supports:
            sup = s.get("supporter") or {}
            pose_sup = s.get("supporter_pose")
            pose_sub = s.get("supported_pose")
            line = (f"    supporter={sup.get('key')}  supported={s.get('supported_key')}"
                    f"  evidence={s.get('evidence')}  force={s.get('force', 0):.3f}"
                    f"  dz={s.get('dz')}  vertical={s.get('vertical_support')}")
            print(line)
            if pose_sup is not None and pose_sub is not None:
                sp = np.asarray(pose_sup, dtype=float)
                sb = np.asarray(pose_sub, dtype=float)
                xy_gap = float(np.linalg.norm(sp[:2] - sb[:2]))
                print(f"      supporter_pose={sp[:3].round(3).tolist()}  "
                      f"supported_pose={sb[:3].round(3).tolist()}  "
                      f"xy_gap_at_record={xy_gap:.3f}")
        ojc = r.get("obj_contacts", []) or []
        print(f"  obj_contacts ({len(ojc)}):")
        for c in ojc[:5]:
            a_pose = c.get("a_pose")
            b_pose = c.get("b_pose")
            extra = ""
            if a_pose is not None and b_pose is not None:
                ap = np.asarray(a_pose, dtype=float)
                bp = np.asarray(b_pose, dtype=float)
                extra = (f"  a_xyz={ap[:3].round(3).tolist()}"
                         f"  b_xyz={bp[:3].round(3).tolist()}")
            print(f"    a={c.get('a_key')}  b={c.get('b_key')}"
                  f"  |F|={c.get('force', 0):.3f}{extra}")
        if len(ojc) > 5:
            print(f"    ... ({len(ojc) - 5} more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
