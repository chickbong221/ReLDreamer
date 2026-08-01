"""Dump scene entity identities for merge-rule / ignore-list design.

Builds N parallel envs (one per distinct build config) and enumerates every
actor and articulation link with the exact identity keys the graph runtime
uses (``stable_node_id`` / ``entity_match_key``). The output feeds two
decisions:

* which entities to hard-ignore in open-admission mode (floor / wall / stage);
* which link families to merge into one node (e.g. drawer2_top/_middle/_bottom
  -> drawer2) while keeping handle-style links separate.

Usage::

    export MS_ASSET_DIR=/root/.maniskill
    python -m teemo_sim_probe.tools.dump_scene_entities \
        --task prepare_groceries --task set_table --task tidy_house \
        --subtask pick --num-build-configs 6

Writes ``teemo_sim_probe/outputs/scene_entities_<task>_<subtask>.json`` per
task and prints a grouped summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import List, Optional


HERE = Path(__file__).resolve().parent.parent


def _entity_record(entity) -> dict:
    from teemo_sim_probe.core.entity_identity import (
        canonical_scene_name,
        entity_name,
        stable_node_id,
    )
    from teemo_sim_probe.core.whitelist import entity_match_key

    name = entity_name(entity)
    return dict(
        raw_name=name,
        canonical=canonical_scene_name(name),
        node_id=stable_node_id(entity),
        match_key=entity_match_key(entity),
    )


_STRUCTURE_HINTS = ("stage", "ground", "wall", "floor", "background", "ceiling")


def dump_one(
    task: str, subtask: str, split: str, obj: str, n_bc: int, out_dir: Path,
) -> Path:
    import gymnasium as gym
    from mani_skill import ASSET_DIR
    import mshab.envs  # noqa: F401  registers *SubtaskTrain-v0
    from mshab.envs.planner import plan_data_from_file

    rd = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    plan_fp = rd / "task_plans" / task / subtask / split / f"{obj}.json"
    if not plan_fp.exists():
        raise FileNotFoundError(f"missing task plan: {plan_fp}")
    plan_data = plan_data_from_file(plan_fp)

    by_bc: dict = {}
    for p in plan_data.plans:
        by_bc.setdefault(p.build_config_name, p)
    picked = [by_bc[k] for k in sorted(by_bc)][: max(1, n_bc)]

    env = gym.make(
        f"{subtask.capitalize()}SubtaskTrain-v0",
        num_envs=len(picked),
        obs_mode="state",
        sim_backend="gpu",
        robot_uids="fetch",
        control_mode="pd_joint_delta_pos",
        max_episode_steps=10,
        task_plans=picked,
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=rd / "spawn_data" / task / subtask / split / "spawn_data.pt",
        require_build_configs_repeated_equally_across_envs=False,
    )
    try:
        env.reset(seed=0, options=dict(reconfigure=True))
        scene = env.unwrapped.scene

        actors = {}
        for name, actor in (getattr(scene, "actors", {}) or {}).items():
            actors[str(name)] = _entity_record(actor)

        articulations = {}
        for name, art in (getattr(scene, "articulations", {}) or {}).items():
            links = [
                _entity_record(link)
                for link in (getattr(art, "links", []) or [])
            ]
            articulations[str(name)] = dict(
                canonical=_entity_record(art)["canonical"]
                if hasattr(art, "name") else str(name),
                links=links,
            )
    finally:
        env.close()

    out = dict(
        task=task,
        subtask=subtask,
        split=split,
        obj=obj,
        build_configs=[p.build_config_name for p in picked],
        actors=actors,
        articulations=articulations,
    )
    out_path = out_dir / f"scene_entities_{task}_{subtask}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    # ------------------------------------------------------------- summary
    print(f"\n=== {task}/{subtask}  ({len(picked)} build configs) ===")
    match_keys = sorted({r["match_key"] for r in actors.values() if r["match_key"]})
    structure = [k for k in match_keys
                 if any(h in k.lower() for h in _STRUCTURE_HINTS)]
    print(f"[actors] {len(actors)} raw -> {len(match_keys)} unique match keys")
    for k in match_keys:
        tag = "  <-- structure?" if k in structure else ""
        print(f"  {k}{tag}")
    print(f"[articulations] {len(articulations)}")
    for art_name in sorted(articulations):
        entry = articulations[art_name]
        link_keys = sorted({
            l["match_key"] for l in entry["links"] if l["match_key"]
        })
        print(f"  {art_name}  ({len(entry['links'])} links, "
              f"{len(link_keys)} unique keys)")
        for k in link_keys:
            print(f"    {k}")
    print(f"[wrote] {out_path}")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--task", action="append", default=[],
        help="Repeatable; default: prepare_groceries, set_table, tidy_house.",
    )
    p.add_argument(
        "--subtask", default="pick",
        choices=["pick", "place", "open", "close", "navigate"],
    )
    p.add_argument("--split", default="train")
    p.add_argument("--obj", default="all", help="Task-plan file stem.")
    p.add_argument("--num-build-configs", type=int, default=6)
    p.add_argument(
        "--out-dir", default=str(HERE / "outputs"),
        help="Directory for the JSON dumps.",
    )
    args = p.parse_args(argv)

    tasks = args.task or ["prepare_groceries", "set_table", "tidy_house"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for task in tasks:
        try:
            dump_one(
                task, args.subtask, args.split, args.obj,
                args.num_build_configs, out_dir,
            )
        except Exception:
            print(f"[error] {task}/{args.subtask}:")
            traceback.print_exc()
            failed.append(task)
    if failed:
        print(f"\n[summary] failures: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
