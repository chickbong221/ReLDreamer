"""Offline miner: domain-level allowed physical entity vocabulary (R4).

Outputs ``e_domain.json``. The mining recipe reads what is already available
without re-running any rollouts:

  1. O_task     -- the ``obj_id`` of every MS-HAB task plan + every articulation
                   ``handle_name`` reachable from the plan tree.
  2. O_support  -- every supporting actor / link referenced as parent/support
                   in the task plan ``Subtask`` dataclasses.
  3. O_state    -- every articulation owner of a handle link (one hop).

The result is a sparse, domain-level vocabulary independent of any particular
test episode. We never recompute from test episodes (R4 provenance rule).

Usage::

    python -m teemo_sim_probe.tools.build_e_domain \\
        --task-plans-dir $MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange/task_plans \\
        --out teemo_sim_probe/configs/e_domain.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


log = logging.getLogger("build_e_domain")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# Conservative role inference for MS-HAB plan fields.
_TASK_FIELDS    = ("obj_id", "object_id", "target_id")
_SUPPORT_FIELDS = ("goal_rectangle_corners", "goal_pos", "support_id",
                   "container_id", "parent_id", "receptacle_id")
_HANDLE_FIELDS  = ("articulation_handle_link_name", "handle_link_name",
                   "handle_name")
_STATE_FIELDS   = ("articulation_id", "articulation_name", "articulation_type")


def _canonical(name: Optional[str]) -> Optional[str]:
    from teemo_sim_probe.core.affordance import canonical_affordance_key
    return canonical_affordance_key(name) if name else None


def _add_role(
    roles: Dict[str, Set[str]], key: Optional[str], role: str
) -> None:
    if not key:
        return
    roles.setdefault(key, set()).add(role)


def _mine_from_task_plan_json(path: Path, roles: Dict[str, Set[str]]) -> int:
    """Walk one task-plan JSON, accumulating roles. Returns subtasks seen."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as exc:
        log.warning("skip %s: %r", path, exc)
        return 0

    plans = data.get("plans") if isinstance(data, dict) else None
    if not isinstance(plans, list):
        plans = [data] if isinstance(data, dict) else []

    n_subtasks = 0
    for plan in plans:
        subtasks = (plan or {}).get("subtasks", []) if isinstance(plan, dict) else []
        for st in subtasks:
            if not isinstance(st, dict):
                continue
            n_subtasks += 1
            for f in _TASK_FIELDS:
                if isinstance(st.get(f), str):
                    _add_role(roles, _canonical(st[f]), "task")
            for f in _HANDLE_FIELDS:
                if isinstance(st.get(f), str):
                    _add_role(roles, _canonical(st[f]), "task")
            for f in _STATE_FIELDS:
                if isinstance(st.get(f), str):
                    _add_role(roles, _canonical(st[f]), "state")
            for f in _SUPPORT_FIELDS:
                if isinstance(st.get(f), str):
                    _add_role(roles, _canonical(st[f]), "support")
    return n_subtasks


def _mine_from_success_states_dir(root: Path, roles: Dict[str, Set[str]]) -> int:
    """Collect O_task from .pkl filenames + obj_id (no FK; just keys)."""
    import pickle

    n = 0
    for pkl in sorted(root.rglob("*.pkl")):
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            log.debug("skip %s: %r", pkl, exc)
            continue
        if isinstance(data, dict):
            key = _canonical(data.get("obj_id") or pkl.stem)
        else:
            key = _canonical(pkl.stem)
        if key:
            _add_role(roles, key, "task")
            n += 1
    return n


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-plans-dir", default=None,
                        help="Root of MS-HAB task_plans/<task>/<subtask>/<split>/.")
    parser.add_argument("--success-states-dir", default=None,
                        help="Optional robot_success_states/ root (adds task keys).")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="Splits to walk inside task_plans (R4: train only).")
    parser.add_argument("--out", required=True, help="Output e_domain.json path.")
    parser.add_argument("--domain", default="mshab")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    roles: Dict[str, Set[str]] = {}
    n_plans = 0

    if args.task_plans_dir:
        root = Path(args.task_plans_dir)
        if not root.is_dir():
            log.error("task-plans-dir %s does not exist", root)
            return 2
        for split in args.splits:
            for plan_fp in sorted(root.glob(f"*/*/{split}/*.json")):
                n_plans += _mine_from_task_plan_json(plan_fp, roles)
        log.info("walked %d task plans from %s", n_plans, root)

    if args.success_states_dir:
        root = Path(args.success_states_dir)
        if not root.is_dir():
            log.warning("success-states-dir %s does not exist", root)
        else:
            n_pkl = _mine_from_success_states_dir(root, roles)
            log.info("collected %d task keys from %s pkls", n_pkl, root)

    if not roles:
        log.error("no entities mined; check --task-plans-dir / --success-states-dir")
        return 2

    entities = {
        k: {"roles": sorted(v)} for k, v in sorted(roles.items())
    }
    payload = {
        "_README": "Domain-level allowed physical entity vocabulary (R4). "
                   "Mined from train-split task plans + success states. "
                   "Never recompute from test episodes.",
        "_schema_version": 1,
        "domain": args.domain,
        "split": "train",
        "n_trajectories": n_plans,
        "entities": entities,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    log.info("wrote %d entities to %s", len(entities), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
