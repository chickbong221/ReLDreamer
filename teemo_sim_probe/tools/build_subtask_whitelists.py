"""Offline miner: per-subtask whitelists (replaces R4 domain-level E_domain).

Reads the contact pkls produced by
``teemo_sim_probe.adapters.collect_contact_data.FetchCollectContactDataWrapper``
and emits one JSON per (subtask, target):

    out_dir/
        pick_024_bowl.json
        pick_002_master_chef_can.json
        open_<handle_link>.json
        ...

Each file shape (schema_version=1)::

    {
      "_schema_version": 1,
      "subtask": "pick",
      "target":  "024_bowl",
      "members": {
        "024_bowl":            {"roles": ["task"],             "kind": "actor"},
        "drawer3":             {"roles": ["support"],          "kind": "link",
                                "support_frac": 0.83},
        "kitchen_counter":     {"roles": ["support", "state"], "kind": "link",
                                "support_frac": 0.71}
      }
    }

Closure rule:
  * Seed with the target.
  * BFS to depth ``max_hops`` over the contact-pair graph.
  * Filter: an entity must appear in at least ``min_support_frac`` fraction of
    success frames for the (subtask, target). Filters transient incidental
    contacts (a tipped-over neighbor brushing the target on a single rollout).
  * Inclusive on supports (anything with a load-bearing dz), strict on
    incidental contacts. Spec: "err toward inclusion for supports".

Key normalization mirrors ``teemo_sim_probe.core.whitelist.match_key`` so the
emitted strings match runtime exactly:
  * Free actors:   ``canonical_affordance_key(name)``  (e.g. 024_bowl)
  * Articulation links: bare ``name``                  (e.g. drawer3, body)
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


log = logging.getLogger("build_subtask_whitelists")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _normalize_key(name: str, kind: str) -> Optional[str]:
    """Mirror ``teemo_sim_probe.core.whitelist.match_key`` on raw strings."""
    from teemo_sim_probe.core.affordance import canonical_affordance_key
    if not name:
        return None
    if kind == "actor":
        return canonical_affordance_key(name) or name
    if kind == "link":
        return name
    return name


# --------------------------------------------------------------------------- #
# Pkl ingestion
# --------------------------------------------------------------------------- #
def _iter_contact_pkls(root: Path):
    for pkl in sorted(root.rglob("*.pkl")):
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            log.warning("skip %s: %r", pkl, exc)
            continue
        if not isinstance(data, dict):
            continue
        yield pkl, data


# --------------------------------------------------------------------------- #
# Per-(subtask, target) accumulation
# --------------------------------------------------------------------------- #
class _ClosureBuilder:
    """Accumulates contact statistics for one (subtask, target) pair."""

    def __init__(self, subtask: str, target: str):
        self.subtask = subtask
        self.target = target
        # member_key -> set of {role}
        self.roles: Dict[str, Set[str]] = defaultdict(set)
        # member_key -> kind ("actor" / "link" / "other")
        self.kinds: Dict[str, str] = {}
        # member_key -> count of success frames the member appeared in
        self.support_count: Dict[str, int] = defaultdict(int)
        self.contact_count: Dict[str, int] = defaultdict(int)
        # total success frames observed.
        self.frame_count: int = 0

    def absorb(self, contact_graph: dict, max_hops: int, eps_dz: float) -> None:
        """Walk one success frame's contact graph and update aggregates."""
        self.frame_count += 1

        target_name = contact_graph.get("target")
        target_canonical = contact_graph.get("target_canonical") or self.target
        target_key = _normalize_key(target_canonical, "actor")
        if target_key is None:
            return
        self.roles[target_key].add("task")
        self.kinds.setdefault(target_key, "actor")

        # Build adjacency: name -> [(other_name, other_kind, force, dz)].
        adj: Dict[str, List[Tuple[str, str, float, Optional[float]]]] = defaultdict(list)
        # Also remember each name's kind so we can normalize correctly later.
        name_kind: Dict[str, str] = {target_name: "actor"}
        for p in contact_graph.get("pairs", []) or []:
            a = p.get("a"); b = p.get("b")
            if not a or not b:
                continue
            ak = p.get("a_kind") or "other"
            bk = p.get("b_kind") or "other"
            force = float(p.get("force", 0.0))
            dz = p.get("dz")
            adj[a].append((b, bk, force, dz))
            adj[b].append((a, ak, force, dz))
            name_kind.setdefault(a, ak)
            name_kind.setdefault(b, bk)

        # BFS from the target's raw name. We track per-frame visited to avoid
        # double counting one entity within one frame.
        seen_keys: Set[str] = {target_key}
        queue: List[Tuple[str, int]] = [(target_name, 0)]
        while queue:
            here, depth = queue.pop(0)
            if depth >= max_hops:
                continue
            for other, other_kind, force, dz in adj.get(here, []):
                key = _normalize_key(other, other_kind)
                if key is None or key in seen_keys:
                    continue
                seen_keys.add(key)
                self.kinds.setdefault(key, other_kind)
                if dz is not None and abs(dz) >= eps_dz:
                    # Load-bearing vertical contact -- treat as support.
                    self.roles[key].add("support")
                    self.support_count[key] += 1
                else:
                    self.roles[key].add("contact")
                    self.contact_count[key] += 1
                queue.append((other, depth + 1))

    # ----- emit ----------------------------------------------------------- #
    def to_payload(
        self,
        min_support_frac: float,
        min_contact_frac: float,
    ) -> Optional[dict]:
        if self.frame_count == 0:
            return None
        members: Dict[str, dict] = {}
        # Always include the target itself.
        target_key = _normalize_key(self.target, "actor")
        if target_key is None:
            return None
        members[target_key] = {
            "roles": sorted(self.roles.get(target_key) or {"task"}),
            "kind":  self.kinds.get(target_key, "actor"),
        }
        for key, roles in self.roles.items():
            if key == target_key:
                continue
            support_frac = self.support_count.get(key, 0) / self.frame_count
            contact_frac = self.contact_count.get(key, 0) / self.frame_count
            # Spec: err toward inclusion for supports; strict for free contacts.
            if "support" in roles and support_frac < min_support_frac:
                continue
            if roles == {"contact"} and contact_frac < min_contact_frac:
                continue
            entry = {
                "roles": sorted(roles),
                "kind":  self.kinds.get(key, "other"),
            }
            if support_frac > 0:
                entry["support_frac"] = round(support_frac, 3)
            if contact_frac > 0:
                entry["contact_frac"] = round(contact_frac, 3)
            members[key] = entry
        return {
            "_schema_version": 1,
            "subtask":         self.subtask,
            "target":          target_key,
            "members":         members,
            "_n_success_frames": self.frame_count,
        }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--success-states-dir", required=True,
        help="Root of robot_success_states/<robot_uid>/<subtask>/ produced by "
             "FetchCollectContactDataWrapper.",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Directory to write per-subtask whitelist JSONs into.",
    )
    parser.add_argument(
        "--max-hops", type=int, default=2,
        help="BFS depth from the target over the contact-pair graph.",
    )
    parser.add_argument(
        "--eps-dz", type=float, default=0.01,
        help="Vertical-offset (m) threshold above which a contact counts as "
             "load-bearing 'support'.",
    )
    parser.add_argument(
        "--min-support-frac", type=float, default=0.3,
        help="Min fraction of success frames an entity must appear in to be "
             "kept as a 'support' member (default 0.3 == lenient).",
    )
    parser.add_argument(
        "--min-contact-frac", type=float, default=0.6,
        help="Min fraction for entities seen ONLY as incidental contacts "
             "(no support evidence).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    root = Path(args.success_states_dir)
    if not root.is_dir():
        log.error("success-states-dir %s does not exist", root)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders: Dict[Tuple[str, str], _ClosureBuilder] = {}
    n_pkls = 0
    n_frames = 0
    for pkl, data in _iter_contact_pkls(root):
        subtask = data.get("subtask_type") or pkl.parent.name
        obj_id = data.get("obj_id") or pkl.stem
        contact_graphs = data.get("contact_graphs") or []
        if not isinstance(subtask, str) or not isinstance(obj_id, str):
            continue
        if not contact_graphs:
            continue
        n_pkls += 1
        target_key = _normalize_key(obj_id, "actor")
        if target_key is None:
            continue
        bucket = builders.setdefault((subtask, target_key),
                                     _ClosureBuilder(subtask, target_key))
        for cg in contact_graphs:
            if isinstance(cg, dict):
                bucket.absorb(cg, max_hops=args.max_hops, eps_dz=args.eps_dz)
                n_frames += 1

    if not builders:
        log.error("no contact graphs found under %s", root)
        return 2

    n_written = 0
    for (subtask, target), bucket in sorted(builders.items()):
        payload = bucket.to_payload(
            min_support_frac=args.min_support_frac,
            min_contact_frac=args.min_contact_frac,
        )
        if payload is None:
            continue
        out_path = out_dir / f"{subtask}_{target}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=False)
        n_written += 1
        log.info(
            "wrote %s  (%d members, %d frames)",
            out_path.name, len(payload["members"]), bucket.frame_count,
        )

    log.info(
        "mined %d (subtask,target) pairs from %d pkls / %d success frames; "
        "wrote %d whitelist JSONs to %s",
        len(builders), n_pkls, n_frames, n_written, out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
