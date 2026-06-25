"""Mine one-hop per-subtask whitelists from successful rollout interactions.

For each ``(subtask, target)`` the output contains exactly the union of:

* every non-robot entity contacted during a successful rollout;
* direct supporters of those contacted entities.

Support is never expanded recursively. Frequency counts are emitted for audit
but do not filter membership.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from teemo_sim_probe.core.entity_identity import normalize_asset_key
from teemo_sim_probe.core.whitelist import whitelist_target_slug


log = logging.getLogger("build_subtask_whitelists")


def _iter_pickles(root: Path):
    for path in sorted(root.rglob("*.pkl")):
        try:
            with open(path, "rb") as stream:
                payload = pickle.load(stream)
        except Exception as exc:
            log.warning("skip %s: %r", path, exc)
            continue
        if isinstance(payload, dict):
            yield path, payload


class _WhitelistBuilder:
    def __init__(self, subtask: str, target: str):
        self.subtask = subtask
        self.target = target
        self.roles: Dict[str, Set[str]] = defaultdict(set)
        self.kinds: Dict[str, str] = {}
        self.names: Dict[str, str] = {}
        self.rollout_count = 0
        self.interaction_count: Dict[str, int] = defaultdict(int)
        self.support_count: Dict[str, int] = defaultdict(int)
        self.supports: Dict[str, Set[str]] = defaultdict(set)

    def absorb(self, rollout: Dict[str, Any]) -> None:
        self.rollout_count += 1
        interacted_this_rollout: Set[str] = set()
        for item in rollout.get("interacted", []) or []:
            if not isinstance(item, dict):
                continue
            key = normalize_asset_key(item.get("key"), item.get("kind"))
            if not key:
                continue
            self.roles[key].add("interacted")
            self.kinds.setdefault(key, str(item.get("kind") or "other"))
            self.names.setdefault(key, str(item.get("name") or key))
            interacted_this_rollout.add(key)
        for key in interacted_this_rollout:
            self.interaction_count[key] += 1

        # One hop only: supporters are kept only when they directly support an
        # entity that was actually contacted in this successful rollout.  The
        # task target is metadata for file selection, not an injected member.
        supported_roots = set(interacted_this_rollout)
        supported_pairs_this_rollout: Set[Tuple[str, str]] = set()
        for relation in rollout.get("supports", []) or []:
            if not isinstance(relation, dict):
                continue
            supported = normalize_asset_key(relation.get("supported_key"))
            supporter = relation.get("supporter")
            if supported not in supported_roots or not isinstance(supporter, dict):
                continue
            supporter_key = normalize_asset_key(
                supporter.get("key"), supporter.get("kind")
            )
            if not supporter_key or supporter_key == supported:
                continue
            self.roles[supporter_key].add("support")
            self.kinds.setdefault(
                supporter_key, str(supporter.get("kind") or "other")
            )
            self.names.setdefault(
                supporter_key, str(supporter.get("name") or supporter_key)
            )
            self.supports[supporter_key].add(supported)
            supported_pairs_this_rollout.add((supporter_key, supported))
        for supporter_key, _supported in supported_pairs_this_rollout:
            self.support_count[supporter_key] += 1

    def payload(self) -> Dict[str, Any]:
        members: Dict[str, Dict[str, Any]] = {}
        for key in sorted(self.roles):
            entry: Dict[str, Any] = {
                "roles": sorted(self.roles[key]),
                "kind": self.kinds.get(key, "other"),
            }
            if key in self.names:
                entry["name"] = self.names[key]
            if self.interaction_count.get(key):
                entry["interaction_rollouts"] = self.interaction_count[key]
            if self.support_count.get(key):
                entry["support_rollouts"] = self.support_count[key]
                entry["supports"] = sorted(self.supports[key])
            members[key] = entry
        return {
            "_schema_version": 2,
            "subtask": self.subtask,
            "target": self.target,
            "members": members,
            "_n_successful_rollouts": self.rollout_count,
        }


def _target_key(data: Dict[str, Any]) -> Optional[str]:
    return normalize_asset_key(data.get("entity_key"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--success-states-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    root = Path(args.success_states_dir)
    if not root.is_dir():
        log.error("success-states-dir %s does not exist", root)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    builders: Dict[Tuple[str, str], _WhitelistBuilder] = {}
    n_rollouts = 0
    for path, data in _iter_pickles(root):
        rollouts = data.get("interaction_rollouts") or []
        if int(data.get("_schema_version", 0)) < 3 or not rollouts:
            log.warning(
                "skip %s: schema-v3 interaction_rollouts required; recollect "
                "with --no-skip-done",
                path,
            )
            continue
        subtask = str(data.get("subtask_type") or path.parent.name)
        target = _target_key(data)
        if not target:
            log.warning("skip %s: schema-v3 entity_key is required", path)
            continue
        builder = builders.setdefault(
            (subtask, target),
            _WhitelistBuilder(subtask, target),
        )
        for rollout in rollouts:
            if isinstance(rollout, dict):
                builder.absorb(rollout)
                n_rollouts += 1

    if not builders:
        log.error("no schema-v3 successful interaction rollouts found under %s", root)
        return 2

    empty = [
        (subtask, target)
        for (subtask, target), builder in sorted(builders.items())
        if not builder.roles
    ]
    if empty:
        for subtask, target in empty:
            log.error(
                "empty whitelist for subtask=%s target=%s; collection recorded "
                "no robot-interacted entities for successful rollouts",
                subtask, target,
            )
        log.error(
            "refusing to write invalid whitelist assets; recollect with the "
            "current collector and --no-skip-done"
        )
        return 2

    for (subtask, target), builder in sorted(builders.items()):
        filename_target = whitelist_target_slug(target)
        out_path = out_dir / f"{subtask}_{filename_target}.json"
        with open(out_path, "w") as stream:
            json.dump(builder.payload(), stream, indent=2)
        log.info("wrote %s (%d members)", out_path.name, len(builder.roles))

    log.info(
        "mined %d whitelists from %d successful rollouts",
        len(builders), n_rollouts,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
