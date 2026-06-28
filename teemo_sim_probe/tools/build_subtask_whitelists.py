"""Mine one-hop per-subtask whitelists from successful rollout interactions.

For each ``(subtask, target)`` the output contains exactly the union of:

* every non-robot entity contacted by an ee link during a successful rollout;
* direct supporters of those contacted entities.

Support is never expanded recursively. Frequency counts are emitted for audit
but do not filter membership. In addition the asset records, per member, the
set of ee-driven interaction types (``contact`` and/or ``grasp``) seen across
rollouts and, at the asset level, the per-relation bin edges derived from the
collector's per-rollout running maxes.
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

import numpy as np

from teemo_sim_probe.core.entity_identity import normalize_asset_key
from teemo_sim_probe.core.whitelist import (
    INTERACTION_CONTACT,
    INTERACTION_CONTAIN,
    INTERACTION_GRASP,
    INTERACTION_SUPPORT,
    WHITELIST_SCHEMA_VERSION,
    derive_bin_edges,
    whitelist_target_slug,
)


# Quantile used to aggregate per-relation samples across all rollouts. Lower
# than 1.0 so a single bad frame (autoreset transient, physics blow-up) can't
# pin the bin edges to absurd values.
_MINER_QUANTILE = 0.9
# Per-relation sanity ceiling. Anything above this is treated as a numerical
# blow-up rather than a meaningful EE-operating range. Tuned for Fetch in a
# kitchen scene (max reach ~1.5 m planar; ~1.2 m vertical).
_BIN_VALUE_CEILING: Dict[str, float] = {
    "planar_distance": 2.0,
    "height_offset": 1.5,
    "planar_distance_change": 2.0,
    "height_offset_change": 1.5,
}


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
        self.interaction_types: Dict[str, Set[str]] = defaultdict(set)
        self.rollout_count = 0
        self.interaction_count: Dict[str, int] = defaultdict(int)
        self.support_count: Dict[str, int] = defaultdict(int)
        self.supports: Dict[str, Set[str]] = defaultdict(set)
        # Aggregated robust value per relation. Filled at payload() time.
        self.bin_value: Dict[str, float] = {}
        # Raw per-relation sample pool across rollouts.
        self.bin_samples: Dict[str, List[float]] = defaultdict(list)

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
            # Every interacted record means an ee link touched the object;
            # grasped=True additionally implies the grasp predicate fired.
            self.interaction_types[key].add(INTERACTION_CONTACT)
            if bool(item.get("grasped")):
                self.interaction_types[key].add(INTERACTION_GRASP)
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
            # Both endpoints of a support pair carry the ``support`` token so
            # the runtime can gate obj-obj support-compatibility on it.
            self.interaction_types[supporter_key].add(INTERACTION_SUPPORT)
            self.interaction_types[supported].add(INTERACTION_SUPPORT)
            supported_pairs_this_rollout.add((supporter_key, supported))
        for supporter_key, _supported in supported_pairs_this_rollout:
            self.support_count[supporter_key] += 1

        # Obj-obj contact events (schema-v6 collector). Both endpoints get the
        # ``contact`` token so the runtime can emit obj-obj
        # contact-compatibility for whitelisted pairs. A separate ``contain``
        # token is opt-in: it's added only when the source data explicitly
        # marks an event as a containment (no MS-HAB env does this today).
        for ev in rollout.get("obj_contacts", []) or []:
            if not isinstance(ev, dict):
                continue
            a_key = normalize_asset_key(ev.get("a_key"))
            b_key = normalize_asset_key(ev.get("b_key"))
            for k in (a_key, b_key):
                if not k:
                    continue
                self.interaction_types[k].add(INTERACTION_CONTACT)
            if bool(ev.get("contain")):
                for k in (a_key, b_key):
                    if k:
                        self.interaction_types[k].add(INTERACTION_CONTAIN)

        raw_samples = rollout.get("bin_samples")
        if isinstance(raw_samples, dict):
            for k, values in raw_samples.items():
                if not isinstance(values, (list, tuple)):
                    continue
                bucket = self.bin_samples[str(k)]
                for v in values:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(fv):
                        bucket.append(fv)

    def _aggregate_bins(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Return ``(robust_value, observed_max)`` per relation.

        The robust value -- fed to ``derive_bin_edges`` -- is the configured
        quantile across all raw samples. A per-relation ceiling caps numerical
        blow-ups so a single bad pickle cannot push the bin edges to absurd
        ranges.
        """
        robust: Dict[str, float] = {}
        observed: Dict[str, float] = {}
        for k, samples in self.bin_samples.items():
            if not samples:
                continue
            value = float(np.quantile(samples, _MINER_QUANTILE))
            obs = float(np.max(samples))
            ceiling = _BIN_VALUE_CEILING.get(k)
            if ceiling is not None and value > ceiling:
                log.warning(
                    "bin '%s' for subtask=%s target=%s capped %.3f -> %.3f "
                    "(observed max=%.3f); raw samples likely contain an "
                    "outlier",
                    k, self.subtask, self.target, value, ceiling, obs,
                )
                value = ceiling
            robust[k] = value
            observed[k] = obs
        return robust, observed

    def payload(self) -> Dict[str, Any]:
        members: Dict[str, Dict[str, Any]] = {}
        for key in sorted(self.roles):
            entry: Dict[str, Any] = {
                "roles": sorted(self.roles[key]),
                "interaction_types": sorted(self.interaction_types.get(key, set())),
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

        # Surface the missing-supporter regression loudly. A pick target that
        # is interacted across every rollout but has zero supporters almost
        # always means the receptacle is resting-contact-only and the force-
        # based detector silenced it; the geometric fallback in the collector
        # should have caught it.
        has_supporter = any("support" in roles for roles in self.roles.values())
        if not has_supporter:
            interacted_targets = [
                k for k, roles in self.roles.items() if "interacted" in roles
            ]
            for k in interacted_targets:
                log.warning(
                    "subtask=%s target=%s: '%s' is interacted across %d "
                    "rollouts but no supporters were recorded; check the "
                    "collector's geometric supporter detection",
                    self.subtask, self.target, k,
                    self.interaction_count.get(k, 0),
                )

        robust, observed = self._aggregate_bins()
        bin_edges = derive_bin_edges(robust)
        return {
            "_schema_version": WHITELIST_SCHEMA_VERSION,
            "subtask": self.subtask,
            "target": self.target,
            "members": members,
            "bin_edges": bin_edges,
            "bin_stats_robust": robust,
            "bin_stats_observed": observed,
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
                "skip %s: schema-v3+ interaction_rollouts required; recollect "
                "with --no-skip-done",
                path,
            )
            continue
        subtask = str(data.get("subtask_type") or path.parent.name)
        target = _target_key(data)
        if not target:
            log.warning("skip %s: entity_key is required", path)
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
        log.error("no successful interaction rollouts found under %s", root)
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
