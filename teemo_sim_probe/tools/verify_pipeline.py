"""End-to-end verification of the offline pipeline for a given (subtask, obj).

Runs three checks against the current on-disk artifacts:

1. **pkl audit** -- for the target's ``024_bowl.pkl`` (or whatever ``--obj``):
   how many rollouts committed, how many carry ``supports`` rows, how many
   carry ``obj_contacts`` rows, and the specific supporter / obj_contact
   endpoint keys observed. Answers "did the multi-env force query fix
   actually record scene-config-specific supporters end-to-end?"

2. **A-vs-B triage for obj_contacts** -- for the first N successful rollouts,
   for every ``supports`` row we print the (supporter, supported) pair. If
   the supporter is an ordinary scene actor (cracker_box, tuna_fish_can,
   pudding_box, ...) then the support pass IS finding actor-vs-actor
   contacts; the obj_contacts branch just filters them out via the
   ``support_pairs`` skip at collect_contact_data.py:544. That's case B
   (expected). If actor-vs-actor pairs never show up in either supports
   or obj_contacts, that's case A (fix regression on actor-vs-actor).

3. **whitelist JSON audit** -- loads
   ``<whitelist-dir>/<subtask>_<slug>.json`` and prints the members whose
   role includes ``support``, whether ``drawer3`` (or any user-specified
   ``--expect-key``) is present, and its recorded ``supports`` list. This
   is the final check that ``build_subtask_whitelists.py`` propagated the
   pkl's support edges into the runtime asset.

Usage::

    python -m teemo_sim_probe.tools.verify_pipeline \\
        --pkl /root/.maniskill/data/robot_success_states/fetch/pick/024_bowl.pkl \\
        --whitelist-dir <YOUR_WHITELIST_DIR> \\
        --subtask pick --obj 024_bowl \\
        --expect-key link:kitchen_counter-0/drawer3

Non-destructive: reads files only, never writes anything.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Stage 1 + 2: pkl audit
# --------------------------------------------------------------------------- #
def _fmt_record(rec: Dict[str, Any], key_field: str) -> str:
    if not isinstance(rec, dict):
        return str(rec)
    k = rec.get(key_field) or rec.get("key")
    return str(k) if k is not None else "?"


def audit_pkl(pkl_path: Path, expect_key: str, detail_n: int) -> None:
    print("=" * 78)
    print(f"STAGE 1 -- pkl audit: {pkl_path}")
    print("=" * 78)
    if not pkl_path.exists():
        print(f"[error] pkl not found at {pkl_path}")
        return
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    n_rollouts = len(data.get("interaction_rollouts") or [])
    print(f"schema       : {data.get('_schema_version')}")
    print(f"entity_key   : {data.get('entity_key')}")
    print(f"n_success    : {len(data.get('robot_qpos', []))}")
    print(f"n_rollouts   : {n_rollouts}")

    rollouts = data.get("interaction_rollouts") or []
    supporter_counter: Counter = Counter()
    supported_counter: Counter = Counter()
    contact_counter: Counter = Counter()
    n_with_supports = 0
    n_with_contacts = 0
    for r in rollouts:
        supports = r.get("supports") or []
        if supports:
            n_with_supports += 1
        for s in supports:
            supporter_counter[_fmt_record(s.get("supporter") or {}, "key")] += 1
            supported_counter[str(s.get("supported_key") or "?")] += 1
        contacts = r.get("obj_contacts") or []
        if contacts:
            n_with_contacts += 1
        for c in contacts:
            for k in (c.get("a_key"), c.get("b_key")):
                if k:
                    contact_counter[k] += 1

    print(
        f"rollouts with supports : {n_with_supports}/{n_rollouts}\n"
        f"rollouts with obj_contacts : {n_with_contacts}/{n_rollouts}"
    )

    print("\nsupporter keys:")
    for k, n in sorted(supporter_counter.items(), key=lambda kv: -kv[1]):
        marker = "  <-- expected" if k == expect_key else ""
        print(f"    {n:3d}x  {k}{marker}")
    if expect_key not in supporter_counter:
        print(f"    [!] expected supporter key not present: {expect_key}")

    print("\nsupported_key values:")
    for k, n in sorted(supported_counter.items(), key=lambda kv: -kv[1]):
        print(f"    {n:3d}x  {k}")

    print("\nobj_contact endpoint keys:")
    if contact_counter:
        for k, n in sorted(contact_counter.items(), key=lambda kv: -kv[1]):
            print(f"    {n:4d}x  {k}")
    else:
        print("    (none)")

    # -------- Stage 2: A-vs-B triage for obj_contacts --------
    print("\n" + "=" * 78)
    print("STAGE 2 -- obj_contacts A-vs-B triage")
    print("=" * 78)
    print(
        "For each of the first %d rollouts, list every support pair. If the\n"
        "supporter side is an ordinary actor (cracker/tuna/pudding/etc.) it\n"
        "proves the fix DOES query actor-vs-actor contact successfully; those\n"
        "pairs then get filtered out of obj_contacts by the support_pairs skip\n"
        "at collect_contact_data.py:544. That is case B (expected).\n"
        % detail_n
    )
    actor_supporter_seen = False
    for i, r in enumerate(rollouts[:detail_n]):
        supports = r.get("supports") or []
        contacts = r.get("obj_contacts") or []
        print(
            f"-- rollout {i} -- n_supports={len(supports)} "
            f"n_obj_contacts={len(contacts)} "
            f"n_interacted={len(r.get('interacted') or [])}"
        )
        for s in supports:
            sup = (s.get("supporter") or {}).get("key") or "?"
            sd = s.get("supported_key") or "?"
            force = float(s.get("force", 0.0) or 0.0)
            marker = ""
            if sup.startswith("actor:") and sd.startswith("actor:"):
                marker = "  <-- actor-vs-actor edge (obj_contacts skip)"
                actor_supporter_seen = True
            elif sup.startswith("actor:") or sd.startswith("actor:"):
                marker = "  <-- actor-vs-link edge"
                actor_supporter_seen = True
            print(f"    supporter={sup} supported={sd} F={force:.2f}{marker}")

    verdict = (
        "case B (expected): support pass covers the actor pairs; obj_contacts "
        "empty because every real contact is a support edge and gets skipped."
        if actor_supporter_seen else
        "case A (regression): no actor-vs-actor or actor-vs-link supports even "
        "though scene should have some. Fix may need another look."
    )
    print(f"\n[obj_contacts verdict] {verdict}")


# --------------------------------------------------------------------------- #
# Stage 3: whitelist JSON audit
# --------------------------------------------------------------------------- #
def audit_whitelist(
    whitelist_dir: Path, subtask: str, target_key: str, expect_key: str,
) -> None:
    print("\n" + "=" * 78)
    print(f"STAGE 3 -- whitelist JSON audit: {whitelist_dir}")
    print("=" * 78)
    # Match the miner's slug exactly (core/whitelist.py:whitelist_target_slug).
    from teemo_sim_probe.core.whitelist import whitelist_target_slug
    slug = whitelist_target_slug(target_key)
    path = whitelist_dir / f"{subtask}_{slug}.json"
    if not path.exists():
        alternates = sorted(whitelist_dir.glob(f"{subtask}_*{target_key.split(':')[-1]}*.json"))
        if alternates:
            path = alternates[0]
            print(f"[info] using {path.name}")
        else:
            print(f"[error] whitelist JSON not found: {path}")
            print(
                f"        looked for {path.name}; try running "
                f"build_subtask_whitelists first"
            )
            return

    with open(path) as f:
        payload = json.load(f)
    members = payload.get("members") or {}
    print(f"file          : {path.name}")
    print(f"schema        : {payload.get('_schema_version')}")
    print(f"subtask       : {payload.get('subtask')}")
    print(f"target        : {payload.get('target')}")
    print(f"n_members     : {len(members)}")
    print(f"n_rollouts    : {payload.get('_n_successful_rollouts')}")

    supporters = {
        k: entry for k, entry in members.items()
        if "support" in (entry.get("roles") or [])
    }
    print(f"\nsupport-role members ({len(supporters)}):")
    for k, entry in sorted(supporters.items()):
        supports_list = entry.get("supports") or []
        marker = "  <-- expected" if k == expect_key else ""
        print(f"    {k}{marker}")
        print(f"        supports          : {supports_list}")
        print(f"        interaction_types : {entry.get('interaction_types')}")
        print(f"        support_rollouts  : {entry.get('support_rollouts')}")

    if expect_key in members:
        entry = members[expect_key]
        supports_target = target_key in (entry.get("supports") or [])
        role_ok = "support" in (entry.get("roles") or [])
        print(
            f"\n[verdict] expected key '{expect_key}' present: "
            f"role={role_ok}, supports_target={supports_target}"
        )
        if role_ok and supports_target:
            print("          Pipeline end-to-end is CORRECT for this target.")
        else:
            print("          Present but not classified as target's supporter.")
    else:
        print(f"\n[verdict] expected key '{expect_key}' NOT in members. Pipeline")
        print("          did not propagate the pkl edge to the whitelist JSON.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pkl", type=Path, required=True,
                   help="Path to the target's success-states pkl.")
    p.add_argument("--whitelist-dir", type=Path, required=True,
                   help="Directory containing the mined whitelist JSONs.")
    p.add_argument("--subtask", default="pick",
                   help="Subtask type (matches whitelist filename prefix).")
    p.add_argument("--obj", default="024_bowl",
                   help="YCB obj_id; used to derive target_key = actor:<obj>.")
    p.add_argument("--target-key", default=None,
                   help="Explicit target entity key. Defaults to actor:<obj>.")
    p.add_argument("--expect-key",
                   default="link:kitchen_counter-0/drawer3",
                   help="Supporter key we expect to see (drawer3 by default).")
    p.add_argument("--detail-n", type=int, default=8,
                   help="How many rollouts to detail in the Stage 2 triage.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target_key = args.target_key or f"actor:{args.obj}"
    audit_pkl(args.pkl, args.expect_key, args.detail_n)
    audit_whitelist(args.whitelist_dir, args.subtask, target_key, args.expect_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
