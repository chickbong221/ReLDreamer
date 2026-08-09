"""Largest fact count a graph of ``n_max`` vertices can carry.

``e_max`` sizes a static array, so every slot costs GPU memory and backward
compute whether or not a fact lands in it. ``smoke_graph_obs`` measures what a
short rollout happened to emit; this bounds what the runtime can ever emit.

Two figures per whitelist. **Every member once** is the floor: those vertices
appear together whenever a scene shows them, so ``e_max`` below it truncates in
ordinary operation. **Every slot filled** is the ceiling, assuming ``n_max - 1``
copies of the densest member -- rigorous, but it needs that many instances of
one asset in a single scene, so it sits far above anything observed. Size
``e_max`` from the smoke run's measured peak between the two.

Union files are skipped: no episode binds one for membership, so a fact count
for one would describe a graph the runtime never builds.

Each gate below reads the same helper the runtime reads, so a change to the
emission rules surfaces here instead of drifting away from them.

    python -m scenegraph.tools.max_relations --e-max 344 --n-max 16
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Dict, List

from scenegraph.configs.loader import load_config
from scenegraph.core.affordance import (
    lookup_bottom_components,
    lookup_components,
    lookup_contact_components,
    lookup_contain_components,
    lookup_key_components,
    lookup_support_components,
)
from scenegraph.core.relation_rules import (
    _both,
    _get_bin_spec,
    interaction_types,
)
from scenegraph.core.schema import Node
from scenegraph.core.whitelist import Whitelist, load_whitelist
from scenegraph.tools.build_union_whitelist import UNION_TARGET

SPECS = (
    "planar-distance", "height-offset", "grasp-compatibility",
    "contact-compatibility", "support-compatibility", "contain-compatibility",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="room_scale")
    p.add_argument("--thresholds", default=None,
                   help="thresholds.yaml; omit for the packaged one")
    p.add_argument("--whitelist-dir", default="",
                   help="empty → the profile's configured directory")
    p.add_argument("--e-max", type=int, default=96,
                   help="dreamerv3/configs.yaml env.maniskill.graph.e_max")
    p.add_argument("--n-max", type=int, default=0,
                   help="0 → the profile's selection.n_max")
    return p.parse_args()


def _member_nodes(wl: Whitelist) -> List[Node]:
    """One node per member, carrying exactly what the gates read off a node."""
    return [
        Node(
            node_id=key, node_type="object", name=key, visible=True,
            segmentation_ids=[1], pose_world=[0.0] * 7,
            attributes={
                "entity_key": key,
                "interaction_types": sorted(wl.types(key)),
            },
        )
        for key in sorted(wl.by_key)
    ]


def _gates(wl: Whitelist, base_cfg: dict):
    """Every spec and config flag the emission rules consult, resolved once."""
    cfg = dict(base_cfg)
    cfg["bin_edges"] = wl.bin_edges
    aff = cfg.get("affordance_set")
    aff_cfg = cfg.get("affordances", {})
    spec = {rel: _get_bin_spec(cfg, rel) is not None for rel in SPECS}
    # Both affordance builders return early without a distance spec.
    near = spec["planar-distance"]
    ee_compat = near and (
        spec["grasp-compatibility"] or spec["contact-compatibility"])
    obj = near and aff is not None
    flags = {
        "contact": obj and spec["contact-compatibility"] and bool(
            aff_cfg.get("object_object_contact_compatibility", True)),
        "support": obj and spec["support-compatibility"] and bool(
            aff_cfg.get("object_object_support_compatibility", False)),
        "contain": obj and spec["contain-compatibility"] and bool(
            aff_cfg.get("object_object_contain_compatibility", True)),
    }
    return aff, spec, ee_compat, flags


def vertex_facts(node: Node, aff, spec, ee_compat) -> int:
    """ee--object facts one vertex contributes."""
    types = interaction_types(node)
    n = 0
    n += spec["planar-distance"]
    n += spec["height-offset"]
    n += "contact" in types
    n += "grasp" in types
    # Both ee compatibility facts are measured against the active grasp anchor,
    # so both need grasp components even though only one is named for grasp.
    if ee_compat and lookup_components(aff, node):
        n += spec["grasp-compatibility"] and "grasp" in types
        n += spec["contact-compatibility"] and "contact" in types
    return int(n)


def pair_facts(a: Node, b: Node, aff, flags) -> int:
    """object--object facts one unordered vertex pair contributes."""
    ta, tb = interaction_types(a), interaction_types(b)
    n = 0
    if _both(ta, tb, "contact"):
        n += 1
    if _both(ta, tb, "support"):
        # Directed, both orderings, and no component gate on the physical
        # fact: the term that grows fastest with vertex count.
        n += 2
    if _both(ta, tb, "contain") and aff is not None:
        for container, containee in ((a, b), (b, a)):
            if (lookup_contain_components(aff, container)
                    and lookup_key_components(aff, containee)):
                n += 1
    if flags["contact"] and _both(ta, tb, "contact"):
        if (lookup_contact_components(aff, a)
                and lookup_contact_components(aff, b)):
            n += 1
    if flags["support"] and _both(ta, tb, "support"):
        for supporter, supported in ((a, b), (b, a)):
            if (lookup_support_components(aff, supporter)
                    and lookup_bottom_components(aff, supported)):
                n += 1
    if flags["contain"] and _both(ta, tb, "contain"):
        for container, containee in ((a, b), (b, a)):
            if (lookup_contain_components(aff, container)
                    and lookup_key_components(aff, containee)):
                n += 1
    return n


def ceiling(wl: Whitelist, base_cfg: dict, n_max: int) -> Dict[str, object]:
    """Fact ceiling for ``n_max`` vertices drawn from ``wl``'s members."""
    aff, spec, ee_compat, flags = _gates(wl, base_cfg)
    nodes = _member_nodes(wl)
    slots = max(int(n_max) - 1, 0)

    best_vertex, vertex_key = 0, ""
    for node in nodes:
        value = vertex_facts(node, aff, spec, ee_compat)
        if value > best_vertex:
            best_vertex, vertex_key = value, node.name

    # Self-pairs included: two instances of one asset are a pair of copies of
    # that member, and nothing in the whitelist forbids them.
    best_pair, pair_keys = 0, ("", "")
    for a, b in itertools.combinations_with_replacement(nodes, 2):
        value = pair_facts(a, b, aff, flags)
        if value > best_pair:
            best_pair, pair_keys = value, (a.name, b.name)

    pairs = slots * (slots - 1) // 2
    distinct = min(len(nodes), slots)
    return {
        "slots": slots,
        "pairs": pairs,
        "per_vertex": best_vertex,
        "vertex_key": vertex_key,
        "per_pair": best_pair,
        "pair_keys": pair_keys,
        "total": slots * best_vertex + pairs * best_pair,
        # Same bound at one vertex per member, so the gap against the ceiling
        # is what the duplicate-instance headroom in n_max costs. Clipped to
        # the slot count, though a per-target whitelist never approaches it.
        "distinct": distinct,
        "distinct_total": (
            distinct * best_vertex + distinct * (distinct - 1) // 2 * best_pair),
    }


def report(path: str, wl: Whitelist, c: Dict[str, object]) -> None:
    print(f"\n{os.path.basename(path)}  "
          f"subtask={wl.subtask or '?'} target={wl.target or '?'}")
    print(f"  {len(wl.by_key)} members, {c['slots']} object slots, "
          f"{c['pairs']} pairs")
    print(f"    per vertex  {c['per_vertex']:3d}  ({c['vertex_key']})")
    print(f"    per pair    {c['per_pair']:3d}  "
          f"({c['pair_keys'][0]} + {c['pair_keys'][1]})")
    print(f"    every member once  {c['distinct_total']:5d}  "
          f"({c['distinct']} vertices)")
    print(f"    every slot filled  {c['total']:5d}  "
          f"({c['slots']} copies of the densest member)")


def main() -> int:
    args = parse_args()
    cfg = load_config(args.profile, args.thresholds)
    wl_dir = args.whitelist_dir or cfg["whitelist_dir"]
    if not os.path.isdir(wl_dir):
        print(f"FAIL: no whitelist directory at {wl_dir}")
        return 1
    paths = sorted(
        os.path.join(wl_dir, name) for name in os.listdir(wl_dir)
        if name.endswith(".json"))
    if not paths:
        print(f"FAIL: no whitelists in {wl_dir}; mine them with "
              "tools/prepare_assets.py")
        return 1
    n_max = args.n_max or int(cfg["selection"]["n_max"])

    print(f"n_max {n_max}: {n_max - 1} object slots, "
          f"{(n_max - 1) * (n_max - 2) // 2} pairs")
    worst, worst_path, floor, floor_path = 0, "", 0, ""
    for path in paths:
        wl = load_whitelist(path)
        # A union file is the bin-edge asset. No episode binds it for
        # membership, so it has no vertex set and a fact count for it would
        # describe a graph the runtime never builds.
        if wl.target == UNION_TARGET:
            print(f"\n{os.path.basename(path)}  bin-edge asset, "
                  f"{len(wl.by_key)} members, never bound for membership")
            continue
        c = ceiling(wl, cfg, n_max)
        report(path, wl, c)
        if int(c["total"]) > worst:
            worst, worst_path = int(c["total"]), path
        if int(c["distinct_total"]) > floor:
            floor, floor_path = int(c["distinct_total"]), path

    print(f"\nevery member once  {floor:5d}  ({os.path.basename(floor_path)})")
    print(f"every slot filled  {worst:5d}  ({os.path.basename(worst_path)})")
    if floor > args.e_max:
        print(f"FAIL: e_max {args.e_max} cannot hold one instance of every "
              "member. Those vertices appear whenever the scene shows them, so "
              f"this truncates in ordinary operation. Raise e_max to {floor}.")
        return 1
    if worst > args.e_max:
        print(f"NOTE: below the {worst}-fact ceiling, which needs "
              f"{n_max - 1} copies of one asset in a single scene. Overflow "
              "drops spatial first and counts into log/graph_fact_drops -- "
              "size from the smoke run's measured peak and watch that counter.")
    print(f"OK: e_max {args.e_max} covers every member once, "
          f"{args.e_max - floor} slots spare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
