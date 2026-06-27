"""Per-subtask whitelist used as the runtime's sole relevance gate.

The whitelist is a *small* per-(subtask, target) JSON asset, mined offline from
demonstration contact graphs by ``tools/build_subtask_whitelists.py``. It is
the hard eligibility gate used by the runtime selector: every non-ee node in a
frame must have a ``match_key`` listed in the active subtask's whitelist or it
is dropped before slot assignment.

The asset also carries:

* ``interaction_types`` per member -- the set of ee-driven interactions that
  actually happened in the demos (``contact`` and/or ``grasp``). This gates
  which affordance compatibility edges runtime emits for that object.
* ``bin_edges`` -- per-(subtask, target) bin edges for spatial, affordance,
  and change relations derived from per-rollout running maxes (equal-width on
  ``[0, max]`` for unsigned, ``[-max, max]`` for signed).
* ``compat_norm`` -- per-component [0,1] normalization scales for position,
  orientation, and gripper-width mismatches used by compatibility scoring.

Asset shape (``_schema_version: 3``)::

    {
      "_schema_version": 3,
      "subtask": "pick",
      "target": "actor:024_bowl",
      "members": {
        "actor:024_bowl": {
            "roles": ["interacted"],
            "interaction_types": ["contact", "grasp"],
            "kind": "actor"
        },
        "link:cabinet/drawer3": {
            "roles": ["support"],
            "interaction_types": [],
            "kind": "link"
        }
      },
      "bin_edges": { "<relation>": [edges...], ... },
      "compat_norm": {"pos": float, "orient": float, "width": float}
    }

Match-key conventions:

  * Free actors: ``actor:<canonical object id>``.
    Strips ``env-N_`` prefix and ``-N`` instance suffix so every instance of the
    same YCB type shares one key.
  * Articulation links: ``link:<articulation instance>/<link name>``.
  * Handles are ordinary links and have no special admission path.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .affordance import canonical_affordance_key
from .entity_identity import normalize_asset_key
from .schema import Node


# --------------------------------------------------------------------------- #
# Match key (used by both runtime and miner)
# --------------------------------------------------------------------------- #
def match_key(node: Node) -> Optional[str]:
    """Resolve the cross-frame whitelist key for one node.

    Returns None when no key can be produced (e.g. blank name on a malformed
    node); the selector treats None as "not in any whitelist" => dropped.
    """
    name = node.name
    if not name:
        return None
    attrs = node.attributes or {}
    stable = attrs.get("entity_key")
    if stable:
        return normalize_asset_key(str(stable), attrs.get("entity_kind"))
    if attrs.get("is_actor"):
        return normalize_asset_key(canonical_affordance_key(name) or name, "actor")
    if attrs.get("is_link") or attrs.get("is_articulation_link"):
        return normalize_asset_key(name, "link")
    return normalize_asset_key(name)


# --------------------------------------------------------------------------- #
# Asset
# --------------------------------------------------------------------------- #
# Canonical interaction-type tokens that runtime understands.
INTERACTION_CONTACT = "contact"
INTERACTION_GRASP = "grasp"
_VALID_INTERACTION_TYPES = frozenset({INTERACTION_CONTACT, INTERACTION_GRASP})


@dataclass
class Whitelist:
    subtask: str = ""
    target: str = ""
    by_key: Dict[str, Set[str]] = field(default_factory=dict)
    interaction_types: Dict[str, Set[str]] = field(default_factory=dict)
    bin_edges: Dict[str, List[float]] = field(default_factory=dict)
    compat_norm: Dict[str, float] = field(default_factory=dict)
    source_path: Optional[str] = None

    @property
    def empty(self) -> bool:
        return not self.by_key

    def contains(self, key: Optional[str]) -> bool:
        if key is None:
            return False
        return key in self.by_key

    def roles(self, key: Optional[str]) -> Set[str]:
        if key is None:
            return set()
        return set(self.by_key.get(key, set()))

    def types(self, key: Optional[str]) -> Set[str]:
        if key is None:
            return set()
        return set(self.interaction_types.get(key, set()))


def load_whitelist(path: str) -> Whitelist:
    """Load a per-subtask whitelist. Raises FileNotFoundError if missing.

    A missing whitelist must not silently fall back to "admit everything".
    Older (v2) assets without ``interaction_types`` / ``bin_edges`` still load;
    the missing fields fall back to runtime defaults at edge-build time.
    """
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"per-subtask whitelist not found at {path!r}; mine it with "
            "tools/build_subtask_whitelists.py before running the probe"
        )
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"whitelist {path!r}: expected JSON object at root")

    by_key: Dict[str, Set[str]] = {}
    interaction_types: Dict[str, Set[str]] = {}
    members = raw.get("members", {})
    if not isinstance(members, dict):
        raise ValueError(f"whitelist {path!r}: 'members' must be an object")
    for k, entry in members.items():
        if not isinstance(k, str) or k.startswith("_"):
            continue
        roles_set: Set[str] = set()
        itypes_set: Set[str] = set()
        kind = None
        if isinstance(entry, dict):
            roles = entry.get("roles")
            if isinstance(roles, (list, tuple)):
                for r in roles:
                    if isinstance(r, str):
                        roles_set.add(r)
            itypes = entry.get("interaction_types")
            if isinstance(itypes, (list, tuple)):
                for t in itypes:
                    if isinstance(t, str) and t in _VALID_INTERACTION_TYPES:
                        itypes_set.add(t)
            kind = entry.get("kind")
        normalized = normalize_asset_key(k, kind)
        if normalized:
            by_key[normalized] = roles_set
            interaction_types[normalized] = itypes_set

    if not by_key:
        raise ValueError(f"whitelist {path!r}: 'members' is empty")

    bin_edges: Dict[str, List[float]] = {}
    raw_edges = raw.get("bin_edges", {})
    if isinstance(raw_edges, dict):
        for rel, edges in raw_edges.items():
            if not isinstance(rel, str) or not isinstance(edges, (list, tuple)):
                continue
            parsed: List[float] = []
            ok = True
            for x in edges:
                try:
                    v = float(x)
                except (TypeError, ValueError):
                    ok = False
                    break
                if not math.isfinite(v):
                    ok = False
                    break
                parsed.append(v)
            if ok and parsed:
                bin_edges[rel] = parsed

    compat_norm: Dict[str, float] = {}
    raw_norm = raw.get("compat_norm", {})
    if isinstance(raw_norm, dict):
        for k, v in raw_norm.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv) and fv > 0:
                compat_norm[str(k)] = fv

    return Whitelist(
        subtask=str(raw.get("subtask", "") or ""),
        target=str(raw.get("target", "") or ""),
        by_key=by_key,
        interaction_types=interaction_types,
        bin_edges=bin_edges,
        compat_norm=compat_norm,
        source_path=path,
    )


# --------------------------------------------------------------------------- #
# Bin-edge derivation
# --------------------------------------------------------------------------- #
# Equal-width splits of [0, max] (unsigned) and [-max, max] (signed).
# 3-label unsigned: 2 edges at max/3 and 2*max/3.
# 3-label signed:   2 edges at -max/3 and  max/3.
# 5-label signed:   4 edges at -3*max/5, -max/5, max/5, 3*max/5.
# Compatibility absolute edges are fixed because the score is already in [0,1].


def _equal_width_3_unsigned(max_v: float) -> Optional[List[float]]:
    if max_v <= 0 or not math.isfinite(max_v):
        return None
    return [max_v / 3.0, 2.0 * max_v / 3.0]


def _equal_width_3_signed(max_v: float) -> Optional[List[float]]:
    if max_v <= 0 or not math.isfinite(max_v):
        return None
    return [-max_v / 3.0, max_v / 3.0]


def _equal_width_5_signed(max_v: float) -> Optional[List[float]]:
    if max_v <= 0 or not math.isfinite(max_v):
        return None
    return [-3.0 * max_v / 5.0, -max_v / 5.0, max_v / 5.0, 3.0 * max_v / 5.0]


# Relations and the kind of bin-derivation they use.
_BIN_DERIVATION = {
    "planar-distance": ("unsigned3", "planar_distance"),
    "height-offset": ("signed3", "height_offset"),
    "planar-distance-change": ("signed5", "planar_distance_change"),
    "height-offset-change": ("signed5", "height_offset_change"),
    "grasp-compatibility-change": ("signed5", "grasp_compatibility_change"),
    "contact-compatibility-change": ("signed5", "contact_compatibility_change"),
}


def derive_bin_edges(max_values: Dict[str, float]) -> Dict[str, List[float]]:
    """Derive runtime bin edges from per-relation demo maxes.

    ``max_values`` keys are the *snake_case* names emitted by the collector:
    ``planar_distance``, ``height_offset`` (abs), ``planar_distance_change``,
    ``height_offset_change``, ``grasp_compatibility_change``,
    ``contact_compatibility_change``. The returned dict is keyed by the
    *dashed* relation name used by the schema.

    Compatibility absolute edges are always ``[1/3, 2/3]`` because the score
    is normalized to ``[0, 1]`` before binning.
    """
    out: Dict[str, List[float]] = {
        "grasp-compatibility": [1.0 / 3.0, 2.0 / 3.0],
        "contact-compatibility": [1.0 / 3.0, 2.0 / 3.0],
    }
    for relation, (kind, src) in _BIN_DERIVATION.items():
        try:
            v = float(max_values.get(src, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        edges: Optional[List[float]] = None
        if kind == "unsigned3":
            edges = _equal_width_3_unsigned(v)
        elif kind == "signed3":
            edges = _equal_width_3_signed(v)
        elif kind == "signed5":
            edges = _equal_width_5_signed(v)
        if edges is not None:
            out[relation] = edges
    return out


# --------------------------------------------------------------------------- #
# Filename resolver
# --------------------------------------------------------------------------- #
def resolve_whitelist_path(
    whitelist_dir: Optional[str],
    subtask_type: Optional[str],
    target_canonical: Optional[str],
) -> Optional[str]:
    """Return the on-disk path for ``<subtask>_<target>.json`` if it exists.

    Returns None if any of the inputs is missing or the file is absent. The
    caller decides whether to raise.
    """
    if not whitelist_dir or not subtask_type or not target_canonical:
        return None
    target_slug = whitelist_target_slug(target_canonical)
    fname = f"{subtask_type}_{target_slug}.json"
    path = os.path.join(whitelist_dir, fname)
    return path if os.path.isfile(path) else None


def whitelist_target_slug(target_key: str) -> str:
    """Filesystem-safe target portion shared by runtime and offline miner."""
    value = str(target_key).split(":", 1)[-1]
    return value.replace("/", "__").replace("\\", "__").replace(":", "_")
