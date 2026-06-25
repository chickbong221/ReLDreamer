"""Per-subtask whitelist used as the runtime's sole relevance gate.

The whitelist is a *small* per-(subtask, target) JSON asset, mined offline from
demonstration contact graphs by ``tools/build_subtask_whitelists.py``. It is
the hard eligibility gate used by the runtime selector: every non-ee node in a
frame must have a ``match_key`` listed in the active subtask's whitelist or it
is dropped before slot assignment.

Asset shape (``_schema_version: 2``)::

    {
      "_schema_version": 2,
      "subtask": "pick",
      "target": "actor:024_bowl",
      "members": {
        "actor:024_bowl":       {"roles": ["interacted"], "kind": "actor"},
        "link:cabinet/drawer3": {"roles": ["support"],    "kind": "link"}
      }
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
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

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
@dataclass
class Whitelist:
    subtask: str = ""
    target: str = ""
    by_key: Dict[str, Set[str]] = field(default_factory=dict)
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


def load_whitelist(path: str) -> Whitelist:
    """Load a per-subtask whitelist. Raises FileNotFoundError if missing.

    A missing whitelist must not silently fall back to "admit everything".
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
    members = raw.get("members", {})
    if not isinstance(members, dict):
        raise ValueError(f"whitelist {path!r}: 'members' must be an object")
    for k, entry in members.items():
        if not isinstance(k, str) or k.startswith("_"):
            continue
        roles_set: Set[str] = set()
        if isinstance(entry, dict):
            roles = entry.get("roles")
            if isinstance(roles, (list, tuple)):
                for r in roles:
                    if isinstance(r, str):
                        roles_set.add(r)
        kind = entry.get("kind") if isinstance(entry, dict) else None
        normalized = normalize_asset_key(k, kind)
        if normalized:
            by_key[normalized] = roles_set

    if not by_key:
        raise ValueError(f"whitelist {path!r}: 'members' is empty")

    return Whitelist(
        subtask=str(raw.get("subtask", "") or ""),
        target=str(raw.get("target", "") or ""),
        by_key=by_key,
        source_path=path,
    )


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
