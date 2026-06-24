"""R4 -- domain-level allowed physical entity vocabulary.

Loaded once at startup. JSON shape::

    {
      "_schema_version": 1,
      "domain": "mshab",
      "split": "train",
      "n_trajectories": 200,
      "entities": {
        "024_bowl":           {"roles": ["task"]},
        "kitchen_counter_15": {"roles": ["support"]},
        "fridge_body":        {"roles": ["state", "support"]}
      }
    }

Empty / missing asset -> ``contains`` always True so non-MS-HAB envs and the
first runs (before mining) still work.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

from .affordance import canonical_affordance_key
from .schema import Node


@dataclass
class EDomain:
    by_key: Dict[str, Set[str]] = field(default_factory=dict)
    empty: bool = True

    # ---- key resolution -------------------------------------------------- #
    def _resolve_key(self, node: Node) -> Optional[str]:
        if node.attributes:
            mshab_id = node.attributes.get("mshab_obj_id")
            if mshab_id:
                k = canonical_affordance_key(mshab_id)
                if k and k in self.by_key:
                    return k
        k = canonical_affordance_key(node.name)
        if k and k in self.by_key:
            return k
        # Also try unstripped name (handle links like "fridge:fridge_body_0").
        if node.name in self.by_key:
            return node.name
        return None

    # ---- queries --------------------------------------------------------- #
    def contains(self, node: Node) -> bool:
        if self.empty:
            return True
        return self._resolve_key(node) is not None

    def roles(self, node: Node) -> Set[str]:
        if self.empty:
            return set()
        k = self._resolve_key(node)
        return set(self.by_key.get(k, set())) if k else set()

    def is_state(self, node: Node) -> bool:
        return "state" in self.roles(node)

    def is_support(self, node: Node) -> bool:
        return "support" in self.roles(node)

    def is_task(self, node: Node) -> bool:
        return "task" in self.roles(node)


def load_e_domain(path: Optional[str]) -> EDomain:
    if not path or not os.path.isfile(path):
        warnings.warn(
            f"E_domain asset not found at {path!r}; "
            "every visible entity will be eligible.",
            RuntimeWarning,
        )
        return EDomain()
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception as exc:
        warnings.warn(f"failed to load E_domain {path!r}: {exc!r}", RuntimeWarning)
        return EDomain()

    by_key: Dict[str, Set[str]] = {}
    entities = (raw or {}).get("entities", {}) if isinstance(raw, dict) else {}
    for key, entry in entities.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        roles = (entry or {}).get("roles") if isinstance(entry, dict) else None
        roles_set: Set[str] = set()
        if isinstance(roles, (list, tuple)):
            for r in roles:
                if isinstance(r, str):
                    roles_set.add(r)
        by_key[key] = roles_set or {"task"}
    return EDomain(by_key=by_key, empty=not by_key)
