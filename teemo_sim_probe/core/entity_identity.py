"""Stable identities shared by offline collection and runtime graph building.

The graph has only two node types (``ee`` and ``object``).  Free actors and
articulation links are both ordinary object nodes; their stable keys differ so
that a support link and a handle link never collapse into one entity.
"""

from __future__ import annotations

import re
from typing import Optional

from .affordance import canonical_affordance_key


_ENV_PREFIX_RE = re.compile(r"^env-\d+_")


def entity_name(entity) -> str:
    return str(getattr(entity, "name", None) or entity)


def entity_kind(entity) -> str:
    name = type(entity).__name__
    if name == "Actor":
        return "actor"
    if name == "Link":
        return "link"
    return "other"


def _articulation(entity):
    for attr in ("articulation", "parent_articulation"):
        value = getattr(entity, attr, None)
        if value is not None:
            return value
    for method in ("get_articulation", "get_parent_articulation"):
        fn = getattr(entity, method, None)
        if callable(fn):
            try:
                value = fn()
            except Exception:
                continue
            if value is not None:
                return value
    return None


def canonical_scene_name(name: Optional[str]) -> Optional[str]:
    """Remove only the per-environment prefix, preserving instance suffixes."""
    if not name:
        return None
    return _ENV_PREFIX_RE.sub("", str(name)) or None


def stable_entity_key(entity) -> Optional[str]:
    """Return ``actor:<id>`` or ``link:<articulation>/<link>``.

    Link qualification is best-effort because SAPIEN versions expose the
    parent articulation through different attributes.  The bare link name is
    retained as a deterministic fallback.
    """
    if entity is None:
        return None
    name = entity_name(entity)
    kind = entity_kind(entity)
    if kind == "actor":
        canonical = canonical_affordance_key(name) or name
        return f"actor:{canonical}"
    if kind == "link":
        link_name = canonical_scene_name(name) or name
        art = _articulation(entity)
        art_name = canonical_scene_name(entity_name(art)) if art is not None else None
        qualified = f"{art_name}/{link_name}" if art_name else link_name
        return f"link:{qualified}"
    return f"object:{canonical_scene_name(name) or name}"


def stable_node_id(entity) -> str:
    if entity_kind(entity) == "actor":
        # Node identity preserves the simulator instance suffix while the
        # whitelist key intentionally canonicalizes actors by asset type.
        name = canonical_scene_name(entity_name(entity)) or entity_name(entity)
        return f"actor:{name}"
    return stable_entity_key(entity) or f"object:{entity_name(entity)}"


def normalize_asset_key(key: Optional[str], kind: Optional[str] = None) -> Optional[str]:
    """Normalize new and legacy whitelist keys to the stable-key namespace."""
    if not key:
        return None
    value = str(key)
    if value.startswith(("actor:", "link:", "object:")):
        return value
    if kind == "actor":
        return f"actor:{canonical_affordance_key(value) or value}"
    if kind == "link":
        return f"link:{value}"
    return value
