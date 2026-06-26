"""Affordance components mined from manipulation-success rollouts.

Per stable object key the asset stores a list of components, each holding
``anchor_obj_frame`` (point), optional ``approach_dir_obj_frame`` (unit), and
``preferred_width`` (qpos-sum). Frames are OBJECT-local, metres.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .schema import Node


@dataclass
class AffordanceComponent:
    anchor_obj_frame: np.ndarray            # (3,) OBJECT frame, metres
    preferred_width: float                  # qpos[-2] + qpos[-1]
    approach_dir_obj_frame: Optional[np.ndarray] = None  # unit, OBJECT frame


@dataclass
class AffordanceSet:
    by_object: Dict[str, List[AffordanceComponent]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.by_object


def _normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _quat_wxyz_to_rotmat(q: np.ndarray) -> Optional[np.ndarray]:
    """SAPIEN (qw, qx, qy, qz) -> 3x3 rotation matrix."""
    qn = _normalize(np.asarray(q, dtype=float))
    if qn is None:
        return None
    w, x, y, z = qn
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# MS-HAB actor names look like ``env-0_024_bowl-3``; canonical form is ``024_bowl``.
_ENV_PREFIX_RE = re.compile(r"^env-\d+_")
_INSTANCE_SUFFIX_RE = re.compile(r"-\d+$")


def canonical_affordance_key(name: Optional[str]) -> Optional[str]:
    """Strip ``env-N_`` prefix and ``-N`` instance suffix (``env-0_024_bowl-3`` -> ``024_bowl``)."""
    if not name:
        return None
    key = _ENV_PREFIX_RE.sub("", str(name))
    key = _INSTANCE_SUFFIX_RE.sub("", key)
    return key or None


def load_affordance_set(path: Optional[str]) -> AffordanceSet:
    """Load JSON asset; missing/unreadable file -> empty set (warn).

    Shape: ``{"objects": {key: {"components": [{"anchor": [x,y,z],
    "approach_dir": [..]?, "width": w}, ...]}}}``. Keys starting with ``_`` are
    metadata. ``approach_dir`` is optional (no orientation-alignment edge without it).
    """
    if not path or not os.path.isfile(path):
        warnings.warn(
            f"affordance asset not found at {path!r}; "
            "affordance relations will be silently skipped.",
            RuntimeWarning,
        )
        return AffordanceSet()

    with open(path, "r") as f:
        raw = json.load(f)

    by_object: Dict[str, List[AffordanceComponent]] = {}
    objects = raw.get("objects", {}) if isinstance(raw, dict) else {}
    for key, entry in objects.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        comps_raw = (entry or {}).get("components", []) if isinstance(entry, dict) else []
        comps: List[AffordanceComponent] = []
        for c in comps_raw:
            if not isinstance(c, dict):
                continue
            anchor = c.get("anchor")
            width = c.get("width")
            if anchor is None or width is None:
                continue
            arr = np.asarray(anchor, dtype=float).reshape(-1)
            if arr.size != 3 or not np.all(np.isfinite(arr)):
                continue
            try:
                w = float(width)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(w):
                continue

            approach = c.get("approach_dir")
            approach_arr: Optional[np.ndarray] = None
            if approach is not None:
                a = np.asarray(approach, dtype=float).reshape(-1)
                if a.size == 3 and np.all(np.isfinite(a)):
                    a = _normalize(a)
                    if a is not None:
                        approach_arr = a

            comps.append(
                AffordanceComponent(
                    anchor_obj_frame=arr,
                    preferred_width=w,
                    approach_dir_obj_frame=approach_arr,
                )
            )
        if comps:
            by_object[str(key)] = comps
    return AffordanceSet(by_object=by_object)


def _obj_rotmat(obj_pose_world: Optional[List[float]]) -> Optional[np.ndarray]:
    if obj_pose_world is None or len(obj_pose_world) < 7:
        return None
    q = np.asarray(obj_pose_world[3:7], dtype=float)
    return _quat_wxyz_to_rotmat(q)


def transform_anchors(
    obj_pose_world: Optional[List[float]],
    components: List[AffordanceComponent],
) -> Optional[np.ndarray]:
    """(N,3) world anchors. ``obj_pose_world`` is ``[xyz, qw, qx, qy, qz]``."""
    if obj_pose_world is None or len(obj_pose_world) < 7 or not components:
        return None
    p = np.asarray(obj_pose_world[:3], dtype=float)
    if not np.all(np.isfinite(p)):
        return None
    R = _obj_rotmat(obj_pose_world)
    if R is None:
        return None
    anchors_obj = np.stack(
        [np.asarray(c.anchor_obj_frame, dtype=float).reshape(3) for c in components],
        axis=0,
    )
    return p[None, :] + anchors_obj @ R.T


def transform_approach_dir(
    obj_pose_world: Optional[List[float]],
    component: AffordanceComponent,
) -> Optional[np.ndarray]:
    """World-frame unit approach direction, or None if not mined / degenerate."""
    if component.approach_dir_obj_frame is None:
        return None
    R = _obj_rotmat(obj_pose_world)
    if R is None:
        return None
    d_world = R @ np.asarray(component.approach_dir_obj_frame, dtype=float).reshape(3)
    return _normalize(d_world)


def select_active_component(
    tcp_xyz_world: np.ndarray,
    anchors_world: Optional[np.ndarray],
    *,
    components: Optional[List[AffordanceComponent]] = None,
    obj_pose_world: Optional[List[float]] = None,
    tcp_pose_world: Optional[List[float]] = None,
    tcp_axis_local: Optional[List[float]] = None,
    orientation_weight: float = 0.10,
) -> Optional[int]:
    """Index of the candidate closest to the TCP. Adds ``orientation_weight * angle/pi``
    (metres) when TCP/component orientations are available."""
    if anchors_world is None or len(anchors_world) == 0:
        return None
    tcp = np.asarray(tcp_xyz_world, dtype=float).reshape(-1)
    if tcp.shape[0] != 3 or not np.all(np.isfinite(tcp)):
        return None
    diffs = anchors_world - tcp[None, :]
    scores = np.linalg.norm(diffs, axis=1)

    if (
        components is not None
        and obj_pose_world is not None
        and tcp_pose_world is not None
        and tcp_axis_local is not None
        and orientation_weight > 0.0
    ):
        tcp_R = _quat_wxyz_to_rotmat(np.asarray(tcp_pose_world[3:7], dtype=float))
        if tcp_R is not None:
            tcp_dir = _normalize(
                tcp_R @ np.asarray(tcp_axis_local, dtype=float).reshape(3)
            )
            if tcp_dir is not None:
                for idx, comp in enumerate(components[: len(scores)]):
                    aff_dir = transform_approach_dir(obj_pose_world, comp)
                    if aff_dir is None:
                        continue
                    cos = float(np.clip(np.dot(tcp_dir, aff_dir), -1.0, 1.0))
                    angle = float(np.arccos(cos))
                    scores[idx] += orientation_weight * (angle / np.pi)

    return int(np.argmin(scores))


def lookup_components(
    aff_set: AffordanceSet, node: Node,
) -> Optional[List[AffordanceComponent]]:
    """Component list for a node, or None.

    Lookup: entity_key -> actor_key (legacy, strip ``actor:``) -> mshab_obj_id -> name.
    """
    if aff_set is None or aff_set.is_empty():
        return None

    entity_key = node.attributes.get("entity_key") if node.attributes else None
    if entity_key:
        entity_key = str(entity_key)
        if entity_key in aff_set.by_object:
            return aff_set.by_object[entity_key]
        if entity_key.startswith("actor:"):
            legacy_actor = entity_key.split(":", 1)[1]
            if legacy_actor in aff_set.by_object:
                return aff_set.by_object[legacy_actor]

    mshab_id = node.attributes.get("mshab_obj_id") if node.attributes else None
    key = canonical_affordance_key(mshab_id) if mshab_id else None
    if key and key in aff_set.by_object:
        return aff_set.by_object[key]

    key = canonical_affordance_key(node.name)
    if key and key in aff_set.by_object:
        return aff_set.by_object[key]
    return None


def has_affordance(aff_set: AffordanceSet, node: Node) -> bool:
    """True if the node has any mined components (eligibility for ``interactive_object``)."""
    return bool(lookup_components(aff_set, node))
