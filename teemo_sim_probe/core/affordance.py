"""Affordance components mined from manipulation-success rollouts.

This module owns the static, pose-invariant grip information that the runtime
needs to emit affordance-grounded relations. It is intentionally framework-
agnostic (numpy only, no SAPIEN / ManiSkill imports) so the asset can be reused
across benchmarks -- MS-HAB today, vanilla ManiSkill / BEHAVIOR-1K next -- as
long as the offline miner produces the JSON shape below.

Per stable actor or articulation-link key, the asset stores components. Each is
``(anchor_obj_frame, approach_dir_obj_frame, preferred_width)``:

  * ``anchor_obj_frame`` -- 3D point in the OBJECT frame, in metres.
    "where on the object the gripper should grip". Pose-invariant: the same
    value is reused every frame regardless of where the object currently sits.
  * ``approach_dir_obj_frame`` -- 3D UNIT vector in the OBJECT frame giving the
    gripper approach axis at success. Optional; required only for
    ``orientation-alignment``. Assets mined before orientation support
    (schema_version 1) load fine -- that component just yields no
    orientation-alignment edge.
  * ``preferred_width`` -- scalar gripper width (qpos-sum convention, identical
    to what ``adapters/privileged_state.compute_gripper_width`` produces at
    runtime) at the moment of success.

To extend to a new benchmark / robot, add a miner that produces the same JSON
shape with canonical object keys; nothing in this file is MS-HAB-specific.
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


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class AffordanceComponent:
    anchor_obj_frame: np.ndarray            # shape (3,), OBJECT frame, metres
    preferred_width: float                  # qpos[-2] + qpos[-1] convention
    # Unit approach direction in OBJECT frame, or None if not mined.
    approach_dir_obj_frame: Optional[np.ndarray] = None


@dataclass
class AffordanceSet:
    by_object: Dict[str, List[AffordanceComponent]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.by_object


# --------------------------------------------------------------------------- #
# Geometry helpers (self-contained -- relation_rules imports this module)
# --------------------------------------------------------------------------- #
def _normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _quat_wxyz_to_rotmat(q: np.ndarray) -> Optional[np.ndarray]:
    """SAPIEN quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix, or None."""
    qn = _normalize(np.asarray(q, dtype=float))
    if qn is None:
        return None
    w, x, y, z = qn
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# --------------------------------------------------------------------------- #
# Key canonicalization
# --------------------------------------------------------------------------- #
# MS-HAB exposes per-env-prefixed and per-instance-suffixed actor names
# (env-0_024_bowl-3). Actor fallbacks canonicalize them to the benchmark key.
# Articulation links use their exact qualified stable key.
_ENV_PREFIX_RE = re.compile(r"^env-\d+_")
_INSTANCE_SUFFIX_RE = re.compile(r"-\d+$")


def canonical_affordance_key(name: Optional[str]) -> Optional[str]:
    """Strip env-N_ prefix and -N instance suffix.

    Examples::

        env-0_024_bowl-3  -> 024_bowl
        024_bowl-3        -> 024_bowl
        024_bowl          -> 024_bowl
        None / ""         -> None
    """
    if not name:
        return None
    key = _ENV_PREFIX_RE.sub("", str(name))
    key = _INSTANCE_SUFFIX_RE.sub("", key)
    return key or None


# --------------------------------------------------------------------------- #
# Asset I/O
# --------------------------------------------------------------------------- #
def load_affordance_set(path: Optional[str]) -> AffordanceSet:
    """Load JSON asset. Missing / unreadable file => empty set (warn, no raise).

    JSON shape (schema_version 2)::

        { "_README": "...",
          "_schema_version": 2,
          "objects": {
            "actor:024_bowl": { "components": [
                {"anchor": [x, y, z],
                 "approach_dir": [ax, ay, az],   # optional, OBJECT frame unit
                 "width": 0.045},
                ...
            ]}
          }
        }

    ``approach_dir`` is optional. Schema-v1 assets (no approach_dir) still load;
    those components simply have ``approach_dir_obj_frame is None`` and yield
    no orientation-alignment edge. Keys starting with ``_`` are reserved for
    metadata and skipped.
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

            # Optional approach direction (object frame, unit). Reject
            # degenerate / non-finite vectors -> None (skip orientation only).
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


# --------------------------------------------------------------------------- #
# Runtime helpers
# --------------------------------------------------------------------------- #
def _obj_rotmat(obj_pose_world: Optional[List[float]]) -> Optional[np.ndarray]:
    if obj_pose_world is None or len(obj_pose_world) < 7:
        return None
    q = np.asarray(obj_pose_world[3:7], dtype=float)
    return _quat_wxyz_to_rotmat(q)


def transform_anchors(
    obj_pose_world: Optional[List[float]],
    components: List[AffordanceComponent],
) -> Optional[np.ndarray]:
    """World-frame positions of each component anchor.

    ``obj_pose_world`` is ``[x, y, z, qw, qx, qy, qz]`` (SAPIEN convention).
    Returns ``(N, 3)`` ndarray or ``None`` for degenerate / missing inputs.
    """
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
    return p[None, :] + anchors_obj @ R.T  # (N, 3)


def transform_approach_dir(
    obj_pose_world: Optional[List[float]],
    component: AffordanceComponent,
) -> Optional[np.ndarray]:
    """World-frame unit approach direction for one component, or None.

    Rotates the stored OBJECT-frame approach direction by the object's current
    world rotation. Returns None if the component has no mined direction or the
    pose is degenerate.
    """
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
) -> Optional[int]:
    """argmin distance from TCP to each anchor. Returns None if no candidates."""
    if anchors_world is None or len(anchors_world) == 0:
        return None
    tcp = np.asarray(tcp_xyz_world, dtype=float).reshape(-1)
    if tcp.shape[0] != 3 or not np.all(np.isfinite(tcp)):
        return None
    diffs = anchors_world - tcp[None, :]
    d2 = np.einsum("ij,ij->i", diffs, diffs)
    return int(np.argmin(d2))


def lookup_components(
    aff_set: AffordanceSet, node: Node,
) -> Optional[List[AffordanceComponent]]:
    """Resolve component list for an object node.

    Lookup order:
      1. exact stable ``entity_key`` (supports actors and articulation links);
      2. legacy actor key without the ``actor:`` prefix;
      3. legacy MS-HAB object id;
      4. canonicalized display name.
    Returns ``None`` if the set is empty or no match.
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
    """True iff this object node has any mined affordance components.

    Primary eligibility test for node classification: an object that has an
    affordance set is treated as ``interactive_object``.
    """
    comps = lookup_components(aff_set, node)
    return bool(comps)
