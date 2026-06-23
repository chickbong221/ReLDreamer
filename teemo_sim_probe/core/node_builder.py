"""Build the two-type node set (``ee`` + ``object``) for one frame.

Implements the detection rules from the design doc:

  R1  exclude segmentation id 0 (background)
  R2  merge gripper / tcp / finger links into the single ``ee`` node
  R3  every non-bg, non-robot Actor  -> object node
  R4  every non-robot Link           -> object node (articulation parts/handles)
  R5  exclude helper/goal actors by default (``include_goals`` to keep)
  R6  minimum visible-area threshold (with active-target exception)
  R7  add MS-HAB active target even if not currently visible (persistent)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .affordance import canonical_affordance_key, has_affordance
from .schema import Node
from .mask_extractor import (
    MaskAccumulator,
    extract_camera_obs,
    mask_for_id,
    pixel_area,
    pick_camera,
    unique_seg_ids,
)
from ..adapters.privileged_state import PrivilegedState, pose_to_world_array


# --------------------------------------------------------------------------- #
# Entity classification
# --------------------------------------------------------------------------- #
def _entity_name(entity) -> str:
    return getattr(entity, "name", str(entity))


def _is_actor(entity) -> bool:
    return type(entity).__name__ == "Actor"


def _is_link(entity) -> bool:
    return type(entity).__name__ == "Link"


def _is_robot_link(entity, robot_links: set) -> bool:
    if not _is_link(entity):
        return False
    if entity in robot_links:
        return True
    # Fallback by name match (merged views can break identity equality).
    names = {getattr(l, "name", None) for l in robot_links}
    return getattr(entity, "name", None) in names


def _is_ee_link(entity, ee_links: List[Any]) -> bool:
    if entity in ee_links:
        return True
    ee_names = {getattr(l, "name", None) for l in ee_links}
    return getattr(entity, "name", None) in ee_names


# Substrings that mark a helper/goal/visualization actor (R5).
_GOAL_HINTS = ("goal", "ee_rest", "site", "marker", "target_site")

# Substrings that mark scene background / static furniture clutter.
# MS-HAB exposes these as actors (e.g. "scs-[0]_scene_background",
# "scs-[0]_frl_apartment_table_02-4"); the draft keeps only the end-effector
# and manipulation-relevant objects, so these are filtered by default.
_BACKGROUND_HINTS = (
    "scene_background", "background", "_bg",
)
_STATIC_SCENE_HINTS = (
    "frl_apartment", "apartment_", "_wall", "_floor", "_ceiling",
    "room_", "stage", "arena",
)

# Substrings that strongly imply a STATIC scene object (support surface /
# obstacle / structure) even when not filtered out of the graph entirely.
# Used only to set ``pair_type`` when the object has no affordance set.
_STATIC_PAIR_HINTS = (
    "table", "wall", "floor", "ground", "counter", "cabinet_body",
    "fridge_body", "sink", "shelf", "support", "obstacle", "ceiling",
)

# Articulated-part hints that are manipulation-relevant => interactive even
# without a mined affordance asset (e.g. drawer / cabinet / fridge handles).
# Active MS-HAB handle links are also forced interactive below.
_INTERACTIVE_PART_HINTS = ("handle", "knob", "lever", "drawer")


def _is_helper_goal(entity) -> bool:
    name = _entity_name(entity).lower()
    return any(h in name for h in _GOAL_HINTS)


def _is_background(entity) -> bool:
    name = _entity_name(entity).lower()
    return any(h in name for h in _BACKGROUND_HINTS)


def _is_static_scene(entity) -> bool:
    name = _entity_name(entity).lower()
    return any(h in name for h in _STATIC_SCENE_HINTS)


def canonical_object_key(entity) -> str:
    """Stable node id. ManiSkill already provides stable simulator objects."""
    if _is_actor(entity):
        return f"actor:{_entity_name(entity)}"
    if _is_link(entity):
        return f"link:{_entity_name(entity)}"
    return f"obj:{_entity_name(entity)}"


# --------------------------------------------------------------------------- #
# Node factories
# --------------------------------------------------------------------------- #
def make_ee_node(state: PrivilegedState) -> Node:
    return Node(
        node_id="ee",
        node_type="ee",
        name="end_effector",
        visible=False,                      # set True once a mask is merged in
        pose_world=list(state.tcp_pose_world)
        if state.tcp_pose_world is not None
        else None,
        source="segmentation",
    )


def make_object_node(entity, state: PrivilegedState) -> Node:
    pose_world = None
    pose = getattr(entity, "pose", None)
    if pose is not None:
        try:
            pose_world = list(pose_to_world_array(pose, state.env_idx))
        except Exception:
            pose_world = None
    return Node(
        node_id=canonical_object_key(entity),
        node_type="object",
        name=_entity_name(entity),
        visible=True,
        pose_world=pose_world,
        source="segmentation",
        attributes=dict(
            is_actor=_is_actor(entity),
            is_link=_is_link(entity),
            is_articulation_link=_is_link(entity),
        ),
    )


def make_persistent_target_node(entity, state: PrivilegedState, kind: str) -> Node:
    node = make_object_node(entity, state)
    node.visible = False
    node.persistent = True
    node.source = "mshab_task"
    node.attributes["is_mshab_active_target"] = True
    node.attributes["mshab_kind"] = kind          # "obj" | "handle"
    if kind == "obj":
        # Carry the canonical YCB key forward so affordance lookup works even
        # when the segmentation node uses the env-prefixed scene name. Handles
        # do not have an affordance asset, so we deliberately omit it for them.
        oid = state.active_obj_id
        canonical = canonical_affordance_key(oid) if oid else \
            canonical_affordance_key(node.name)
        if canonical:
            node.attributes["mshab_obj_id"] = canonical
    return node


# --------------------------------------------------------------------------- #
# Main builder
# --------------------------------------------------------------------------- #
def build_nodes(
    obs: dict,
    state: PrivilegedState,
    *,
    camera: Optional[str] = None,
    include_goals: bool = False,
    include_background: bool = False,
    include_static_scene: bool = False,
    min_pixels: int = 32,
    min_area_ratio: float = 0.0005,
    seg_override: Optional[np.ndarray] = None,
    rgb_override: Optional[np.ndarray] = None,
    camera_override: Optional[str] = None,
) -> Tuple[Dict[str, Node], MaskAccumulator, str, np.ndarray]:
    """Return (nodes_by_id, masks, camera_name, rgb).

    If ``seg_override`` is given the segmentation image is taken from it (and
    ``rgb_override`` as the backdrop) instead of from ``obs``. This is the
    MS-HAB depth-mode path, where the policy obs has no segmentation and the
    probe reads it from the unwrapped env separately.
    """
    if seg_override is not None:
        seg = seg_override
        rgb = rgb_override if rgb_override is not None else \
            np.zeros((*seg.shape, 3), dtype=np.uint8)
        cam = camera_override or camera or "fetch_head"
    else:
        cam = pick_camera(obs, camera)
        rgb, seg, _depth = extract_camera_obs(obs, cam, state.env_idx)
    H, W = seg.shape
    masks = MaskAccumulator(H, W)

    nodes: Dict[str, Node] = {}

    # 1. ee node always exists.
    nodes["ee"] = make_ee_node(state)

    # 2. iterate visible seg ids (R1 excludes id 0).
    for seg_id in unique_seg_ids(seg, exclude_background=True):
        entity = state.seg_id_map.get(seg_id)
        if entity is None:
            continue
        m = mask_for_id(seg, seg_id)

        # R2: robot links -> ee (if gripper) or excluded.
        if _is_robot_link(entity, state.robot_links):
            if _is_ee_link(entity, state.ee_links):
                masks.add("ee", m)
                nodes["ee"].visible = True
                nodes["ee"].segmentation_ids.append(seg_id)
            continue

        # R5: helper/goal actors excluded unless requested.
        if _is_helper_goal(entity) and not include_goals:
            continue

        # Background scene actor excluded unless requested (draft filters bg).
        if _is_background(entity) and not include_background:
            continue

        # Static scene furniture (apartment props, walls) excluded by default.
        if _is_static_scene(entity) and not include_static_scene:
            continue

        # R6: area threshold.
        area = pixel_area(m)
        if area < min_pixels and (area / (H * W)) < min_area_ratio:
            continue

        # R3 / R4: actor or non-robot link -> object node.
        key = canonical_object_key(entity)
        if key not in nodes:
            nodes[key] = make_object_node(entity, state)
        nodes[key].segmentation_ids.append(seg_id)
        masks.add(key, m)
        nodes[key].pixel_area = masks.area(key)

    nodes["ee"].pixel_area = masks.area("ee")

    # 7. MS-HAB active target persistence (added even if mask empty).
    if state.is_mshab:
        _add_mshab_targets(nodes, masks, state)

    return nodes, masks, cam, rgb


def _add_mshab_targets(
    nodes: Dict[str, Node], masks: MaskAccumulator, state: PrivilegedState
) -> None:
    """R7: ensure the current subtask's obj / handle exist as object nodes."""
    for entity, kind in (
        (state.active_obj, "obj"),
        (state.active_handle_link, "handle"),
    ):
        if entity is None:
            continue
        key = canonical_object_key(entity)
        if key in nodes:
            nodes[key].attributes["is_mshab_active_target"] = True
            nodes[key].attributes["mshab_kind"] = kind
            if kind == "obj":
                # mshab_obj_id lets affordance lookup work even when the
                # segmentation node's display name is env-prefixed
                # ("env-0_024_bowl-3") rather than the canonical YCB key.
                oid = state.active_obj_id
                canonical = canonical_affordance_key(oid) if oid else \
                    canonical_affordance_key(nodes[key].name)
                if canonical:
                    nodes[key].attributes["mshab_obj_id"] = canonical
        else:
            # Not visible this frame -> persistent, mask empty.
            nodes[key] = make_persistent_target_node(entity, state, kind)


# --------------------------------------------------------------------------- #
# Pair-type classification (eligibility-based relation vocabulary)
# --------------------------------------------------------------------------- #
# Every object node is classified as exactly one of
#   "static_object"      -> center-based scene context + contact state
#   "interactive_object" -> affordance-grounded manipulation state
# stored on ``node.attributes["pair_type"]``. The ee node is left untyped.
#
# Eligibility rule (most specific wins):
#   1. object has a mined affordance set            -> interactive_object
#   2. object is the MS-HAB active handle link      -> interactive_object
#   3. name matches an interactive part hint         -> interactive_object
#   4. name matches a static-pair hint               -> static_object
#   5. otherwise: free actors default interactive, links default static.
# Rule (1) is the primary "has-an-affordance-set" criterion; (2)-(5) are
# fallbacks for objects not yet mined or benchmarks that have no asset yet.
def classify_pair_types(nodes: Dict[str, Node], cfg: dict) -> None:
    """Annotate each object node in place with ``attributes['pair_type']``."""
    aff_set = cfg.get("affordance_set")
    for node in nodes.values():
        if node.node_type != "object":
            continue
        name = node.name.lower()
        attrs = node.attributes

        # (1) affordance asset present -> interactive.
        if aff_set is not None and has_affordance(aff_set, node):
            attrs["pair_type"] = "interactive_object"
            continue
        # (2) active MS-HAB handle link -> interactive (no asset, but it is a
        #     manipulation-relevant articulated part this subtask).
        if (attrs.get("is_mshab_active_target")
                and attrs.get("mshab_kind") == "handle"):
            attrs["pair_type"] = "interactive_object"
            continue
        # (3) interactive part by name (handle / knob / lever / drawer).
        if any(h in name for h in _INTERACTIVE_PART_HINTS):
            attrs["pair_type"] = "interactive_object"
            continue
        # (4) static structure / support surface by name.
        if any(h in name for h in _STATIC_PAIR_HINTS):
            attrs["pair_type"] = "static_object"
            continue
        # (5) default by entity kind: links are usually structure, free actors
        #     are usually manipulable. Interactive default lets a not-yet-mined
        #     manipulation object still receive center-based interactive
        #     relations (orientation-alignment is simply skipped without an
        #     asset).
        attrs["pair_type"] = (
            "interactive_object" if attrs.get("is_actor") else "static_object"
        )
