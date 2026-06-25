"""Build the two-type node set (``ee`` + ``object``) for one frame.

Implements the detection rules from the design doc:

  R1  exclude segmentation id 0 (background)
  R2  merge gripper / tcp / finger links into the single ``ee`` node
  R3  every non-bg, non-robot Actor  -> object node
  R4  every non-robot Link           -> object node (articulation parts/handles)
  R5  every admitted object comes from current segmentation

Task relevance is decided later by the hard per-subtask whitelist.  This
module deliberately avoids name-based scene filtering so a visible supporter
or articulation link cannot be discarded before the whitelist sees it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .affordance import has_affordance
from .entity_identity import entity_kind, stable_entity_key, stable_node_id
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


def canonical_object_key(entity) -> str:
    """Stable node id. ManiSkill already provides stable simulator objects."""
    return stable_node_id(entity)


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
            entity_kind=entity_kind(entity),
            entity_key=stable_entity_key(entity),
        ),
    )


# --------------------------------------------------------------------------- #
# Main builder
# --------------------------------------------------------------------------- #
def build_nodes(
    obs: dict,
    state: PrivilegedState,
    *,
    camera: Optional[str] = None,
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

        # Every non-empty current mask becomes a candidate.  The whitelist is
        # the sole relevance gate, so supporters and ordinary links are not
        # lost to name or area heuristics before mask registration.
        area = pixel_area(m)
        if area <= 0:
            continue

        # R3 / R4: actor or non-robot link -> object node.
        key = canonical_object_key(entity)
        if key not in nodes:
            nodes[key] = make_object_node(entity, state)
        nodes[key].segmentation_ids.append(seg_id)
        masks.add(key, m)
        nodes[key].pixel_area = masks.area(key)

    nodes["ee"].pixel_area = masks.area("ee")

    return nodes, masks, cam, rgb


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
#   2. whitelist role contains interacted            -> interactive_object
#   3. whitelist role is support only                -> static_object
#   4. otherwise: free actors default interactive, links default static.
def classify_pair_types(nodes: Dict[str, Node], cfg: dict) -> None:
    """Annotate each object node in place with ``attributes['pair_type']``."""
    aff_set = cfg.get("affordance_set")
    for node in nodes.values():
        if node.node_type != "object":
            continue
        attrs = node.attributes

        # (1) affordance asset present -> interactive.
        if aff_set is not None and has_affordance(aff_set, node):
            attrs["pair_type"] = "interactive_object"
            continue
        roles = set(attrs.get("whitelist_roles") or [])
        # (2) interacted members are ordinary interactive objects.
        if "interacted" in roles:
            attrs["pair_type"] = "interactive_object"
            continue
        # (3) direct supporters are static unless they have their own mined
        # affordance, handled above.
        if "support" in roles:
            attrs["pair_type"] = "static_object"
            continue
        # (4) default by entity kind: links are usually structure, free actors
        #     are usually manipulable. Interactive default lets a not-yet-mined
        #     manipulation object still receive center-based interactive
        #     relations (orientation-alignment is simply skipped without an
        #     asset).
        attrs["pair_type"] = (
            "interactive_object" if attrs.get("is_actor") else "static_object"
        )
