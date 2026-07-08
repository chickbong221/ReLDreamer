"""Build the two-type node set (``ee`` + ``object``) for one frame.

The builder excludes background, folds gripper links into the single ``ee``
node, and creates object nodes for visible non-robot actors and links.

Task relevance is decided later by the hard per-subtask whitelist.  This
module deliberately avoids name-based scene filtering so a visible supporter
or articulation link cannot be discarded before the whitelist sees it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .affordance import has_affordance
from .entity_identity import entity_kind, stable_entity_key, stable_node_id
from .schema import Node
from .mask_extractor import (
    MaskAccumulator,
    extract_camera_obs,
    mask_for_id,
    pick_camera,
)
from ..adapters.privileged_state import (
    PrivilegedState,
    entity_pose_world_array,
)


# --------------------------------------------------------------------------- #
# Entity classification
# --------------------------------------------------------------------------- #
def _entity_name(entity) -> str:
    return getattr(entity, "name", str(entity))


def _is_actor(entity) -> bool:
    return type(entity).__name__ == "Actor"


def _is_link(entity) -> bool:
    return type(entity).__name__ == "Link"


def _is_robot_link(entity, robot_links: set, link_names=None) -> bool:
    if not _is_link(entity):
        return False
    if entity in robot_links:
        return True
    # Fallback by name match (merged views can break identity equality).
    if link_names is None:
        link_names = {getattr(l, "name", None) for l in robot_links}
    return getattr(entity, "name", None) in link_names


def _is_ee_link(entity, ee_links: List[Any], link_names=None) -> bool:
    if entity in ee_links:
        return True
    if link_names is None:
        link_names = {getattr(l, "name", None) for l in ee_links}
    return getattr(entity, "name", None) in link_names


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
    try:
        arr = entity_pose_world_array(entity, state.env_idx)
        pose_world = list(arr) if arr is not None else None
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
def _ingest_camera(
    seg: np.ndarray,
    state: PrivilegedState,
    nodes: Dict[str, Node],
    area_by_key: Dict[str, int],
    masks: MaskAccumulator,
    *,
    need_masks: bool,
    admit: Optional[Callable[[Any], bool]] = None,
) -> None:
    """Union one camera's segmentation into the shared node dict.

    ``admit`` is an optional early whitelist gate: entities it rejects are
    skipped before node construction. It must admit a superset of what the
    downstream ``apply_whitelist`` keeps so the final graph is unchanged.
    """
    ids, counts = np.unique(seg, return_counts=True)
    robot_link_names = getattr(state, "robot_link_names", None)
    ee_link_names = getattr(state, "ee_link_names", None)
    for seg_id, count in zip(ids, counts):
        seg_id = int(seg_id)
        if seg_id == 0 or count <= 0:
            continue
        entity = state.seg_id_map.get(seg_id)
        if entity is None:
            continue

        if _is_robot_link(entity, state.robot_links, robot_link_names):
            if _is_ee_link(entity, state.ee_links, ee_link_names):
                if need_masks:
                    masks.add("ee", mask_for_id(seg, seg_id))
                nodes["ee"].visible = True
                nodes["ee"].segmentation_ids.append(seg_id)
                area_by_key["ee"] += int(count)
            continue

        if admit is not None and not admit(entity):
            continue

        key = canonical_object_key(entity)
        if key not in nodes:
            nodes[key] = make_object_node(entity, state)
            area_by_key[key] = 0
        nodes[key].segmentation_ids.append(seg_id)
        area_by_key[key] += int(count)
        if need_masks:
            masks.add(key, mask_for_id(seg, seg_id))
        nodes[key].pixel_area = area_by_key[key]


def build_nodes(
    obs: dict,
    state: PrivilegedState,
    *,
    camera: Optional[str] = None,
    seg_override: Optional[np.ndarray] = None,
    seg_overrides: Optional[Dict[str, np.ndarray]] = None,
    rgb_override: Optional[np.ndarray] = None,
    camera_override: Optional[str] = None,
    primary_camera: Optional[str] = None,
    need_masks: bool = True,
    admit: Optional[Callable[[Any], bool]] = None,
) -> Tuple[Dict[str, Node], MaskAccumulator, str, np.ndarray]:
    """Return (nodes_by_id, masks, camera_name, rgb).

    ``seg_overrides`` (dict of ``cam -> [H, W]``) unions visibility across
    cameras; masks are collected only for ``primary_camera``. ``seg_override``
    (singular) is the single-camera path used by the offline probe.
    """
    if seg_overrides is not None:
        if not seg_overrides:
            raise ValueError("seg_overrides is empty")
        cam = primary_camera or camera_override or camera or next(iter(seg_overrides))
        if cam not in seg_overrides:
            cam = next(iter(seg_overrides))
        primary_seg = seg_overrides[cam]
        rgb = rgb_override if rgb_override is not None else \
            np.zeros((*primary_seg.shape, 3), dtype=np.uint8)
        H, W = primary_seg.shape
    elif seg_override is not None:
        seg_overrides = {camera_override or camera or "fetch_head": seg_override}
        cam = next(iter(seg_overrides))
        rgb = rgb_override if rgb_override is not None else \
            np.zeros((*seg_override.shape, 3), dtype=np.uint8)
        H, W = seg_override.shape
    else:
        cam = pick_camera(obs, camera)
        rgb, seg, _depth = extract_camera_obs(obs, cam, state.env_idx)
        seg_overrides = {cam: seg}
        H, W = seg.shape

    masks = MaskAccumulator(H, W)
    nodes: Dict[str, Node] = {"ee": make_ee_node(state)}
    area_by_key: Dict[str, int] = {"ee": 0}

    for cam_name, cam_seg in seg_overrides.items():
        _ingest_camera(
            cam_seg, state, nodes, area_by_key, masks,
            need_masks=need_masks and cam_name == cam,
            admit=admit,
        )

    nodes["ee"].pixel_area = area_by_key["ee"]
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
        #     relations (compatibility edges are simply skipped without an
        #     asset).
        attrs["pair_type"] = (
            "interactive_object" if attrs.get("is_actor") else "static_object"
        )
