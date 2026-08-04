"""Build the two-type node set (``ee`` + ``object``) for one frame.

The builder excludes background, folds gripper links into the single ``ee``
node, and creates object nodes for visible non-robot actors and links. It also
pools each node's appearance: the mean masked depth per camera, in metres.

Task relevance is decided later by the hard per-subtask whitelist.  This
module deliberately avoids name-based scene filtering so a visible supporter
or articulation link cannot be discarded before the whitelist sees it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

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
class _DepthPool:
    """Per-node, per-camera masked depth accumulator."""

    def __init__(self, n_cams: int):
        self.n_cams = n_cams
        self.sums: Dict[str, List[float]] = {}
        self.counts: Dict[str, List[int]] = {}

    def ensure(self, key: str) -> None:
        if key not in self.sums:
            self.sums[key] = [0.0] * self.n_cams
            self.counts[key] = [0] * self.n_cams

    def add(self, key: str, cam_slot: int, total: float, pixels: int) -> None:
        self.ensure(key)
        self.sums[key][cam_slot] += float(total)
        self.counts[key][cam_slot] += int(pixels)

    def mean(self, key: str) -> Optional[List[Optional[float]]]:
        if key not in self.sums:
            return None
        out: List[Optional[float]] = []
        for c in range(self.n_cams):
            n = self.counts[key][c]
            out.append(self.sums[key][c] / n if n > 0 else None)
        return out


def _ingest_camera(
    seg: np.ndarray,
    depth: Optional[np.ndarray],
    state: PrivilegedState,
    nodes: Dict[str, Node],
    area_by_key: Dict[str, int],
    pool: _DepthPool,
    masks: MaskAccumulator,
    *,
    cam_slot: int,
    depth_scale: float,
    need_masks: bool,
    admit: Optional[Callable[[Any], bool]] = None,
) -> None:
    """Union one camera's segmentation into the shared node dict.

    Per-segment pixel counts and depth sums come from two ``bincount`` passes,
    so appearance pooling costs one sweep of the frame and never materialises a
    per-node mask.

    ``admit`` is an optional early whitelist gate: entities it rejects are
    skipped before node construction. It must admit a superset of what the
    downstream ``apply_whitelist`` keeps so the final graph is unchanged.
    """
    flat = seg.reshape(-1)
    counts_by_id = np.bincount(flat)
    if depth is not None:
        weights = depth.reshape(-1).astype(np.float64) * depth_scale
        sums_by_id = np.bincount(flat, weights=weights)
    else:
        sums_by_id = None

    robot_link_names = getattr(state, "robot_link_names", None)
    ee_link_names = getattr(state, "ee_link_names", None)
    for seg_id in np.nonzero(counts_by_id)[0]:
        seg_id = int(seg_id)
        if seg_id == 0:
            continue
        count = int(counts_by_id[seg_id])
        total = float(sums_by_id[seg_id]) if sums_by_id is not None else 0.0
        entity = state.seg_id_map.get(seg_id)
        if entity is None:
            continue

        if _is_robot_link(entity, state.robot_links, robot_link_names):
            if _is_ee_link(entity, state.ee_links, ee_link_names):
                if need_masks:
                    masks.add("ee", mask_for_id(seg, seg_id))
                nodes["ee"].visible = True
                nodes["ee"].segmentation_ids.append(seg_id)
                area_by_key["ee"] += count
                if sums_by_id is not None:
                    pool.add("ee", cam_slot, total, count)
            continue

        if admit is not None and not admit(entity):
            continue

        key = canonical_object_key(entity)
        if key not in nodes:
            nodes[key] = make_object_node(entity, state)
            area_by_key[key] = 0
        nodes[key].segmentation_ids.append(seg_id)
        area_by_key[key] += count
        if sums_by_id is not None:
            pool.add(key, cam_slot, total, count)
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
    depth_overrides: Optional[Dict[str, np.ndarray]] = None,
    rgb_override: Optional[np.ndarray] = None,
    camera_override: Optional[str] = None,
    primary_camera: Optional[str] = None,
    camera_order: Optional[List[str]] = None,
    need_masks: bool = True,
    admit: Optional[Callable[[Any], bool]] = None,
    depth_scale: float = 1e-3,
) -> Tuple[Dict[str, Node], MaskAccumulator, str, np.ndarray]:
    """Return (nodes_by_id, masks, camera_name, rgb).

    ``seg_overrides`` (dict of ``cam -> [H, W]``) unions visibility across
    cameras; masks are collected only for ``primary_camera``. ``seg_override``
    (singular) is the single-camera path used by the offline probe.

    ``camera_order`` fixes which appearance slot each camera writes to and must
    stay constant for the whole run; it defaults to sorted camera names.
    ``depth_overrides`` supplies ``cam -> [H, W]`` raw depth; slots without a
    depth map produce ``None`` and fall back to the selector's retained value.
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
        rgb, seg, depth = extract_camera_obs(obs, cam, state.env_idx)
        seg_overrides = {cam: seg}
        if depth_overrides is None and depth is not None:
            depth_overrides = {cam: np.asarray(depth).squeeze()}
        H, W = seg.shape

    order = list(camera_order) if camera_order else sorted(seg_overrides)
    slot_of = {name: i for i, name in enumerate(order)}
    n_cams = len(order)

    masks = MaskAccumulator(H, W)
    nodes: Dict[str, Node] = {"ee": make_ee_node(state)}
    area_by_key: Dict[str, int] = {"ee": 0}
    pool = _DepthPool(n_cams)

    for cam_name, cam_seg in seg_overrides.items():
        slot = slot_of.get(cam_name)
        if slot is None:
            continue
        cam_depth = None
        if depth_overrides is not None:
            cam_depth = depth_overrides.get(cam_name)
        _ingest_camera(
            cam_seg, cam_depth, state, nodes, area_by_key, pool, masks,
            cam_slot=slot,
            depth_scale=depth_scale,
            need_masks=need_masks and cam_name == cam,
            admit=admit,
        )

    nodes["ee"].pixel_area = area_by_key["ee"]
    for key, node in nodes.items():
        node.feat = pool.mean(key) or [None] * n_cams
    return nodes, masks, cam, rgb
