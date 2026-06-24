"""Snapshot helpers used by the selector to retain frozen-pose object nodes."""

from __future__ import annotations

from typing import Dict

from .schema import Node


# Per-frame MS-HAB task-state attributes that must NOT leak across retention:
# stamping them on a frozen snapshot would keep stale flags after the subtask
# advances.
_DYNAMIC_MSHAB_ATTRS = (
    "is_mshab_active_target",
    "mshab_kind",
    "mshab_obj_id",
    "affordance_a_star",
)


def _stripped_attrs(attrs: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in attrs.items() if k not in _DYNAMIC_MSHAB_ATTRS}


def _snapshot(node: Node) -> Node:
    """Frozen copy of a visible node used as the retention seed."""
    return Node(
        node_id=node.node_id,
        node_type=node.node_type,
        name=node.name,
        visible=node.visible,
        segmentation_ids=list(node.segmentation_ids),
        pixel_area=node.pixel_area,
        pose_world=list(node.pose_world) if node.pose_world else None,
        persistent=node.persistent,
        steps_since_seen=node.steps_since_seen,
        source=node.source,
        frozen_pose=False,
        attributes=_stripped_attrs(node.attributes),
    )
