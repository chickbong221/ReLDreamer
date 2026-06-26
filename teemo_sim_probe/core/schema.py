"""TEEMO manipulation-semantic-graph schema.

One graph per frame. Two node types only (``ee`` and ``object``). Edges carry
both a discrete ``label`` and the raw continuous ``value``.

Pure-python (no torch / maniskill imports).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    node_id: str
    node_type: str            # "ee" | "object"
    name: str

    visible: bool = True
    segmentation_ids: List[int] = field(default_factory=list)
    pixel_area: int = 0

    pose_world: Optional[List[float]] = None

    persistent: bool = False
    steps_since_seen: int = 0
    source: str = "segmentation"
    frozen_pose: bool = False

    # Slot bookkeeping. slot_id is None for ee and for candidates that did not
    # earn a slot this frame.
    slot_id: Optional[int] = None
    entity_id: Optional[str] = None
    valid_mask: bool = True           # False == padding slot
    reset_flag: bool = False          # True iff this slot just changed identity

    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def padding_node(slot_id: int) -> "Node":
    """Padding node for an unused object slot."""
    return Node(
        node_id=f"<pad:{slot_id}>",
        node_type="object",
        name="<pad>",
        visible=False,
        slot_id=slot_id,
        valid_mask=False,
    )


# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #
@dataclass
class Edge:
    src: str
    dst: str
    relation: str
    label: str
    raw_value: Optional[float] = None
    temporal: bool = False
    masked: bool = False
    # A stale edge is the last fully observed relation for a pair touching a
    # frozen persistent node. It is never recomputed from mixed-time poses.
    stale: bool = False
    observed_frame: Optional[int] = None
    age: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
@dataclass
class Graph:
    frame: int
    env_id: str
    camera: str
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def node_ids(self) -> List[str]:
        return [n.node_id for n in self.nodes]

    def get_node(self, node_id: str) -> Optional[Node]:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def upsert_node(self, node: Node) -> None:
        for i, n in enumerate(self.nodes):
            if n.node_id == node.node_id:
                self.nodes[i] = node
                return
        self.nodes.append(node)

    def valid_nodes(self) -> List[Node]:
        return [n for n in self.nodes if n.valid_mask]

    def to_dict(self, drawable_only: bool = False) -> Dict[str, Any]:
        nodes = self.nodes
        edges = self.edges
        if drawable_only:
            nodes = [n for n in nodes if n.valid_mask]
            edges = [e for e in edges if not e.masked]
        return {
            "frame": self.frame,
            "env_id": self.env_id,
            "camera": self.camera,
            "nodes": [n.to_dict() for n in nodes],
            "edges": [e.to_dict() for e in edges],
            "meta": self.meta,
        }

    def to_json(self, drawable_only: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(drawable_only=drawable_only), indent=indent)

    def save(self, path: str, drawable_only: bool = False) -> None:
        with open(path, "w") as f:
            f.write(self.to_json(drawable_only=drawable_only))
