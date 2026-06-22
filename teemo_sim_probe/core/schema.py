"""TEEMO manipulation-semantic-graph schema.

One graph per frame. Two node types only (``ee`` and ``object``), matching the
draft's V_t = {v_ee} u {v_i}. Edges carry both a discrete ``label`` and the raw
continuous ``value`` they were discretized from, so nothing is lost in JSON.

This module is pure-python (no torch / maniskill imports) so it can be unit
tested and imported anywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    # Canonical, stable key. "ee" for the end-effector, "actor:<name>" or
    # "link:<name>" for objects (see core.node_builder.canonical_object_key).
    node_id: str
    node_type: str            # "ee" | "object"
    name: str

    # Visibility / mask bookkeeping.
    visible: bool = True
    segmentation_ids: List[int] = field(default_factory=list)
    pixel_area: int = 0

    # World-frame pose [x, y, z, qw, qx, qy, qz] (SAPIEN wxyz convention).
    pose_world: Optional[List[float]] = None

    # Persistence (draft's persistent-node mechanism). steps_since_seen == tau_i.
    persistent: bool = False
    steps_since_seen: int = 0
    source: str = "segmentation"     # "segmentation" | "mshab_task"
    # True for nodes retained across a visibility gap with the LAST observed
    # pose (no live SAPIEN refresh). MS-HAB-active-target persistents stay False
    # because they get fresh poses from the simulator each frame.
    frozen_pose: bool = False

    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #
@dataclass
class Edge:
    src: str
    dst: str
    relation: str             # e.g. "planar-distance", "contact", "grasp"
    label: str                # discrete bin, e.g. "near", "contact", "gain-grasp"
    raw_value: Optional[float] = None
    temporal: bool = False    # True for *-change / *-transition relations
    masked: bool = False      # True => kept in JSON but should NOT be drawn

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

    # ---- node helpers ---------------------------------------------------- #
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

    # ---- serialization --------------------------------------------------- #
    def to_dict(self, drawable_only: bool = False) -> Dict[str, Any]:
        edges = self.edges
        if drawable_only:
            edges = [e for e in edges if not e.masked]
        return {
            "frame": self.frame,
            "env_id": self.env_id,
            "camera": self.camera,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in edges],
            "meta": self.meta,
        }

    def to_json(self, drawable_only: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(drawable_only=drawable_only), indent=indent)

    def save(self, path: str, drawable_only: bool = False) -> None:
        with open(path, "w") as f:
            f.write(self.to_json(drawable_only=drawable_only))
