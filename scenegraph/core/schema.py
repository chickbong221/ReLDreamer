"""TEEMO manipulation scene-graph schema.

One graph per frame. Two node types (``ee`` and ``object``). ``index`` is stable
after first sight until capacity is reached; a newly seen instance then reuses
the oldest resident instance's index.

Facts are hyper-relational: one :class:`Edge` per admissible ``(src, rel, dst)``
instance carrying an absolute state ``label`` and, for families that define one,
a temporal-change ``temp_label`` over the last ``K`` frames.

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

    # Episode-stable identity for latent-slot alignment. 0 is padding, 1 is the
    # end effector, 2+ are object instances drawn from a per-episode random
    # permutation so the id carries no category information.
    uid: int = 0

    # Append-only vertex index, assigned by EntityRegistry on first sight.
    index: Optional[int] = None

    steps_since_seen: int = 0
    source: str = "segmentation"

    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #
@dataclass
class Edge:
    src: str
    dst: str
    relation: str
    label: str                          # absolute state sigma
    temp_label: Optional[str] = None    # temporal change delta, None when absent
    raw_value: Optional[float] = None
    # A stale fact is the last observed object--object physical state touching a
    # node that is no longer visible. It is never recomputed from mixed-time
    # poses and never carries a temporal label.
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "env_id": self.env_id,
            "camera": self.camera,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "meta": self.meta,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())
