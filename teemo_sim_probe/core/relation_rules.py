"""Absolute relation labels from privileged state.

ee-object   : planar-distance, height-offset, contact, grasp
object-object: contact, support

orientation-alignment and containment are intentionally deferred (see design
doc) until the basic demo is validated.

A node is "graspable" only if it is an Actor; static structure (table, walls,
cabinet body) and articulation links get the grasp label masked.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .schema import Edge, Graph, Node
from ..adapters.privileged_state import PrivilegedState


# --------------------------------------------------------------------------- #
# Discretization
# --------------------------------------------------------------------------- #
def bin_label(value: float, edges: List[float], labels: List[str]) -> str:
    """Upper-exclusive ascending bins. len(labels) == len(edges) + 1."""
    idx = int(np.searchsorted(edges, value, side="right"))
    idx = min(idx, len(labels) - 1)
    return labels[idx]


# --------------------------------------------------------------------------- #
# Geometry helpers (operate on world pose arrays [x,y,z,qw,qx,qy,qz])
# --------------------------------------------------------------------------- #
def _xyz(node: Node) -> Optional[np.ndarray]:
    if node.pose_world is None:
        return None
    return np.asarray(node.pose_world[:3], dtype=float)


def planar_distance(a: Node, b: Node) -> Optional[float]:
    pa, pb = _xyz(a), _xyz(b)
    if pa is None or pb is None:
        return None
    return float(np.linalg.norm(pa[:2] - pb[:2]))


def height_offset(a: Node, b: Node) -> Optional[float]:
    """z_a - z_b (signed)."""
    pa, pb = _xyz(a), _xyz(b)
    if pa is None or pb is None:
        return None
    return float(pa[2] - pb[2])


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #
# Actors that are static scene structure -> grasp label is meaningless.
_STATIC_HINTS = (
    "table", "wall", "floor", "ground", "counter_body", "cabinet_body",
    "fridge_body", "kitchen_counter", "sink", "shelf",
)


def _graspable(node: Node) -> bool:
    if not bool(node.attributes.get("is_actor", False)):
        return False
    name = node.name.lower()
    if any(h in name for h in _STATIC_HINTS):
        return False
    return True


def _resolve_entity(node: Node, state: PrivilegedState):
    """Map a node back to its live simulator entity for force queries."""
    name = node.name
    for seg_id in node.segmentation_ids:
        ent = state.seg_id_map.get(seg_id)
        if ent is not None and getattr(ent, "name", None) == name:
            return ent
    # Persistent MS-HAB targets: use the cached active handles.
    if node.attributes.get("is_mshab_active_target"):
        if node.attributes.get("mshab_kind") == "obj":
            return state.active_obj
        if node.attributes.get("mshab_kind") == "handle":
            return state.active_handle_link
    # Fall back to first seg-id entity if name didn't match (merged views).
    for seg_id in node.segmentation_ids:
        ent = state.seg_id_map.get(seg_id)
        if ent is not None:
            return ent
    return None


# --------------------------------------------------------------------------- #
# Edge builders
# --------------------------------------------------------------------------- #
def ee_object_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    ee = graph.get_node("ee")
    if ee is None:
        return []
    prof = cfg["profile"]
    eps_contact = cfg["contact"]["eps_force"]
    grasp_angle = cfg["grasp"]["max_angle"]

    edges: List[Edge] = []
    for node in graph.nodes:
        if node.node_type != "object":
            continue
        ent = _resolve_entity(node, state)

        # planar-distance
        d = planar_distance(ee, node)
        if d is not None:
            label = bin_label(
                d,
                prof["planar_distance"]["edges"],
                prof["planar_distance"]["labels"],
            )
            edges.append(Edge("ee", node.node_id, "planar-distance", label, d))

        # height-offset
        dz = height_offset(ee, node)
        if dz is not None:
            label = bin_label(
                dz,
                prof["height_offset"]["edges"],
                prof["height_offset"]["labels"],
            )
            edges.append(Edge("ee", node.node_id, "height-offset", label, dz))

        # contact (both fingers)
        force = state.ee_object_contact_force(ent)
        edges.append(
            Edge(
                "ee", node.node_id, "contact",
                "contact" if force > eps_contact else "no-contact",
                force,
                masked=(force <= eps_contact),   # only draw positive contact
            )
        )

        # grasp (Actors only)
        if _graspable(node):
            grasped = state.is_grasping(ent, max_angle=grasp_angle)
            edges.append(
                Edge(
                    "ee", node.node_id, "grasp",
                    "grasp" if grasped else "no-grasp",
                    1.0 if grasped else 0.0,
                    masked=(not grasped),
                )
            )
    return edges


def object_object_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    eps_contact = cfg["contact"]["eps_force"]
    r_support = cfg["support"]["r_support"]
    eps_z = cfg["support"]["eps_z"]

    objs = [n for n in graph.nodes if n.node_type == "object"]
    edges: List[Edge] = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            ea = _resolve_entity(a, state)
            eb = _resolve_entity(b, state)

            force = state.pairwise_force(ea, eb)
            in_contact = force > eps_contact
            edges.append(
                Edge(
                    a.node_id, b.node_id, "contact",
                    "contact" if in_contact else "no-contact",
                    force,
                    masked=(not in_contact),
                )
            )

            # support: contact + vertical ordering + xy proximity.
            # Emit a directed edge (higher supported-by lower) when it holds.
            if in_contact:
                pa, pb = _xyz(a), _xyz(b)
                if pa is not None and pb is not None:
                    xy = float(np.linalg.norm(pa[:2] - pb[:2]))
                    if xy < r_support:
                        if pa[2] > pb[2] + eps_z:
                            edges.append(
                                Edge(a.node_id, b.node_id, "support",
                                     "support", force)
                            )
                        elif pb[2] > pa[2] + eps_z:
                            edges.append(
                                Edge(b.node_id, a.node_id, "support",
                                     "support", force)
                            )
    return edges


def build_absolute_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> None:
    """Populate ``graph.edges`` in place with absolute relations."""
    graph.edges.extend(ee_object_edges(graph, state, cfg))
    graph.edges.extend(object_object_edges(graph, state, cfg))
