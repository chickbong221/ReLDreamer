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


def _normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _quat_wxyz_to_rotmat(q: np.ndarray) -> Optional[np.ndarray]:
    """SAPIEN quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix."""
    qn = _normalize(q)
    if qn is None:
        return None
    w, x, y, z = qn
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _approach_alignment_deg(
    ee_node: Node, object_node: Node, cfg: Optional[dict]
) -> Optional[float]:
    """Angle (deg) between TCP approach axis (world) and TCP->object vector.

    Safe: returns None if any pose / quaternion / vector is degenerate.
    """
    if cfg is None:
        return None
    if ee_node.pose_world is None or object_node.pose_world is None:
        return None
    if len(ee_node.pose_world) < 7 or len(object_node.pose_world) < 7:
        return None

    p_ee = np.asarray(ee_node.pose_world[:3], dtype=float)
    q_ee = np.asarray(ee_node.pose_world[3:7], dtype=float)
    p_obj = np.asarray(object_node.pose_world[:3], dtype=float)

    R_ee = _quat_wxyz_to_rotmat(q_ee)
    if R_ee is None:
        return None

    a_tcp = _normalize(np.asarray(cfg.get("tcp_axis", [0.0, 0.0, 1.0]), dtype=float))
    if a_tcp is None:
        return None
    a_world = R_ee @ a_tcp

    u = _normalize(p_obj - p_ee)
    if u is None:
        return None

    cos_a = float(np.clip(np.dot(a_world, u), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


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
    # Fall back to the entity that appears under the most seg_ids (mode);
    # compares by identity since entity wrappers may not be hashable.
    ents = [state.seg_id_map.get(s) for s in node.segmentation_ids]
    ents = [e for e in ents if e is not None]
    if not ents:
        return None
    best_ent, best_count = None, 0
    for ent in ents:
        count = sum(1 for e in ents if e is ent)
        if count > best_count:
            best_ent, best_count = ent, count
    return best_ent


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

        # approach-alignment (TCP-centric, robot frame)
        aa_cfg = cfg.get("approach_alignment")
        angle = _approach_alignment_deg(ee, node, aa_cfg)
        if angle is not None:
            label = bin_label(angle, aa_cfg["edges_deg"], aa_cfg["labels"])
            edges.append(
                Edge("ee", node.node_id, "approach-alignment", label, angle)
            )

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
    eps_z = cfg["support"]["eps_z"]
    min_vertical_ratio = cfg["support"].get("min_vertical_force_ratio", 0.5)

    objs = [n for n in graph.nodes if n.node_type == "object"]
    edges: List[Edge] = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            ea = _resolve_entity(a, state)
            eb = _resolve_entity(b, state)

            force_vector = state.pairwise_force_vector(ea, eb)
            force = float(np.linalg.norm(force_vector))
            in_contact = force > eps_contact

            # Support is a load-bearing contact: the force must be primarily
            # vertical and the supporter must be below the supported object.
            # Contact and support are exclusive semantic graph relations.
            support_pair = None
            if in_contact and force > 0.0:
                vertical_ratio = abs(float(force_vector[2])) / force
                if vertical_ratio >= min_vertical_ratio:
                    pa, pb = _xyz(a), _xyz(b)
                    if pa is not None and pb is not None:
                        if pa[2] + eps_z < pb[2]:
                            support_pair = (a, b)
                        elif pb[2] + eps_z < pa[2]:
                            support_pair = (b, a)

            if support_pair is not None:
                supporter, supported = support_pair
                edges.append(
                    Edge(
                        supporter.node_id,
                        supported.node_id,
                        "support",
                        "support",
                        force,
                    )
                )
            else:
                edges.append(
                    Edge(
                        a.node_id, b.node_id, "contact",
                        "contact" if in_contact else "no-contact",
                        force,
                        masked=(not in_contact),
                    )
                )
                # Explicit negative support edge so absence isn't ambiguous
                # in the absolute graph (mirrors contact's negative-emission).
                edges.append(
                    Edge(
                        a.node_id, b.node_id, "support", "no-support",
                        force, masked=True,
                    )
                )
    return edges


def build_absolute_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> None:
    """Populate ``graph.edges`` in place with absolute relations."""
    graph.edges.extend(ee_object_edges(graph, state, cfg))
    graph.edges.extend(object_object_edges(graph, state, cfg))
