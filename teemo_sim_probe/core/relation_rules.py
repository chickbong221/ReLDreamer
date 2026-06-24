"""Absolute relation labels from privileged state.

Vocabulary follows ``node.attributes['pair_type']``:

  ee--static_object       (CENTER-based):
      planar-distance, height-offset, contact
  ee--interactive_object  (ANCHOR-based when an asset exists, else CENTER):
      planar-distance, height-offset, orientation-alignment, contact, grasp
  object--object          (mutually exclusive support / contact):
      contact, support

Interactive spatial reference is annotated on the node as
``attributes['spatial_ref'] in {"anchor", "center"}`` so audits can tell
anchor-grounded edges from center fallbacks per frame. ``orientation-
alignment`` is emitted only when both a mined affordance asset AND an
approach_dir exist.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .affordance import (
    lookup_components,
    select_active_component,
    transform_anchors,
    transform_approach_dir,
)
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


def planar_distance_xyz(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def height_offset_xyz(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[2] - b[2])


def planar_distance(a: Node, b: Node) -> Optional[float]:
    pa, pb = _xyz(a), _xyz(b)
    if pa is None or pb is None:
        return None
    return planar_distance_xyz(pa, pb)


def height_offset(a: Node, b: Node) -> Optional[float]:
    pa, pb = _xyz(a), _xyz(b)
    if pa is None or pb is None:
        return None
    return height_offset_xyz(pa, pb)


def _normalize(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _quat_wxyz_to_rotmat(q: np.ndarray) -> Optional[np.ndarray]:
    qn = _normalize(np.asarray(q, dtype=float))
    if qn is None:
        return None
    w, x, y, z = qn
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def tcp_approach_dir_world(
    tcp_pose_world: Optional[np.ndarray], axis_local: List[float]
) -> Optional[np.ndarray]:
    """World-frame unit approach axis of the TCP.

    ``axis_local`` is the gripper-link local axis that points "out" of the
    gripper. Fetch / Panda default is +Z. Rotated by the TCP world quaternion.
    """
    if tcp_pose_world is None or len(tcp_pose_world) < 7:
        return None
    R = _quat_wxyz_to_rotmat(np.asarray(tcp_pose_world[3:7], dtype=float))
    if R is None:
        return None
    d = R @ np.asarray(axis_local, dtype=float).reshape(3)
    return _normalize(d)


def orientation_alignment_angle(
    tcp_dir_world: np.ndarray, aff_dir_world: np.ndarray
) -> float:
    """Angle (radians, [0, pi]) between TCP and affordance approach axes."""
    c = float(np.clip(np.dot(tcp_dir_world, aff_dir_world), -1.0, 1.0))
    return float(np.arccos(c))


# --------------------------------------------------------------------------- #
# Predicates
# --------------------------------------------------------------------------- #
# Static-actor hints: things named like "table"/"wall" are scene structure and
# cannot be grasped, regardless of pair-type classification.
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


def _is_interactive(node: Node) -> bool:
    return node.attributes.get("pair_type") == "interactive_object"


def _resolve_entity(node: Node, state: PrivilegedState):
    """Map a node back to its live simulator entity for force queries."""
    name = node.name
    for seg_id in node.segmentation_ids:
        ent = state.seg_id_map.get(seg_id)
        if ent is not None and getattr(ent, "name", None) == name:
            return ent
    if node.attributes.get("is_mshab_active_target"):
        if node.attributes.get("mshab_kind") == "obj":
            return state.active_obj
        if node.attributes.get("mshab_kind") == "handle":
            return state.active_handle_link
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
# Affordance anchor resolution (shared by interactive spatial + orientation)
# --------------------------------------------------------------------------- #
def _resolve_active_anchor(node: Node, state: PrivilegedState, cfg: dict):
    """Return ``(anchor_world (3,), component, a_star)`` for an interactive node.

    Returns ``(None, None, None)`` when no usable affordance asset / pose
    exists, in which case callers fall back to the object center for spatial
    relations. Also stamps ``node.attributes['affordance_a_star']`` so
    ``temporal_buffer`` can detect ``a_star`` switches.
    """
    aff_set = cfg.get("affordance_set")
    if aff_set is None or getattr(aff_set, "is_empty", lambda: True)():
        return None, None, None
    if state.tcp_pose_world is None or node.pose_world is None:
        return None, None, None
    tcp_world = np.asarray(state.tcp_pose_world[:3], dtype=float)
    if tcp_world.shape[0] != 3 or not np.all(np.isfinite(tcp_world)):
        return None, None, None

    comps = lookup_components(aff_set, node)
    if not comps:
        return None, None, None
    anchors_world = transform_anchors(node.pose_world, comps)
    if anchors_world is None:
        return None, None, None
    a_star = select_active_component(tcp_world, anchors_world)
    if a_star is None:
        return None, None, None

    node.attributes["affordance_a_star"] = int(a_star)
    return anchors_world[a_star], comps[a_star], a_star


# --------------------------------------------------------------------------- #
# Edge builders
# --------------------------------------------------------------------------- #
def ee_static_object_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """ee--static_object: center-based planar-distance, height-offset, contact."""
    ee = graph.get_node("ee")
    if ee is None or ee.pose_world is None:
        return []
    ee_xyz = np.asarray(ee.pose_world[:3], dtype=float)
    prof = cfg["profile"]
    eps_contact = cfg["contact"]["eps_force"]

    edges: List[Edge] = []
    for node in graph.nodes:
        if node.node_type != "object" or _is_interactive(node):
            continue
        if not node.valid_mask:
            continue
        ent = _resolve_entity(node, state)
        obj_xyz = _xyz(node)

        if obj_xyz is not None:
            d = planar_distance_xyz(ee_xyz, obj_xyz)
            edges.append(Edge(
                "ee", node.node_id, "planar-distance",
                bin_label(d, prof["planar_distance"]["edges"],
                          prof["planar_distance"]["labels"]), d,
            ))
            dz = height_offset_xyz(ee_xyz, obj_xyz)
            edges.append(Edge(
                "ee", node.node_id, "height-offset",
                bin_label(dz, prof["height_offset"]["edges"],
                          prof["height_offset"]["labels"]), dz,
            ))

        force = state.ee_object_contact_force(ent)
        edges.append(Edge(
            "ee", node.node_id, "contact",
            "contact" if force > eps_contact else "no-contact",
            force, masked=(force <= eps_contact),
        ))
    return edges


def ee_interactive_object_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """ee--interactive_object: anchor-based spatial + orientation + grasp.

    planar-distance / height-offset use the selected affordance anchor when an
    asset is available, else fall back to the object center (annotated as
    ``node.attributes['spatial_ref']``). orientation-alignment requires a mined
    approach direction. contact + grasp are physical predicates.
    """
    ee = graph.get_node("ee")
    if ee is None or ee.pose_world is None:
        return []
    ee_xyz = np.asarray(ee.pose_world[:3], dtype=float)
    prof = cfg["profile"]
    eps_contact = cfg["contact"]["eps_force"]
    grasp_angle = cfg["grasp"]["max_angle"]
    tcp_axis_local = cfg["grasp"].get("tcp_approach_axis_local", [0.0, 0.0, 1.0])
    align_spec = prof.get("orientation_alignment")

    tcp_dir = tcp_approach_dir_world(state.tcp_pose_world, tcp_axis_local)

    edges: List[Edge] = []
    for node in graph.nodes:
        if node.node_type != "object" or not _is_interactive(node):
            continue
        if not node.valid_mask:
            continue
        ent = _resolve_entity(node, state)

        anchor_world, comp, _ = _resolve_active_anchor(node, state, cfg)
        # Spatial reference: anchor if available, else object center.
        ref_xyz = anchor_world if anchor_world is not None else _xyz(node)
        node.attributes["spatial_ref"] = (
            "anchor" if anchor_world is not None else "center"
        )

        if ref_xyz is not None:
            d = planar_distance_xyz(ee_xyz, ref_xyz)
            edges.append(Edge(
                "ee", node.node_id, "planar-distance",
                bin_label(d, prof["planar_distance"]["edges"],
                          prof["planar_distance"]["labels"]), d,
            ))
            dz = height_offset_xyz(ee_xyz, ref_xyz)
            edges.append(Edge(
                "ee", node.node_id, "height-offset",
                bin_label(dz, prof["height_offset"]["edges"],
                          prof["height_offset"]["labels"]), dz,
            ))

        # orientation-alignment: angle between TCP and affordance approach axes.
        if comp is not None and tcp_dir is not None and align_spec is not None:
            aff_dir = transform_approach_dir(node.pose_world, comp)
            if aff_dir is not None:
                ang = orientation_alignment_angle(tcp_dir, aff_dir)
                edges.append(Edge(
                    "ee", node.node_id, "orientation-alignment",
                    bin_label(ang, align_spec["edges"], align_spec["labels"]),
                    ang,
                ))

        # contact predicate.
        force = state.ee_object_contact_force(ent)
        edges.append(Edge(
            "ee", node.node_id, "contact",
            "contact" if force > eps_contact else "no-contact",
            force, masked=(force <= eps_contact),
        ))

        # grasp predicate (graspable interactive objects only).
        if _graspable(node):
            grasped = state.is_grasping(ent, max_angle=grasp_angle)
            edges.append(Edge(
                "ee", node.node_id, "grasp",
                "grasp" if grasped else "no-grasp",
                1.0 if grasped else 0.0, masked=(not grasped),
            ))
    return edges


def object_object_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """Pairwise object--object: mutually exclusive contact or directed support.

    Bug 3a fix: a node is eligible for object-object physics only when it has
    a non-empty ``segmentation_ids`` list. Maskless graph nodes (e.g. the
    MS-HAB persistent target during an occlusion, or the now-removed phantom
    'body' synthesized by the old local-contact path) carry no real geometry,
    so running ``pairwise_force_vector`` against them produces phantom support
    edges that propagate into ``lose-support`` temporal transitions.
    """
    eps_contact = cfg["contact"]["eps_force"]
    eps_z = cfg["support"]["eps_z"]
    min_vertical_ratio = cfg["support"].get("min_vertical_force_ratio", 0.5)

    objs = [
        n for n in graph.nodes
        if n.node_type == "object"
        and n.valid_mask
        and n.segmentation_ids
    ]
    edges: List[Edge] = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i], objs[j]
            ea = _resolve_entity(a, state)
            eb = _resolve_entity(b, state)

            force_vector = state.pairwise_force_vector(ea, eb)
            force = float(np.linalg.norm(force_vector))
            in_contact = force > eps_contact

            # Support = load-bearing contact: vertical-dominated force AND
            # supporter centre below supported centre.
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
                edges.append(Edge(
                    supporter.node_id, supported.node_id,
                    "support", "support", force,
                ))
            else:
                edges.append(Edge(
                    a.node_id, b.node_id, "contact",
                    "contact" if in_contact else "no-contact",
                    force, masked=(not in_contact),
                ))
                edges.append(Edge(
                    a.node_id, b.node_id, "support", "no-support",
                    force, masked=True,
                ))
    return edges


def build_absolute_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> None:
    """Populate ``graph.edges`` in place with absolute relations.

    Eligibility split: static objects get center-based scene-context edges,
    interactive objects get affordance-grounded manipulation edges.
    """
    graph.edges.extend(ee_static_object_edges(graph, state, cfg))
    graph.edges.extend(ee_interactive_object_edges(graph, state, cfg))
    graph.edges.extend(object_object_edges(graph, state, cfg))
