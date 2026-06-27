"""Absolute relation labels from privileged state.

Vocabulary (independent of pair_type):

* Event:     ``contact``, ``grasp``, ``support``
* Spatial:   ``planar-distance`` (object-center), ``height-offset`` (object-center)
* Affordance: ``grasp-compatibility``, ``contact-compatibility``
  Emitted only when (a) the object has affordance components, (b) the
  current ``planar-distance`` is the closest bin (``near``), and (c) the
  whitelist asset records that interaction type for the object.
  ``contact-compatibility`` is masked while the object is grasped, mirroring
  the contact-edge mask.

Bin edges come from the active per-(subtask, target) whitelist
(``cfg["bin_edges"]``); ``cfg["profile"]`` is used as a fallback for any
relation the asset omits. The per-object interaction-type map lives in
``cfg["interaction_types"]`` (also wired from the whitelist by the builder).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .affordance import (
    AffordanceComponent,
    CompatibilityMeasurement,
    compatibility_components,
    lookup_components,
    select_active_component,
    transform_anchors,
)
from .schema import Edge, Graph, Node
from ..adapters.privileged_state import PrivilegedState


# --------------------------------------------------------------------------- #
# Canonical label vocabulary (relation -> labels, in ascending bin order)
# --------------------------------------------------------------------------- #
SPATIAL_LABELS: Dict[str, List[str]] = {
    "planar-distance": ["near", "medium", "far"],
    "height-offset": ["below", "level", "above"],
}
COMPAT_LABELS: Dict[str, List[str]] = {
    "grasp-compatibility": ["match", "partial-match", "poor-match"],
    "contact-compatibility": ["match", "partial-match", "poor-match"],
}
CHANGE_LABELS: Dict[str, List[str]] = {
    "planar-distance-change": [
        "approaching-fast", "approaching-slow", "stable-distance",
        "receding-slow", "receding-fast",
    ],
    "height-offset-change": [
        "lowering-fast", "lowering-slow", "stable-height",
        "rising-slow", "rising-fast",
    ],
    "grasp-compatibility-change": [
        "grasp-fit-better-fast", "grasp-fit-better-slow", "stable-grasp-fit",
        "grasp-fit-worse-slow", "grasp-fit-worse-fast",
    ],
    "contact-compatibility-change": [
        "contact-fit-better-fast", "contact-fit-better-slow",
        "stable-contact-fit",
        "contact-fit-worse-slow", "contact-fit-worse-fast",
    ],
}
ALL_LABELS: Dict[str, List[str]] = {
    **SPATIAL_LABELS, **COMPAT_LABELS, **CHANGE_LABELS,
}


def bin_label(value: float, edges: List[float], labels: List[str]) -> str:
    """Upper-exclusive ascending bins. ``len(labels) == len(edges) + 1``."""
    idx = int(np.searchsorted(edges, value, side="right"))
    idx = min(idx, len(labels) - 1)
    return labels[idx]


def _get_bin_spec(cfg: dict, relation: str) -> Optional[Tuple[List[float], List[str]]]:
    """Resolve ``(edges, labels)`` for ``relation`` from cfg.

    Whitelist-derived ``cfg["bin_edges"]`` wins; ``cfg["profile"]`` is the
    fallback (legacy snake_case keys). Returns None when no source provides
    edges for this relation.
    """
    edges_map = cfg.get("bin_edges") or {}
    edges = edges_map.get(relation)
    if edges is None:
        profile = cfg.get("profile") or {}
        spec = profile.get(relation.replace("-", "_"))
        if not isinstance(spec, dict):
            return None
        edges = spec.get("edges")
    if not edges:
        return None
    labels = ALL_LABELS.get(relation)
    if labels is None or len(labels) != len(edges) + 1:
        return None
    return list(edges), list(labels)


def _compat_norm(cfg: dict) -> Dict[str, float]:
    norm = cfg.get("compat_norm") or {}
    out = {
        "pos": float(norm.get("pos", 0.10)),
        "orient": float(norm.get("orient", np.pi / 2.0)),
        "width": float(norm.get("width", 0.04)),
    }
    for k, v in out.items():
        if not np.isfinite(v) or v <= 0:
            out[k] = 1.0
    return out


def _interaction_types(cfg: dict, key: Optional[str]) -> Set[str]:
    if key is None:
        return set()
    table: Dict[str, Set[str]] = cfg.get("interaction_types") or {}
    return set(table.get(key, set()))


# Pose arrays are ``[x, y, z, qw, qx, qy, qz]`` (SAPIEN).
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


# Static actors by name are scene structure regardless of pair_type.
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
    """Node -> live simulator entity (for force queries)."""
    name = node.name
    for seg_id in node.segmentation_ids:
        ent = state.seg_id_map.get(seg_id)
        if ent is not None and getattr(ent, "name", None) == name:
            return ent
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


def _resolve_active_anchor(node: Node, state: PrivilegedState, cfg: dict):
    """``(anchor_world (3,), component, a_star)`` or ``(None, None, None)``.

    Component index is cached per node and reused across frames so the
    compatibility reference does not jump mid-rollout. The world anchor is
    re-derived each frame from the node's current pose.
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
    cache = cfg.setdefault("_affordance_selection_cache", {})
    cached = cache.get(node.node_id)
    if isinstance(cached, int) and 0 <= cached < len(comps):
        a_star = cached
    else:
        tcp_axis_local = cfg["grasp"].get(
            "tcp_approach_axis_local", [0.0, 0.0, 1.0]
        )
        orientation_weight = float(
            cfg.get("affordances", {}).get("orientation_selection_weight", 0.10)
        )
        a_star = select_active_component(
            tcp_world,
            anchors_world,
            components=comps,
            obj_pose_world=node.pose_world,
            tcp_pose_world=state.tcp_pose_world,
            tcp_axis_local=tcp_axis_local,
            orientation_weight=orientation_weight,
        )
        if a_star is not None:
            cache[node.node_id] = int(a_star)
    if a_star is None:
        return None, None, None

    node.attributes["affordance_a_star"] = int(a_star)
    return anchors_world[a_star], comps[a_star], a_star


def _compatibility_score(
    meas: CompatibilityMeasurement, norm: Dict[str, float], *, include_width: bool,
) -> float:
    """Unweighted mean of per-component [0,1] mismatches.

    Missing components (orientation without ``approach_dir``, width without
    gripper qpos) are skipped from the average rather than treated as 1.
    """
    parts: List[float] = []
    parts.append(min(meas.pos_mismatch / norm["pos"], 1.0))
    if meas.orient_mismatch is not None:
        parts.append(min(meas.orient_mismatch / norm["orient"], 1.0))
    if include_width and meas.width_mismatch is not None:
        parts.append(min(meas.width_mismatch / norm["width"], 1.0))
    if not parts:
        return 1.0
    return float(np.mean(parts))


# --------------------------------------------------------------------------- #
# Edge builders
# --------------------------------------------------------------------------- #
def ee_object_spatial_event_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """Object-center spatial relations plus contact / grasp events for every
    object node. Spatial is always ee->object-center regardless of pair_type."""
    ee = graph.get_node("ee")
    if ee is None or ee.pose_world is None:
        return []
    ee_xyz = np.asarray(ee.pose_world[:3], dtype=float)
    eps_contact = cfg["contact"]["eps_force"]
    grasp_angle = cfg["grasp"]["max_angle"]

    pd_spec = _get_bin_spec(cfg, "planar-distance")
    ho_spec = _get_bin_spec(cfg, "height-offset")

    edges: List[Edge] = []
    for node in graph.nodes:
        if node.node_type != "object":
            continue
        if not node.valid_mask or node.frozen_pose:
            continue
        ent = _resolve_entity(node, state)
        obj_xyz = _xyz(node)

        if obj_xyz is not None and pd_spec is not None:
            d = planar_distance_xyz(ee_xyz, obj_xyz)
            edges.append(Edge(
                "ee", node.node_id, "planar-distance",
                bin_label(d, pd_spec[0], pd_spec[1]), d,
            ))
        if obj_xyz is not None and ho_spec is not None:
            dz = height_offset_xyz(ee_xyz, obj_xyz)
            edges.append(Edge(
                "ee", node.node_id, "height-offset",
                bin_label(dz, ho_spec[0], ho_spec[1]), dz,
            ))

        grasped = False
        if _graspable(node):
            grasped = state.is_grasping(ent, max_angle=grasp_angle)

        force = state.ee_object_contact_force(ent)
        contact_attrs: dict = {"suppressed_by_grasp": True} if grasped else {}
        edges.append(Edge(
            "ee", node.node_id, "contact",
            "contact" if force > eps_contact else "no-contact",
            force, masked=(force <= eps_contact) or grasped,
            attributes=contact_attrs,
        ))

        if _graspable(node):
            edges.append(Edge(
                "ee", node.node_id, "grasp",
                "grasp" if grasped else "no-grasp",
                1.0 if grasped else 0.0, masked=(not grasped),
            ))
    return edges


def ee_object_compatibility_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """Affordance compatibility edges, gated by near + whitelist + grasp.

    Per object:
      * Skip unless the object has mined affordance components.
      * Compute planar-distance; skip unless the label resolves to the closest
        bin (the first label of ``SPATIAL_LABELS['planar-distance']``).
      * Score against the active affordance component.
      * Emit ``grasp-compatibility`` only when the whitelist says this object
        was grasped during demonstrations.
      * Emit ``contact-compatibility`` only when the whitelist says this
        object was contacted by an ee link; masked while currently grasped.
    """
    ee = graph.get_node("ee")
    if ee is None or ee.pose_world is None:
        return []
    if state.tcp_pose_world is None:
        return []
    ee_xyz = np.asarray(ee.pose_world[:3], dtype=float)
    pd_spec = _get_bin_spec(cfg, "planar-distance")
    if pd_spec is None:
        return []
    grasp_spec = _get_bin_spec(cfg, "grasp-compatibility")
    contact_spec = _get_bin_spec(cfg, "contact-compatibility")
    if grasp_spec is None and contact_spec is None:
        return []
    near_label = SPATIAL_LABELS["planar-distance"][0]
    norm = _compat_norm(cfg)
    tcp_axis_local = cfg["grasp"].get("tcp_approach_axis_local", [0.0, 0.0, 1.0])
    grasp_angle = cfg["grasp"]["max_angle"]
    gripper_width = getattr(state, "gripper_width", None)

    edges: List[Edge] = []
    for node in graph.nodes:
        if node.node_type != "object" or not node.valid_mask or node.frozen_pose:
            continue
        obj_xyz = _xyz(node)
        if obj_xyz is None:
            continue
        d = planar_distance_xyz(ee_xyz, obj_xyz)
        if bin_label(d, pd_spec[0], pd_spec[1]) != near_label:
            continue

        anchor_world, comp, a_star = _resolve_active_anchor(node, state, cfg)
        if anchor_world is None or comp is None or a_star is None:
            continue

        wl_key = node.attributes.get("whitelist_key")
        wl_types = _interaction_types(cfg, wl_key)
        emit_grasp = grasp_spec is not None and "grasp" in wl_types
        emit_contact = contact_spec is not None and "contact" in wl_types
        if not emit_grasp and not emit_contact:
            continue

        meas = compatibility_components(
            comp, a_star, anchor_world,
            obj_pose_world=node.pose_world,
            tcp_pose_world=state.tcp_pose_world,
            tcp_axis_local=tcp_axis_local,
            gripper_width=gripper_width,
        )

        currently_grasped = False
        if emit_contact and _graspable(node):
            ent = _resolve_entity(node, state)
            currently_grasped = state.is_grasping(ent, max_angle=grasp_angle)

        if emit_grasp:
            score = _compatibility_score(meas, norm, include_width=True)
            edges.append(Edge(
                "ee", node.node_id, "grasp-compatibility",
                bin_label(score, grasp_spec[0], grasp_spec[1]), score,
            ))
        if emit_contact:
            score = _compatibility_score(meas, norm, include_width=False)
            attrs: dict = {"suppressed_by_grasp": True} if currently_grasped else {}
            edges.append(Edge(
                "ee", node.node_id, "contact-compatibility",
                bin_label(score, contact_spec[0], contact_spec[1]),
                score, masked=currently_grasped, attributes=attrs,
            ))
    return edges


def object_object_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """One of ``support`` (supporter -> supported), ``contact``, or nothing per pair.
    Evaluated only on fresh nodes; frozen nodes get cached edges in ``GraphBuilder``."""
    eps_contact = cfg["contact"]["eps_force"]
    eps_z = cfg["support"]["eps_z"]
    min_vertical_ratio = cfg["support"].get("min_vertical_force_ratio", 0.5)

    objs = [
        n for n in graph.nodes
        if n.node_type == "object"
        and n.valid_mask
        and n.segmentation_ids
        and not n.frozen_pose
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
            if not in_contact:
                continue

            # Support = vertical-dominated force AND supporter below supported.
            support_pair = None
            if force > 0.0:
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
                    attributes={"support_role": "supporter"},
                ))
            else:
                edges.append(Edge(
                    a.node_id, b.node_id, "contact", "contact", force,
                ))
    return edges


def build_absolute_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> None:
    """Append absolute edges to ``graph.edges`` in place."""
    graph.edges.extend(ee_object_spatial_event_edges(graph, state, cfg))
    graph.edges.extend(ee_object_compatibility_edges(graph, state, cfg))
    graph.edges.extend(object_object_edges(graph, state, cfg))
