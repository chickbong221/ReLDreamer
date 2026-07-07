"""Absolute relation labels from privileged state.

Three families, all driven off the per-(subtask, target) whitelist + the
multi-modal affordance asset:

* **Physical state** (single positive label, no paired ``no-*``):
  ``contact`` (actor--obj, obj--obj), ``grasp`` (actor--obj),
  ``support`` (obj--obj, supporter -> supported),
  ``contain`` (obj--obj, container -> containee).
  Transitions are NOT separately annotated -- they fall out of consecutive
  absolute frames.

* **Spatial** (actor--obj only): ``planar-distance``, ``height-offset``.

* **Affordance compatibility**:
    - ``grasp-compatibility``        (actor--near_obj)
    - ``contact-compatibility``      (actor--obj AND obj--obj)
    - ``support-compatibility``      (obj--near_obj)
    - ``contain-compatibility``      (obj--near_obj)
  Each compat edge is gated on (a) the relevant affordance components being
  mined for the object(s) involved, (b) the planar-distance bin being
  ``near``, and (c) the whitelist asset listing the matching interaction type.
  ``contact-compatibility`` is masked while the object is grasped, mirroring
  the contact edge mask.

Bin edges come from the active per-(subtask, target) whitelist
(``cfg["bin_edges"]``); ``cfg["profile"]`` is a fallback for any relation the
asset omits. The per-object interaction-type table lives in
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
    lookup_contact_components,
    lookup_contain_components,
    lookup_bottom_components,
    lookup_key_components,
    lookup_support_components,
    select_active_component,
    transform_anchors,
)
from .containment import (
    contain_compatibility,
    contain_holds,
    obj_contact_compatibility,
    support_compatibility,
)
from .schema import Edge, Graph, Node
from ..adapters.privileged_state import PrivilegedState


# --------------------------------------------------------------------------- #
# Canonical label vocabulary (relation -> labels, in ascending bin order)
# --------------------------------------------------------------------------- #
SPATIAL_LABELS: Dict[str, List[str]] = {
    "planar-distance": ["very-near", "near", "medium", "far", "very-far"],
    "height-offset": ["far-below", "below", "level", "above", "far-above"],
}
COMPAT_LABELS: Dict[str, List[str]] = {
    "grasp-compatibility":   ["match", "partial-match", "poor-match"],
    "contact-compatibility": ["match", "partial-match", "poor-match"],
    "support-compatibility": ["match", "partial-match", "poor-match"],
    "contain-compatibility": ["match", "partial-match", "poor-match"],
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
    "support-compatibility-change": [
        "support-fit-better-fast", "support-fit-better-slow",
        "stable-support-fit",
        "support-fit-worse-slow", "support-fit-worse-fast",
    ],
    "contain-compatibility-change": [
        "contain-fit-better-fast", "contain-fit-better-slow",
        "stable-contain-fit",
        "contain-fit-worse-slow", "contain-fit-worse-fast",
    ],
}
ALL_LABELS: Dict[str, List[str]] = {
    **SPATIAL_LABELS, **COMPAT_LABELS, **CHANGE_LABELS,
}


def _planar_near_labels() -> Set[str]:
    labels = SPATIAL_LABELS["planar-distance"]
    if len(labels) >= 5:
        return set(labels[:2])
    return {labels[0]}


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
    """Per-component normalizers (metres / radians) shared by all compat scorers.

    Defaults:
      pos      = 0.10 m   (close-approach distance)
      orient   = pi / 2   (90deg half-cone)
      width    = 0.04 m   (gripper spread)
      xy       = 0.05 m   (in-plane support offset)
      vertical = 0.03 m   (support gap/interpenetration)
      radial   = 0.02 m   (contain radial slack)
      axial    = 0.03 m   (contain axial slack)
    """
    norm = cfg.get("compat_norm") or {}
    out = {
        "pos":      float(norm.get("pos",      0.10)),
        "orient":   float(norm.get("orient",   np.pi / 2.0)),
        "width":    float(norm.get("width",    0.04)),
        "xy":       float(norm.get("xy",       0.05)),
        "vertical": float(norm.get("vertical", 0.03)),
        "radial":   float(norm.get("radial",   0.02)),
        "axial":    float(norm.get("axial",    0.03)),
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


def _mean_normalized(parts: List[float]) -> float:
    if not parts:
        return 1.0
    return float(np.mean([min(max(p, 0.0), 1.0) for p in parts]))


# --------------------------------------------------------------------------- #
# Edge builders -- ee -> object
# --------------------------------------------------------------------------- #
def ee_object_spatial_event_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """Spatial + physical-state edges for every visible object node.

    Spatial is always ee->object-center. At most one physical-state edge per
    pair: ``grasp`` takes precedence over ``contact``. When grasped, the
    ``contact`` edge is not emitted at all (the temporal buffer's contact-
    compatibility history reset is driven by the ``contact-compatibility``
    edge's own ``suppressed_by_grasp`` attribute).
    """
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
        in_contact = force > eps_contact

        # One physical-state edge per pair: grasp > contact.
        if grasped:
            edges.append(Edge(
                "ee", node.node_id, "grasp", "grasp",
                1.0, masked=False,
            ))
        elif in_contact:
            edges.append(Edge(
                "ee", node.node_id, "contact", "contact", force,
                masked=False,
            ))
    return edges


def ee_object_compatibility_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """ee--object affordance compatibility edges, near + whitelist + grasp gated.

    Per object:
      * Skip unless the object has mined grasp components.
      * Compute planar-distance; skip unless the label resolves to the closest
        bin (``near``).
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
    near_labels = _planar_near_labels()
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
        if bin_label(d, pd_spec[0], pd_spec[1]) not in near_labels:
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


# --------------------------------------------------------------------------- #
# Edge builders -- object -> object
# --------------------------------------------------------------------------- #
def _object_pairs(graph: Graph) -> List[Tuple[Node, Node]]:
    objs = [
        n for n in graph.nodes
        if n.node_type == "object"
        and n.valid_mask
        and n.segmentation_ids
        and not n.frozen_pose
    ]
    out: List[Tuple[Node, Node]] = []
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            out.append((objs[i], objs[j]))
    return out


def _pair_planar_distance(a: Node, b: Node) -> Optional[float]:
    """Center-to-center planar distance in metres, or None when either pose
    is missing. Uses the object-frame origin; conservative for the physics
    short-circuit because true contact distance is always <= center
    distance + summed extents."""
    if a.pose_world is None or b.pose_world is None:
        return None
    return float(np.linalg.norm(
        np.asarray(a.pose_world[:2], dtype=float)
        - np.asarray(b.pose_world[:2], dtype=float)
    ))


def object_object_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """At most one physical-state edge per object pair, in priority order
    ``contain > support > contact``:

      * ``contain`` (container -> containee) when the containee's mined key
        point lies inside the container's mined entry volume (geometric
        PegInsertion check). Tried in both directions; first hit wins.
      * ``support`` (supporter -> supported) when no contain edge fires AND
        the pair has vertical-dominated contact force AND the supporter sits
        below the supported (center dz).
      * ``contact`` (undirected, A -> B in iteration order) when neither of
        the above fires but the pair is in contact above ``eps_force``.
    """
    eps_contact = cfg["contact"]["eps_force"]
    min_vertical_ratio = cfg["support"].get("min_vertical_force_ratio", 0.5)
    # Skip the (expensive) GPU pair-force query when object centers are
    # farther apart than any physically possible contact distance. Two rigid
    # bodies cannot exert a contact force at planar distance greater than
    # roughly the sum of their bounding half-extents; for ManiSkill scenes
    # nothing exceeds ~1 m, so a 2 m default is byte-identical to the
    # un-gated path. Set to 0 (or negative) in cfg to disable.
    pair_force_max_distance = float(cfg.get("pair_force_max_distance", 2.0))

    aff_set = cfg.get("affordance_set")
    edges: List[Edge] = []
    for a, b in _object_pairs(graph):
        # Priority 1: geometric containment (orthogonal to force evidence).
        contain_emitted = False
        if aff_set is not None:
            for container, containee in ((a, b), (b, a)):
                container_comps = lookup_contain_components(aff_set, container)
                key_comps = lookup_key_components(aff_set, containee)
                if not container_comps or not key_comps:
                    continue
                held = False
                for cc in container_comps:
                    for kc in key_comps:
                        if contain_holds(
                            container.pose_world, cc, containee.pose_world, kc,
                        ):
                            held = True
                            break
                    if held:
                        break
                if held:
                    edges.append(Edge(
                        container.node_id, containee.node_id,
                        "contain", "contain", 1.0,
                        attributes={"contain_role": "container"},
                    ))
                    contain_emitted = True
                    break
        if contain_emitted:
            continue

        # Physics short-circuit: pairs whose centers exceed the max plausible
        # contact distance cannot produce a support / contact edge. Skipping
        # eliminates the SAPIEN GPU query without changing edge output.
        if pair_force_max_distance > 0.0:
            planar = _pair_planar_distance(a, b)
            if planar is not None and planar > pair_force_max_distance:
                continue

        # Priorities 2 and 3: force-driven support / contact.
        ea = _resolve_entity(a, state)
        eb = _resolve_entity(b, state)
        force_vector = state.pairwise_force_vector(ea, eb)
        force = float(np.linalg.norm(force_vector))
        in_contact = force > eps_contact
        if not in_contact:
            continue

        support_pair = None
        if force > 0.0:
            vertical_ratio = abs(float(force_vector[2])) / force
            if vertical_ratio >= min_vertical_ratio:
                # Direction from the contact force sign, not pose-center dz.
                # Link-frame origins are usually not at the contact surface
                # (a drawer's origin sits at the drawer front, not on its
                # top face), so ``supporter = lower-z endpoint`` flips
                # whenever a tall/thin supporter carries a short/wide
                # supported object. ManiSkill's ``get_pairwise_contact_forces``
                # returns "force on ``a`` due to ``b``" (see
                # mani_skill/envs/scene.py:789), so:
                #   fz < 0  -> b's weight pushes a down -> a is supporter
                #   fz > 0  -> reaction pushes a up     -> b is supporter
                if float(force_vector[2]) < 0.0:
                    support_pair = (a, b)
                else:
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


def _near_pair(
    a_xyz: np.ndarray, b_xyz: np.ndarray, pd_spec, near_labels: Set[str],
) -> bool:
    return bin_label(
        planar_distance_xyz(a_xyz, b_xyz), pd_spec[0], pd_spec[1],
    ) in near_labels


def object_object_compatibility_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> List[Edge]:
    """obj--near_obj compatibility edges: contact / support / contain.

    For each ordered object pair:

      * Skip unless their planar distance bins ``near``.
      * Skip unless both endpoints' whitelist ``interaction_types`` carry the
        relation token (``contact`` / ``support`` / ``contain``).
      * Skip unless the matching affordance components are mined for both
        endpoints.

    ``contact-compatibility`` is also masked when either endpoint is currently
    grasped (mirrors the ee-object mask).
    """
    pd_spec = _get_bin_spec(cfg, "planar-distance")
    if pd_spec is None:
        return []
    near_labels = _planar_near_labels()
    aff_set = cfg.get("affordance_set")
    if aff_set is None:
        return []

    aff_cfg = cfg.get("affordances", {})
    contact_spec = (
        _get_bin_spec(cfg, "contact-compatibility")
        if bool(aff_cfg.get("object_object_contact_compatibility", True))
        else None
    )
    support_enabled = bool(aff_cfg.get("object_object_support_compatibility", False))
    support_subtasks = aff_cfg.get("object_object_support_compatibility_subtasks")
    if support_enabled and support_subtasks is not None:
        active_subtask = graph.meta.get("active_subtask")
        support_enabled = str(active_subtask) in {str(s) for s in support_subtasks}
    support_spec = (
        _get_bin_spec(cfg, "support-compatibility") if support_enabled else None
    )
    contain_spec = (
        _get_bin_spec(cfg, "contain-compatibility")
        if bool(aff_cfg.get("object_object_contain_compatibility", True))
        else None
    )
    if contact_spec is None and support_spec is None and contain_spec is None:
        return []

    norm = _compat_norm(cfg)
    grasp_angle = cfg["grasp"]["max_angle"]

    edges: List[Edge] = []
    for a, b in _object_pairs(graph):
        a_xyz = _xyz(a)
        b_xyz = _xyz(b)
        if a_xyz is None or b_xyz is None:
            continue
        if not _near_pair(a_xyz, b_xyz, pd_spec, near_labels):
            continue

        a_key = a.attributes.get("whitelist_key")
        b_key = b.attributes.get("whitelist_key")
        a_types = _interaction_types(cfg, a_key)
        b_types = _interaction_types(cfg, b_key)

        # ---- contact-compatibility (obj-obj, symmetric) --------------------
        if contact_spec is not None and "contact" in a_types and "contact" in b_types:
            a_comps = lookup_contact_components(aff_set, a)
            b_comps = lookup_contact_components(aff_set, b)
            if a_comps and b_comps:
                meas = obj_contact_compatibility(
                    a.pose_world, a_comps, b.pose_world, b_comps,
                )
                if meas is not None:
                    parts: List[float] = [meas.pos_mismatch / norm["pos"]]
                    if meas.orient_mismatch is not None:
                        parts.append(meas.orient_mismatch / norm["orient"])
                    score = _mean_normalized(parts)
                    grasped_a = (
                        _graspable(a)
                        and state.is_grasping(_resolve_entity(a, state),
                                              max_angle=grasp_angle)
                    )
                    grasped_b = (
                        _graspable(b)
                        and state.is_grasping(_resolve_entity(b, state),
                                              max_angle=grasp_angle)
                    )
                    suppressed = grasped_a or grasped_b
                    attrs = {"suppressed_by_grasp": True} if suppressed else {}
                    edges.append(Edge(
                        a.node_id, b.node_id, "contact-compatibility",
                        bin_label(score, contact_spec[0], contact_spec[1]),
                        score, masked=suppressed, attributes=attrs,
                    ))

        # ---- support-compatibility (directed: supporter -> supported) ------
        if support_spec is not None and "support" in a_types and "support" in b_types:
            for supporter, supported in ((a, b), (b, a)):
                sup_comps = lookup_support_components(aff_set, supporter)
                bot_comps = lookup_bottom_components(aff_set, supported)
                if not sup_comps or not bot_comps:
                    continue
                meas = support_compatibility(
                    supporter.pose_world, sup_comps,
                    supported.pose_world, bot_comps,
                )
                if meas is None:
                    continue
                parts = [
                    meas.xy_mismatch / norm["xy"],
                    meas.vertical_mismatch / norm["vertical"],
                ]
                if meas.orient_mismatch is not None:
                    parts.append(meas.orient_mismatch / norm["orient"])
                score = _mean_normalized(parts)
                edges.append(Edge(
                    supporter.node_id, supported.node_id,
                    "support-compatibility",
                    bin_label(score, support_spec[0], support_spec[1]),
                    score,
                    attributes={"support_role": "supporter"},
                ))

        # ---- contain-compatibility (directed: container -> containee) ------
        if contain_spec is not None and "contain" in a_types and "contain" in b_types:
            for container, containee in ((a, b), (b, a)):
                con_comps = lookup_contain_components(aff_set, container)
                key_comps = lookup_key_components(aff_set, containee)
                if not con_comps or not key_comps:
                    continue
                meas = contain_compatibility(
                    container.pose_world, con_comps,
                    containee.pose_world, key_comps,
                )
                if meas is None:
                    continue
                parts = [
                    meas.radial_mismatch / norm["radial"],
                    meas.axial_mismatch / norm["axial"],
                ]
                if meas.orient_mismatch is not None:
                    parts.append(meas.orient_mismatch / norm["orient"])
                score = _mean_normalized(parts)
                edges.append(Edge(
                    container.node_id, containee.node_id,
                    "contain-compatibility",
                    bin_label(score, contain_spec[0], contain_spec[1]),
                    score,
                    attributes={"contain_role": "container"},
                ))
    return edges


def build_absolute_edges(
    graph: Graph, state: PrivilegedState, cfg: dict
) -> None:
    """Append absolute edges to ``graph.edges`` in place."""
    graph.edges.extend(ee_object_spatial_event_edges(graph, state, cfg))
    graph.edges.extend(ee_object_compatibility_edges(graph, state, cfg))
    graph.edges.extend(object_object_edges(graph, state, cfg))
    if bool(cfg.get("affordances", {}).get("object_object_compatibility", True)):
        graph.edges.extend(object_object_compatibility_edges(graph, state, cfg))
