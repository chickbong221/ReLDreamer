"""Tests for the affordance asset and the eligibility-based vocabulary."""

import json
import os
import tempfile
import unittest

import numpy as np

from teemo_sim_probe.core.affordance import (
    AffordanceComponent,
    AffordanceSet,
    canonical_affordance_key,
    has_affordance,
    load_affordance_set,
    lookup_components,
    select_active_component,
    transform_anchors,
    transform_approach_dir,
)
from teemo_sim_probe.core.persistence import _snapshot
from teemo_sim_probe.core.relation_rules import (
    ee_interactive_object_edges,
)
from teemo_sim_probe.core.schema import Edge, Graph, Node
from teemo_sim_probe.core.temporal_buffer import TemporalBuffer


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class _State:
    """Minimal stand-in for PrivilegedState used by edge builders."""

    def __init__(self, tcp_xyz, gripper_width=None, tcp_quat=(1.0, 0.0, 0.0, 0.0)):
        self.tcp_pose_world = np.array([*tcp_xyz, *tcp_quat], dtype=float)
        self.gripper_width = gripper_width
        self.seg_id_map = {}
        self.active_obj = None
        self.active_handle_link = None

    def ee_object_contact_force(self, _ent):
        return 0.0

    def is_grasping(self, _ent, max_angle=30):
        return False


def _cfg(aff_set=None):
    return {
        "affordance_set": aff_set if aff_set is not None else AffordanceSet(),
        "contact": {"eps_force": 0.05},
        "grasp": {"max_angle": 30, "tcp_approach_axis_local": [0.0, 0.0, 1.0]},
        "profile": {
            "planar_distance": {
                "edges": [0.03, 0.08, 0.20, 0.50],
                "labels": ["very-near", "near", "medium", "far", "very-far"],
            },
            "height_offset": {
                "edges": [-0.10, -0.03, 0.03, 0.10],
                "labels": ["far-below", "below", "level", "above", "far-above"],
            },
            "orientation_alignment": {
                "edges": [0.17, 0.52, 1.05, 1.57],
                "labels": [
                    "aligned", "near-aligned", "oblique",
                    "perpendicular", "opposed",
                ],
            },
            "gripper_width_alignment": {
                "edges": [-0.03, -0.01, 0.01, 0.03],
                "labels": [
                    "much-tighter", "tighter", "matched", "looser", "much-looser",
                ],
            },
            "orientation_alignment_change": {
                "edges": [-0.35, -0.12, -0.03, 0.03, 0.12, 0.35],
                "labels": [
                    "align-fast", "align-medium", "align-slow",
                    "stable-orientation",
                    "misalign-slow", "misalign-medium", "misalign-fast",
                ],
            },
            "gripper_width_alignment_change": {
                "edges": [-0.02, -0.008, -0.002, 0.002, 0.008, 0.02],
                "labels": [
                    "tighten-fast", "tighten-medium", "tighten-slow",
                    "stable-width",
                    "loosen-slow", "loosen-medium", "loosen-fast",
                ],
            },
        },
    }


def _interactive_obj_node(
    name="024_bowl",
    node_id=None,
    pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    mshab_id="024_bowl",
):
    node = Node(
        node_id=node_id or f"actor:{name}",
        node_type="object",
        name=name,
        pose_world=list(pose),
        attributes={
            "is_actor": True,
            "pair_type": "interactive_object",
        },
    )
    if mshab_id:
        node.attributes["mshab_obj_id"] = mshab_id
    return node


def _ee(pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)):
    return Node(
        node_id="ee",
        node_type="ee",
        name="end_effector",
        pose_world=list(pose),
    )


# --------------------------------------------------------------------------- #
# Pure-fn tests
# --------------------------------------------------------------------------- #
class CanonicalKeyTests(unittest.TestCase):
    def test_strip_env_prefix_and_suffix(self):
        self.assertEqual(canonical_affordance_key("env-0_024_bowl-3"), "024_bowl")

    def test_strip_instance_suffix(self):
        self.assertEqual(canonical_affordance_key("024_bowl-3"), "024_bowl")

    def test_already_canonical(self):
        self.assertEqual(canonical_affordance_key("024_bowl"), "024_bowl")

    def test_preserves_compound_name(self):
        self.assertEqual(canonical_affordance_key("008_pudding_box"), "008_pudding_box")
        self.assertEqual(
            canonical_affordance_key("env-2_002_master_chef_can-1"),
            "002_master_chef_can",
        )

    def test_none_or_empty(self):
        self.assertIsNone(canonical_affordance_key(None))
        self.assertIsNone(canonical_affordance_key(""))


class TransformAnchorsTests(unittest.TestCase):
    def test_identity_pose(self):
        comps = [AffordanceComponent(np.array([0.03, 0.0, 0.01]), 0.045)]
        out = transform_anchors([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], comps)
        self.assertIsNotNone(out)
        np.testing.assert_allclose(out[0], [0.03, 0.0, 0.01], atol=1e-9)

    def test_pure_translation(self):
        comps = [AffordanceComponent(np.array([0.03, 0.0, 0.01]), 0.045)]
        out = transform_anchors([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], comps)
        np.testing.assert_allclose(out[0], [1.03, 2.0, 3.01], atol=1e-9)

    def test_90_deg_z_rotation(self):
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        comps = [AffordanceComponent(np.array([0.1, 0.0, 0.0]), 0.045)]
        out = transform_anchors([0.0, 0.0, 0.0, c, 0.0, 0.0, s], comps)
        # +x in OBJECT frame becomes +y in world after 90deg-Z rotation.
        np.testing.assert_allclose(out[0], [0.0, 0.1, 0.0], atol=1e-9)

    def test_degenerate(self):
        comps = [AffordanceComponent(np.array([0.0, 0.0, 0.0]), 0.045)]
        self.assertIsNone(transform_anchors(None, comps))
        self.assertIsNone(transform_anchors([0, 0, 0, 0, 0, 0, 0], comps))


class TransformApproachDirTests(unittest.TestCase):
    def test_identity_passthrough(self):
        c = AffordanceComponent(
            anchor_obj_frame=np.array([0.0, 0.0, 0.0]),
            preferred_width=0.045,
            approach_dir_obj_frame=np.array([0.0, 0.0, 1.0]),
        )
        d = transform_approach_dir([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], c)
        np.testing.assert_allclose(d, [0.0, 0.0, 1.0], atol=1e-9)

    def test_rotated(self):
        # 90deg around y rotates +Z(obj) to +X(world).
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        comp = AffordanceComponent(
            anchor_obj_frame=np.array([0.0, 0.0, 0.0]),
            preferred_width=0.045,
            approach_dir_obj_frame=np.array([0.0, 0.0, 1.0]),
        )
        d = transform_approach_dir([0.0, 0.0, 0.0, c, 0.0, s, 0.0], comp)
        np.testing.assert_allclose(d, [1.0, 0.0, 0.0], atol=1e-9)

    def test_missing_direction(self):
        comp = AffordanceComponent(
            anchor_obj_frame=np.array([0.0, 0.0, 0.0]),
            preferred_width=0.045,
        )
        self.assertIsNone(
            transform_approach_dir([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], comp)
        )


class SelectActiveComponentTests(unittest.TestCase):
    def test_picks_nearest(self):
        anchors = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        self.assertEqual(
            select_active_component(np.array([0.4, 0.0, 0.0]), anchors), 2
        )

    def test_empty(self):
        self.assertIsNone(select_active_component(np.array([0, 0, 0]), None))
        self.assertIsNone(
            select_active_component(np.array([0, 0, 0]), np.zeros((0, 3)))
        )


class LoadAffordanceSetTests(unittest.TestCase):
    def test_missing_file_is_empty(self):
        s = load_affordance_set("/definitely/not/a/file.json")
        self.assertTrue(s.is_empty())

    def test_basic_load_v1_compat(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "aff.json")
            with open(path, "w") as f:
                json.dump({
                    "_README": "x",
                    "_schema_version": 1,
                    "objects": {
                        "024_bowl": {"components": [
                            {"anchor": [0.0, 0.0, 0.02], "width": 0.045},
                            {"anchor": [0.03, 0.0, 0.01], "width": 0.050},
                        ]},
                        "_meta": "ignored",
                    },
                }, f)
            s = load_affordance_set(path)
        self.assertIn("024_bowl", s.by_object)
        self.assertNotIn("_meta", s.by_object)
        self.assertEqual(len(s.by_object["024_bowl"]), 2)
        # v1 entries have no approach_dir -> orientation skipped at runtime.
        self.assertIsNone(s.by_object["024_bowl"][0].approach_dir_obj_frame)

    def test_load_v2_with_approach_dir(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "aff.json")
            with open(path, "w") as f:
                json.dump({
                    "_schema_version": 2,
                    "objects": {
                        "024_bowl": {"components": [
                            {"anchor": [0.0, 0.0, 0.02],
                             "approach_dir": [0.0, 0.0, 1.0],
                             "width": 0.045},
                        ]},
                    },
                }, f)
            s = load_affordance_set(path)
        comp = s.by_object["024_bowl"][0]
        np.testing.assert_allclose(comp.approach_dir_obj_frame, [0.0, 0.0, 1.0])

    def test_skips_bad_entries(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "aff.json")
            with open(path, "w") as f:
                json.dump({"objects": {
                    "024_bowl": {"components": [
                        {"anchor": [0.0, 0.0, 0.02], "width": 0.045},
                        {"anchor": [0.0, 0.0], "width": 0.045},
                        {"width": 0.045},
                        {"anchor": [0.0, 0.0, 0.02]},
                    ]},
                }}, f)
            s = load_affordance_set(path)
        self.assertEqual(len(s.by_object["024_bowl"]), 1)


class LookupComponentsTests(unittest.TestCase):
    def _set(self):
        return AffordanceSet(by_object={
            "024_bowl": [AffordanceComponent(np.array([0.0, 0.0, 0.02]), 0.045)],
        })

    def test_prefers_mshab_obj_id(self):
        s = self._set()
        node = _interactive_obj_node(name="env-0_024_bowl-3", mshab_id="024_bowl")
        self.assertIsNotNone(lookup_components(s, node))

    def test_canonicalizes_mshab_obj_id(self):
        s = self._set()
        node = _interactive_obj_node(name="random", mshab_id="env-0_024_bowl-3")
        self.assertIsNotNone(lookup_components(s, node))

    def test_falls_back_to_name(self):
        s = self._set()
        node = _interactive_obj_node(name="env-0_024_bowl-3", mshab_id=None)
        node.attributes.pop("mshab_obj_id", None)
        self.assertIsNotNone(lookup_components(s, node))

    def test_missing(self):
        s = self._set()
        node = _interactive_obj_node(name="999_phantom", mshab_id="999_phantom")
        self.assertIsNone(lookup_components(s, node))


class HasAffordanceTests(unittest.TestCase):
    def test_true_when_set(self):
        s = AffordanceSet(by_object={
            "024_bowl": [AffordanceComponent(np.array([0.0, 0.0, 0.02]), 0.045)],
        })
        node = _interactive_obj_node()
        self.assertTrue(has_affordance(s, node))

    def test_false_when_empty(self):
        node = _interactive_obj_node()
        self.assertFalse(has_affordance(AffordanceSet(), node))


# --------------------------------------------------------------------------- #
# Interactive-object edge emission
# --------------------------------------------------------------------------- #
class InteractiveEdgeTests(unittest.TestCase):
    def _aff_set(self, with_dir=True):
        approach = np.array([0.0, 0.0, 1.0]) if with_dir else None
        return AffordanceSet(by_object={
            "024_bowl": [
                AffordanceComponent(
                    anchor_obj_frame=np.array([0.0, 0.0, 0.02]),
                    preferred_width=0.045,
                    approach_dir_obj_frame=approach,
                ),
                AffordanceComponent(
                    anchor_obj_frame=np.array([0.05, 0.0, 0.01]),
                    preferred_width=0.050,
                    approach_dir_obj_frame=approach,
                ),
            ]
        })

    def test_records_a_star_on_node(self):
        cfg = _cfg(self._aff_set())
        state = _State([0.05, 0.0, 0.01], gripper_width=0.050)
        node = _interactive_obj_node()
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        ee_interactive_object_edges(graph, state, cfg)
        self.assertEqual(node.attributes.get("affordance_a_star"), 1)

    def test_emits_anchor_based_spatial_and_width(self):
        cfg = _cfg(self._aff_set())
        state = _State([0.0, 0.0, 0.02], gripper_width=0.020)  # near anchor 0
        node = _interactive_obj_node()
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        edges = ee_interactive_object_edges(graph, state, cfg)
        rels = sorted({e.relation for e in edges})
        # contact is always emitted; grasp emitted because is_actor=True.
        self.assertIn("planar-distance", rels)
        self.assertIn("height-offset", rels)
        self.assertIn("gripper-width-alignment", rels)
        # Width error sign: 0.020 - 0.045 < 0 -> tighter / much-tighter.
        width_edge = [e for e in edges if e.relation == "gripper-width-alignment"][0]
        self.assertLess(width_edge.raw_value, 0.0)

    def test_orientation_alignment_when_direction_present(self):
        cfg = _cfg(self._aff_set(with_dir=True))
        # TCP approach axis (+Z by default) aligned with object +Z (no rotation)
        # so the angle should be 0 -> "aligned".
        state = _State([0.0, 0.0, 0.02], gripper_width=0.045)
        node = _interactive_obj_node()
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        edges = ee_interactive_object_edges(graph, state, cfg)
        align = [e for e in edges if e.relation == "orientation-alignment"]
        self.assertEqual(len(align), 1)
        self.assertEqual(align[0].label, "aligned")
        self.assertAlmostEqual(align[0].raw_value, 0.0, places=6)

    def test_no_orientation_when_direction_missing(self):
        cfg = _cfg(self._aff_set(with_dir=False))
        state = _State([0.0, 0.0, 0.02], gripper_width=0.045)
        node = _interactive_obj_node()
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        edges = ee_interactive_object_edges(graph, state, cfg)
        self.assertFalse(
            any(e.relation == "orientation-alignment" for e in edges)
        )

    def test_falls_back_to_center_without_asset(self):
        cfg = _cfg(AffordanceSet())
        state = _State([0.0, 0.0, 0.0], gripper_width=0.045)
        node = _interactive_obj_node(pose=(0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        edges = ee_interactive_object_edges(graph, state, cfg)
        # No asset -> no a_star recorded.
        self.assertNotIn("affordance_a_star", node.attributes)
        # Spatial edges still emitted relative to object center.
        rels = {e.relation for e in edges}
        self.assertIn("planar-distance", rels)
        self.assertIn("height-offset", rels)

    def test_pose_invariant_reuse(self):
        cfg = _cfg(self._aff_set())
        # Object at (1, 2, 3); TCP at (1.05, 2.0, 3.01) -> on the second anchor.
        bowl = _interactive_obj_node(pose=(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0))
        state = _State([1.05, 2.0, 3.01], gripper_width=0.050)
        graph = Graph(0, "env", "cam", nodes=[_ee(), bowl])
        edges = ee_interactive_object_edges(graph, state, cfg)
        d = [e for e in edges if e.relation == "planar-distance"][0]
        self.assertLess(abs(d.raw_value), 1e-6)

    def test_static_node_is_ignored(self):
        cfg = _cfg(self._aff_set())
        state = _State([0.0, 0.0, 0.0], gripper_width=0.045)
        node = _interactive_obj_node()
        node.attributes["pair_type"] = "static_object"
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        self.assertEqual(ee_interactive_object_edges(graph, state, cfg), [])


# --------------------------------------------------------------------------- #
# Persistence stripping
# --------------------------------------------------------------------------- #
class PersistenceStrippingTests(unittest.TestCase):
    def test_snapshot_strips_dynamic_mshab_attrs(self):
        n = _interactive_obj_node()
        n.attributes["is_mshab_active_target"] = True
        n.attributes["mshab_kind"] = "obj"
        n.attributes["affordance_a_star"] = 1
        snap = _snapshot(n)
        for k in ("is_mshab_active_target", "mshab_kind", "mshab_obj_id",
                  "affordance_a_star"):
            self.assertNotIn(k, snap.attributes)
        # Non-dynamic attrs are preserved.
        self.assertIn("is_actor", snap.attributes)


# --------------------------------------------------------------------------- #
# Temporal anchor-bound reset
# --------------------------------------------------------------------------- #
class TemporalAnchorBoundResetTests(unittest.TestCase):
    KEY = ("ee", "actor:024_bowl", "orientation-alignment")

    def _push(self, buf, frame, value, a_star=0):
        ee = _ee()
        node = _interactive_obj_node()
        if a_star is not None:
            node.attributes["affordance_a_star"] = a_star
        graph = Graph(frame, "env", "cam", nodes=[ee, node], edges=[
            Edge("ee", node.node_id, "orientation-alignment",
                 "near-aligned", float(value)),
        ])
        buf.update(graph)

    def test_history_appends_when_a_star_stable(self):
        buf = TemporalBuffer(K=3)
        self._push(buf, 0, 0.10, a_star=0)
        self._push(buf, 1, 0.08, a_star=0)
        self._push(buf, 2, 0.05, a_star=0)
        self.assertEqual(len(buf._values[self.KEY]), 3)

    def test_history_resets_on_a_star_switch(self):
        buf = TemporalBuffer(K=3)
        self._push(buf, 0, 0.10, a_star=0)
        self._push(buf, 1, 0.08, a_star=0)
        self._push(buf, 2, 0.05, a_star=1)  # switch
        self.assertEqual(len(buf._values[self.KEY]), 1)
        self.assertAlmostEqual(buf._values[self.KEY][0], 0.05)

    def test_history_drops_on_edge_disappearance(self):
        buf = TemporalBuffer(K=3)
        self._push(buf, 0, 0.10, a_star=0)
        self._push(buf, 1, 0.08, a_star=0)
        # Frame 2: same node, no orientation edge -> anchor-bound key drops.
        ee = _ee()
        node = _interactive_obj_node()
        graph = Graph(2, "env", "cam", nodes=[ee, node], edges=[])
        buf.update(graph)
        self.assertNotIn(self.KEY, buf._values)


if __name__ == "__main__":
    unittest.main()
