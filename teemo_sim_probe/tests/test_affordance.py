"""Tests for the affordance asset and the new relation vocabulary."""

import json
import os
import tempfile
import unittest

import numpy as np

from teemo_sim_probe.core.affordance import (
    AffordanceComponent,
    AffordanceSet,
    canonical_affordance_key,
    compatibility_components,
    has_affordance,
    load_affordance_set,
    lookup_components,
    select_active_component,
    transform_anchors,
    transform_approach_dir,
)
from teemo_sim_probe.core.persistence import _snapshot
from teemo_sim_probe.core.relation_rules import (
    ee_object_compatibility_edges,
    ee_object_spatial_event_edges,
)
from teemo_sim_probe.core.schema import Edge, Graph, Node
from teemo_sim_probe.core.temporal_buffer import TemporalBuffer


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class _State:
    """Minimal stand-in for PrivilegedState used by edge builders."""

    def __init__(self, tcp_xyz, gripper_width=None, tcp_quat=(1.0, 0.0, 0.0, 0.0),
                 grasping=False, contact_force=0.0):
        self.tcp_pose_world = np.array([*tcp_xyz, *tcp_quat], dtype=float)
        self.gripper_width = gripper_width
        self.seg_id_map = {}
        self.active_obj = None
        self.active_handle_link = None
        self._grasping = bool(grasping)
        self._contact_force = float(contact_force)

    def ee_object_contact_force(self, _ent):
        return self._contact_force

    def is_grasping(self, _ent, max_angle=30):
        return self._grasping


def _cfg(aff_set=None, *, interaction_types=None, bin_edges=None):
    bins = bin_edges or {
        "planar-distance": [0.05, 0.10, 0.20, 0.40],
        "height-offset": [-0.20, -0.10, 0.10, 0.20],
        "grasp-compatibility": [1.0 / 3.0, 2.0 / 3.0],
        "contact-compatibility": [1.0 / 3.0, 2.0 / 3.0],
        "planar-distance-change": [-0.06, -0.01, 0.01, 0.06],
        "height-offset-change": [-0.06, -0.01, 0.01, 0.06],
        "grasp-compatibility-change": [-0.30, -0.05, 0.05, 0.30],
        "contact-compatibility-change": [-0.30, -0.05, 0.05, 0.30],
    }
    return {
        "affordance_set": aff_set if aff_set is not None else AffordanceSet(),
        "contact": {"eps_force": 0.05},
        "grasp": {"max_angle": 30, "tcp_approach_axis_local": [0.0, 0.0, 1.0]},
        "bin_edges": bins,
        "interaction_types": dict(interaction_types or {}),
        "compat_norm": {"pos": 0.10, "orient": 1.5707963267948966, "width": 0.04},
        "profile": {},
    }


def _obj_node(
    name="024_bowl",
    node_id=None,
    pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    mshab_id="024_bowl",
    whitelist_key=None,
):
    attrs = {"is_actor": True, "pair_type": "interactive_object"}
    if whitelist_key is not None:
        attrs["whitelist_key"] = whitelist_key
    node = Node(
        node_id=node_id or f"actor:{name}",
        node_type="object",
        name=name,
        pose_world=list(pose),
        attributes=attrs,
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
# Pure-fn tests (unchanged surface)
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


class LookupComponentsTests(unittest.TestCase):
    def _set(self):
        return AffordanceSet(by_object={
            "024_bowl": [AffordanceComponent(np.array([0.0, 0.0, 0.02]), 0.045)],
        })

    def test_prefers_mshab_obj_id(self):
        s = self._set()
        node = _obj_node(name="env-0_024_bowl-3", mshab_id="024_bowl")
        self.assertIsNotNone(lookup_components(s, node))

    def test_missing(self):
        s = self._set()
        node = _obj_node(name="999_phantom", mshab_id="999_phantom")
        self.assertIsNone(lookup_components(s, node))


class HasAffordanceTests(unittest.TestCase):
    def test_true_when_set(self):
        s = AffordanceSet(by_object={
            "024_bowl": [AffordanceComponent(np.array([0.0, 0.0, 0.02]), 0.045)],
        })
        self.assertTrue(has_affordance(s, _obj_node()))

    def test_false_when_empty(self):
        self.assertFalse(has_affordance(AffordanceSet(), _obj_node()))


# --------------------------------------------------------------------------- #
# Compatibility helper
# --------------------------------------------------------------------------- #
class CompatibilityComponentsTests(unittest.TestCase):
    def test_zero_mismatch_at_perfect_match(self):
        comp = AffordanceComponent(
            anchor_obj_frame=np.array([0.0, 0.0, 0.02]),
            preferred_width=0.045,
            approach_dir_obj_frame=np.array([0.0, 0.0, 1.0]),
        )
        anchor_world = np.array([0.0, 0.0, 0.02])
        meas = compatibility_components(
            comp, 0, anchor_world,
            obj_pose_world=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            tcp_pose_world=np.array([0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]),
            tcp_axis_local=[0.0, 0.0, 1.0],
            gripper_width=0.045,
        )
        self.assertAlmostEqual(meas.pos_mismatch, 0.0, places=6)
        self.assertAlmostEqual(meas.orient_mismatch, 0.0, places=6)
        self.assertAlmostEqual(meas.width_mismatch, 0.0, places=6)

    def test_width_none_when_gripper_width_missing(self):
        comp = AffordanceComponent(
            anchor_obj_frame=np.array([0.0, 0.0, 0.02]),
            preferred_width=0.045,
        )
        meas = compatibility_components(
            comp, 0, np.array([0.0, 0.0, 0.02]),
            obj_pose_world=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            tcp_pose_world=np.array([0.0, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]),
            tcp_axis_local=[0.0, 0.0, 1.0],
            gripper_width=None,
        )
        self.assertIsNone(meas.width_mismatch)


# --------------------------------------------------------------------------- #
# Spatial / event edges (replaces previous interactive/static split)
# --------------------------------------------------------------------------- #
class SpatialEventEdgeTests(unittest.TestCase):
    def test_object_center_spatial_for_every_object(self):
        cfg = _cfg()
        # Object at (0.10, 0, 0); ee at origin -> planar-distance 0.10 ("medium").
        node = _obj_node(pose=(0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        edges = ee_object_spatial_event_edges(graph, _State([0.0, 0.0, 0.0]), cfg)
        rels = {(e.relation, e.label) for e in edges}
        self.assertIn(("planar-distance", "medium"), rels)
        self.assertIn(("height-offset", "level"), rels)

    def test_grasp_suppresses_contact(self):
        cfg = _cfg()
        node = _obj_node()
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        edges = ee_object_spatial_event_edges(
            graph, _State([0.0, 0.0, 0.0], grasping=True, contact_force=2.0), cfg,
        )
        contact = [e for e in edges if e.relation == "contact"]
        grasp = [e for e in edges if e.relation == "grasp"]
        # One physical-state edge per pair: grasp wins, contact is dropped.
        self.assertEqual(contact, [])
        self.assertEqual(len(grasp), 1)
        self.assertFalse(grasp[0].masked)


# --------------------------------------------------------------------------- #
# Compatibility edge gating
# --------------------------------------------------------------------------- #
class CompatibilityEdgeTests(unittest.TestCase):
    def _aff_set(self):
        return AffordanceSet(by_object={
            "024_bowl": [
                AffordanceComponent(
                    anchor_obj_frame=np.array([0.0, 0.0, 0.02]),
                    preferred_width=0.045,
                    approach_dir_obj_frame=np.array([0.0, 0.0, 1.0]),
                ),
            ]
        })

    def test_emits_grasp_compat_when_whitelist_grasps(self):
        cfg = _cfg(
            self._aff_set(),
            interaction_types={"actor:024_bowl": {"contact", "grasp"}},
        )
        node = _obj_node(whitelist_key="actor:024_bowl",
                         pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        state = _State([0.0, 0.0, 0.02], gripper_width=0.045)
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        edges = ee_object_compatibility_edges(graph, state, cfg)
        rels = sorted(e.relation for e in edges)
        self.assertEqual(rels, ["contact-compatibility", "grasp-compatibility"])
        # Perfect alignment at the anchor -> normalized score 0 -> "match".
        for e in edges:
            self.assertEqual(e.label, "match")

    def test_skips_grasp_compat_without_whitelist_grasp(self):
        cfg = _cfg(
            self._aff_set(),
            interaction_types={"actor:024_bowl": {"contact"}},  # contact only
        )
        node = _obj_node(whitelist_key="actor:024_bowl")
        state = _State([0.0, 0.0, 0.02], gripper_width=0.045)
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        edges = ee_object_compatibility_edges(graph, state, cfg)
        rels = {e.relation for e in edges}
        self.assertNotIn("grasp-compatibility", rels)
        self.assertIn("contact-compatibility", rels)

    def test_skips_when_planar_distance_not_near(self):
        cfg = _cfg(
            self._aff_set(),
            interaction_types={"actor:024_bowl": {"contact", "grasp"}},
        )
        # Far enough away that planar-distance label is neither "very-near" nor "near".
        node = _obj_node(whitelist_key="actor:024_bowl",
                         pose=(0.30, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        state = _State([0.0, 0.0, 0.02], gripper_width=0.045)
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        self.assertEqual(ee_object_compatibility_edges(graph, state, cfg), [])

    def test_contact_compat_masked_under_grasp(self):
        cfg = _cfg(
            self._aff_set(),
            interaction_types={"actor:024_bowl": {"contact", "grasp"}},
        )
        node = _obj_node(whitelist_key="actor:024_bowl")
        state = _State([0.0, 0.0, 0.02], gripper_width=0.045, grasping=True)
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        edges = ee_object_compatibility_edges(graph, state, cfg)
        contact_compat = [e for e in edges if e.relation == "contact-compatibility"]
        self.assertEqual(len(contact_compat), 1)
        self.assertTrue(contact_compat[0].masked)
        self.assertEqual(
            contact_compat[0].attributes.get("suppressed_by_grasp"), True,
        )


# --------------------------------------------------------------------------- #
# Persistence stripping
# --------------------------------------------------------------------------- #
class PersistenceStrippingTests(unittest.TestCase):
    def test_snapshot_strips_dynamic_component_only(self):
        n = _obj_node()
        n.attributes["affordance_a_star"] = 1
        snap = _snapshot(n)
        self.assertNotIn("affordance_a_star", snap.attributes)
        self.assertIn("is_actor", snap.attributes)


# --------------------------------------------------------------------------- #
# Temporal affordance history
# --------------------------------------------------------------------------- #
class TemporalAffordanceHistoryTests(unittest.TestCase):
    KEY = ("ee", "actor:024_bowl", "grasp-compatibility")

    def _push(self, buf, frame, value):
        ee = _ee()
        node = _obj_node()
        graph = Graph(frame, "env", "cam", nodes=[ee, node], edges=[
            Edge("ee", node.node_id, "grasp-compatibility",
                 "partial-match", float(value)),
        ])
        buf.update(graph)

    def test_history_appends_across_frames(self):
        buf = TemporalBuffer(K=3)
        self._push(buf, 0, 0.10)
        self._push(buf, 1, 0.08)
        self._push(buf, 2, 0.05)
        self.assertEqual(len(buf._values[self.KEY]), 3)

    def test_history_drops_on_edge_disappearance(self):
        buf = TemporalBuffer(K=3)
        self._push(buf, 0, 0.10)
        self._push(buf, 1, 0.08)
        ee = _ee()
        node = _obj_node()
        graph = Graph(2, "env", "cam", nodes=[ee, node], edges=[])
        buf.update(graph)
        self.assertNotIn(self.KEY, buf._values)


if __name__ == "__main__":
    unittest.main()
