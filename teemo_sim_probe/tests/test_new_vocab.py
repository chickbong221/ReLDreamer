"""Coverage for the post-rewrite TEEMO relation vocabulary.

* Geometric ``contain`` detection (PegInsertionSide template).
* Obj-obj ``contact-compatibility`` scorer.
* ``support-compatibility`` scorer.
* ``contain-compatibility`` scorer.
* No-no-* paired labels on physical-state edges (only positive labels emit).
* Temporal change edge for one of the new compat relations.
"""

import unittest

import numpy as np

from teemo_sim_probe.core.affordance import (
    AffordanceSet,
    BottomComponent,
    ContactComponent,
    ContainComponent,
    KeyComponent,
    SupportComponent,
)
from teemo_sim_probe.core.containment import (
    contain_compatibility,
    contain_holds,
    obj_contact_compatibility,
    support_compatibility,
)
from teemo_sim_probe.core.relation_rules import (
    ALL_LABELS,
    ee_object_spatial_event_edges,
    object_object_edges,
    object_object_compatibility_edges,
)
from teemo_sim_probe.core.schema import Edge, Graph, Node
from teemo_sim_probe.core.temporal_buffer import TemporalBuffer


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ee_node():
    return Node(
        node_id="ee", node_type="ee", name="end_effector",
        pose_world=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )


def _obj_node(node_id, pose):
    return Node(
        node_id=node_id, node_type="object", name=node_id,
        pose_world=list(pose),
        segmentation_ids=[hash(node_id) & 0xffff],
        attributes={"is_actor": True},
    )


class _StubState:
    def __init__(self, *, force_vector=(0.0, 0.0, 0.0),
                 grasping=False, contact_force=0.0,
                 tcp=(0.0, 0.0, 0.0)):
        self.force_vector = np.asarray(force_vector, dtype=float)
        self._grasping = bool(grasping)
        self._contact_force = float(contact_force)
        self.tcp_pose_world = np.array([*tcp, 1.0, 0.0, 0.0, 0.0], dtype=float)
        self.gripper_width = None
        self.seg_id_map = {}

    def pairwise_force_vector(self, _a, _b):
        return self.force_vector

    def ee_object_contact_force(self, _ent):
        return self._contact_force

    def is_grasping(self, _ent, max_angle=30):
        return self._grasping


# --------------------------------------------------------------------------- #
# Geometric contain (PegInsertion template)
# --------------------------------------------------------------------------- #
class ContainGeometricTests(unittest.TestCase):
    def _hole(self, opening_radius=0.02, depth=0.10):
        return ContainComponent(
            entry_anchor_obj_frame=np.array([0.0, 0.0, 0.0]),
            entry_axis_obj_frame=np.array([1.0, 0.0, 0.0]),
            opening_radius=opening_radius,
            depth=depth,
        )

    def _key(self):
        return KeyComponent(
            key_anchor_obj_frame=np.array([0.0, 0.0, 0.0]),
            key_axis_obj_frame=np.array([1.0, 0.0, 0.0]),
        )

    def test_key_inside_hole_returns_true(self):
        container = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        # Peg head at x=0.05 in world: 5 cm past entry along the hole axis.
        containee = [0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.assertTrue(
            contain_holds(container, self._hole(), containee, self._key()),
        )

    def test_key_radially_off_axis_returns_false(self):
        container = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        # Radially 5 cm off-axis -- well outside opening_radius=2 cm.
        containee = [0.05, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0]
        self.assertFalse(
            contain_holds(container, self._hole(), containee, self._key()),
        )

    def test_key_before_entry_returns_false(self):
        container = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        # Peg head at x=-0.01: outside the [0, depth] interval.
        containee = [-0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.assertFalse(
            contain_holds(container, self._hole(), containee, self._key()),
        )


# --------------------------------------------------------------------------- #
# Obj-obj contact compatibility
# --------------------------------------------------------------------------- #
class ObjContactCompatTests(unittest.TestCase):
    def test_perfect_match_at_aligned_anchors(self):
        a = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        b = [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        # A's contact anchor at (+0.05, 0, 0); B's at (-0.05, 0, 0); both
        # transform to (0.05, 0, 0) in world.
        a_comps = [ContactComponent(np.array([0.05, 0.0, 0.0]),
                                    np.array([1.0, 0.0, 0.0]))]
        b_comps = [ContactComponent(np.array([-0.05, 0.0, 0.0]),
                                    np.array([-1.0, 0.0, 0.0]))]
        meas = obj_contact_compatibility(a, a_comps, b, b_comps)
        self.assertIsNotNone(meas)
        self.assertAlmostEqual(meas.pos_mismatch, 0.0, places=6)
        # Anti-parallel outward normals -> 0 mismatch.
        self.assertAlmostEqual(meas.orient_mismatch, 0.0, places=6)

    def test_returns_none_when_either_side_empty(self):
        a = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        b = [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        a_comps = [ContactComponent(np.array([0.05, 0.0, 0.0]))]
        self.assertIsNone(obj_contact_compatibility(a, a_comps, b, []))
        self.assertIsNone(obj_contact_compatibility(a, [], b, a_comps))


# --------------------------------------------------------------------------- #
# Support compatibility
# --------------------------------------------------------------------------- #
class SupportCompatTests(unittest.TestCase):
    def test_centered_on_surface_zero_xy(self):
        supporter = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        supported = [0.0, 0.0, 0.04, 1.0, 0.0, 0.0, 0.0]
        sup_comps = [SupportComponent(
            surface_anchor_obj_frame=np.array([0.0, 0.0, 0.04]),
            surface_normal_obj_frame=np.array([0.0, 0.0, 1.0]),
            footprint_radius=0.05,
        )]
        bot_comps = [BottomComponent(
            bottom_anchor_obj_frame=np.array([0.0, 0.0, 0.0]),
            bottom_normal_obj_frame=np.array([0.0, 0.0, -1.0]),
        )]
        meas = support_compatibility(supporter, sup_comps, supported, bot_comps)
        self.assertIsNotNone(meas)
        self.assertAlmostEqual(meas.xy_mismatch, 0.0, places=6)
        self.assertAlmostEqual(meas.vertical_mismatch, 0.0, places=6)
        self.assertAlmostEqual(meas.orient_mismatch, 0.0, places=6)

    def test_within_footprint_clips_to_zero_xy(self):
        supporter = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        # supported's bottom anchor at world (0.02, 0.0, 0.04). Surface
        # anchor in supporter local = (0, 0, 0.04). In-plane delta = 2 cm <
        # footprint_radius=5 cm -> clipped to 0.
        supported = [0.02, 0.0, 0.04, 1.0, 0.0, 0.0, 0.0]
        sup_comps = [SupportComponent(
            surface_anchor_obj_frame=np.array([0.0, 0.0, 0.04]),
            surface_normal_obj_frame=np.array([0.0, 0.0, 1.0]),
            footprint_radius=0.05,
        )]
        bot_comps = [BottomComponent(
            bottom_anchor_obj_frame=np.array([0.0, 0.0, 0.0]),
            bottom_normal_obj_frame=np.array([0.0, 0.0, -1.0]),
        )]
        meas = support_compatibility(supporter, sup_comps, supported, bot_comps)
        self.assertIsNotNone(meas)
        self.assertAlmostEqual(meas.xy_mismatch, 0.0, places=6)


# --------------------------------------------------------------------------- #
# Runtime obj-obj compatibility gating
# --------------------------------------------------------------------------- #
class ObjectObjectCompatibilityGateTests(unittest.TestCase):
    def _cfg(self, *, support_enabled=True, support_subtasks=("place",)):
        aff_set = AffordanceSet(
            contact_by_object={
                "knife": [ContactComponent(
                    anchor_obj_frame=np.array([0.05, 0.0, 0.0]),
                    outward_normal_obj_frame=np.array([1.0, 0.0, 0.0]),
                )],
                "onion": [ContactComponent(
                    anchor_obj_frame=np.array([-0.05, 0.0, 0.0]),
                    outward_normal_obj_frame=np.array([-1.0, 0.0, 0.0]),
                )],
            },
            support_by_object={
                "link:drawer": [SupportComponent(
                    surface_anchor_obj_frame=np.array([0.0, 0.0, 0.04]),
                    surface_normal_obj_frame=np.array([0.0, 0.0, 1.0]),
                    footprint_radius=0.10,
                )],
            },
            bottom_by_object={
                "actor:bowl": [BottomComponent(
                    bottom_anchor_obj_frame=np.array([0.0, 0.0, 0.0]),
                    bottom_normal_obj_frame=np.array([0.0, 0.0, -1.0]),
                )],
            },
        )
        return {
            "contact": {"eps_force": 0.05},
            "grasp": {"max_angle": 30, "tcp_approach_axis_local": [0, 0, 1]},
            "bin_edges": {
                "planar-distance": [0.20, 0.50],
                "contact-compatibility": [1.0 / 3.0, 2.0 / 3.0],
                "support-compatibility": [1.0 / 3.0, 2.0 / 3.0],
            },
            "interaction_types": {
                "actor:knife": {"contact"},
                "actor:onion": {"contact"},
                "link:drawer": {"support"},
                "actor:bowl": {"support"},
            },
            "affordance_set": aff_set,
            "affordances": {
                "object_object_contact_compatibility": True,
                "object_object_support_compatibility": support_enabled,
                "object_object_support_compatibility_subtasks": list(support_subtasks),
            },
        }

    def test_contact_compat_stays_enabled_for_tool_object_pairs(self):
        knife = _obj_node("actor:knife", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        onion = _obj_node("actor:onion", (0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        knife.attributes.update({"entity_key": "actor:knife", "whitelist_key": "actor:knife"})
        onion.attributes.update({"entity_key": "actor:onion", "whitelist_key": "actor:onion"})
        graph = Graph(0, "env", "cam", nodes=[knife, onion])

        edges = object_object_compatibility_edges(graph, _StubState(), self._cfg())

        self.assertIn("contact-compatibility", {e.relation for e in edges})

    def test_support_compat_is_off_by_default_for_passive_support_pairs(self):
        drawer = _obj_node("link:drawer", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        bowl = _obj_node("actor:bowl", (0.0, 0.0, 0.04, 1.0, 0.0, 0.0, 0.0))
        drawer.attributes.update({"entity_key": "link:drawer", "whitelist_key": "link:drawer"})
        bowl.attributes.update({"entity_key": "actor:bowl", "whitelist_key": "actor:bowl"})
        graph = Graph(
            0, "env", "cam", nodes=[drawer, bowl],
            meta={"active_subtask": "pick"},
        )

        edges = object_object_compatibility_edges(graph, _StubState(), self._cfg())

        self.assertNotIn("support-compatibility", {e.relation for e in edges})

    def test_support_compat_is_enabled_for_place_when_allowlisted(self):
        drawer = _obj_node("link:drawer", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        bowl = _obj_node("actor:bowl", (0.0, 0.0, 0.04, 1.0, 0.0, 0.0, 0.0))
        drawer.attributes.update({"entity_key": "link:drawer", "whitelist_key": "link:drawer"})
        bowl.attributes.update({"entity_key": "actor:bowl", "whitelist_key": "actor:bowl"})
        graph = Graph(
            0, "env", "cam", nodes=[drawer, bowl],
            meta={"active_subtask": "place"},
        )

        edges = object_object_compatibility_edges(graph, _StubState(), self._cfg())

        self.assertIn("support-compatibility", {e.relation for e in edges})


# --------------------------------------------------------------------------- #
# Contain compatibility
# --------------------------------------------------------------------------- #
class ContainCompatTests(unittest.TestCase):
    def test_inside_hole_zero_mismatches(self):
        container = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        # Peg head at x=0.04 inside [0, depth=0.10].
        containee = [0.04, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        con_comps = [ContainComponent(
            entry_anchor_obj_frame=np.array([0.0, 0.0, 0.0]),
            entry_axis_obj_frame=np.array([1.0, 0.0, 0.0]),
            opening_radius=0.02,
            depth=0.10,
        )]
        key_comps = [KeyComponent(
            key_anchor_obj_frame=np.array([0.0, 0.0, 0.0]),
            key_axis_obj_frame=np.array([1.0, 0.0, 0.0]),
        )]
        meas = contain_compatibility(container, con_comps, containee, key_comps)
        self.assertIsNotNone(meas)
        self.assertAlmostEqual(meas.radial_mismatch, 0.0, places=6)
        self.assertAlmostEqual(meas.axial_mismatch, 0.0, places=6)
        self.assertAlmostEqual(meas.orient_mismatch, 0.0, places=6)


# --------------------------------------------------------------------------- #
# No-no-* paired labels on physical-state edges
# --------------------------------------------------------------------------- #
class PhysicalStateLabelTests(unittest.TestCase):
    def _cfg(self):
        return {
            "contact": {"eps_force": 0.05},
            "grasp": {"max_angle": 30, "tcp_approach_axis_local": [0, 0, 1]},
            "bin_edges": {
                "planar-distance": [0.05, 0.20],
                "height-offset": [-0.10, 0.10],
            },
            "interaction_types": {},
        }

    def test_no_contact_when_force_below_eps(self):
        cfg = self._cfg()
        node = _obj_node("actor:bowl", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        graph = Graph(0, "env", "cam", nodes=[_ee_node(), node])
        edges = ee_object_spatial_event_edges(
            graph, _StubState(contact_force=0.01), cfg,
        )
        rels = {e.relation for e in edges}
        self.assertNotIn("contact", rels)
        self.assertNotIn("grasp", rels)

    def test_grasp_emits_only_positive_label(self):
        cfg = self._cfg()
        node = _obj_node("actor:bowl", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        graph = Graph(0, "env", "cam", nodes=[_ee_node(), node])
        edges = ee_object_spatial_event_edges(
            graph, _StubState(grasping=True, contact_force=2.0), cfg,
        )
        grasp = [e for e in edges if e.relation == "grasp"]
        self.assertEqual(len(grasp), 1)
        self.assertEqual(grasp[0].label, "grasp")


# --------------------------------------------------------------------------- #
# Vocabulary completeness sanity-check
# --------------------------------------------------------------------------- #
class VocabularyTests(unittest.TestCase):
    def test_new_compat_relations_registered(self):
        for rel in (
            "support-compatibility", "contain-compatibility",
            "support-compatibility-change", "contain-compatibility-change",
        ):
            self.assertIn(rel, ALL_LABELS)
            self.assertEqual(len(ALL_LABELS[rel]), 3 if "change" not in rel else 5)

    def test_no_transition_labels(self):
        # *-transition is removed from the vocabulary entirely.
        for rel in ALL_LABELS:
            self.assertFalse(rel.endswith("-transition"))


# --------------------------------------------------------------------------- #
# Temporal: new compat relation produces a change edge
# --------------------------------------------------------------------------- #
class TemporalNewCompatTests(unittest.TestCase):
    KEY = ("actor:a", "actor:b", "support-compatibility")

    def _push(self, buf, frame, value):
        ee = _ee_node()
        a = _obj_node("actor:a", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        b = _obj_node("actor:b", (0.05, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0))
        edge = Edge("actor:a", "actor:b", "support-compatibility",
                    "partial-match", float(value))
        graph = Graph(frame, "env", "cam", nodes=[ee, a, b], edges=[edge])
        buf.update(graph)

    def test_history_accumulates(self):
        buf = TemporalBuffer(K=2)
        self._push(buf, 0, 0.5)
        self._push(buf, 1, 0.3)
        self._push(buf, 2, 0.1)
        self.assertEqual(len(buf._values[self.KEY]), 3)


# --------------------------------------------------------------------------- #
# Geometric contain through `object_object_edges`
# --------------------------------------------------------------------------- #
class ObjectObjectContainEdgeTests(unittest.TestCase):
    def test_emits_contain_when_descriptors_present(self):
        a = _obj_node("actor:box", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        b = _obj_node("actor:peg", (0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        aff_set = AffordanceSet(
            contain_by_object={
                "box": [ContainComponent(
                    entry_anchor_obj_frame=np.array([0.0, 0.0, 0.0]),
                    entry_axis_obj_frame=np.array([1.0, 0.0, 0.0]),
                    opening_radius=0.02, depth=0.10,
                )],
            },
            key_by_object={
                "peg": [KeyComponent(
                    key_anchor_obj_frame=np.array([0.0, 0.0, 0.0]),
                    key_axis_obj_frame=np.array([1.0, 0.0, 0.0]),
                )],
            },
        )
        # Make sure the lookup hits canonical_affordance_key("actor:box") = "box".
        a.attributes["mshab_obj_id"] = "box"
        b.attributes["mshab_obj_id"] = "peg"
        cfg = {
            "contact": {"eps_force": 0.05},
            "support": {"eps_z": 0.02, "min_vertical_force_ratio": 0.5},
            "affordance_set": aff_set,
        }
        graph = Graph(0, "env", "cam", nodes=[a, b])
        edges = object_object_edges(graph, _StubState(), cfg)
        contain_edges = [e for e in edges if e.relation == "contain"]
        self.assertEqual(len(contain_edges), 1)
        self.assertEqual(contain_edges[0].src, "actor:box")
        self.assertEqual(contain_edges[0].dst, "actor:peg")


if __name__ == "__main__":
    unittest.main()
