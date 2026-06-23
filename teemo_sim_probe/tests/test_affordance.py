"""Tests for affordance-based relations and supporting plumbing."""

import json
import os
import tempfile
import unittest

import numpy as np

from teemo_sim_probe.core.affordance import (
    AffordanceComponent,
    AffordanceSet,
    canonical_affordance_key,
    load_affordance_set,
    lookup_components,
    select_active_component,
    transform_anchors,
)
from teemo_sim_probe.core.persistence import _snapshot
from teemo_sim_probe.core.relation_rules import affordance_edges
from teemo_sim_probe.core.schema import Edge, Graph, Node
from teemo_sim_probe.core.temporal_buffer import TemporalBuffer


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class _State:
    """Minimal stand-in for PrivilegedState used by relation_rules."""

    def __init__(self, tcp_xyz, gripper_width=None):
        self.tcp_pose_world = np.array(
            [*tcp_xyz, 1.0, 0.0, 0.0, 0.0], dtype=float
        )
        self.gripper_width = gripper_width


def _cfg(aff_set=None):
    return {
        "affordance_set": aff_set if aff_set is not None else AffordanceSet(),
        "profile": {
            "tcp_affordance_alignment": {
                "edges": [0.01, 0.03, 0.07, 0.15],
                "labels": ["on-anchor", "very-near", "near", "off", "far"],
            },
            "gripper_width_alignment": {
                "edges": [-0.03, -0.01, 0.01, 0.03],
                "labels": [
                    "much-tighter", "tighter", "matched", "looser", "much-looser",
                ],
            },
            "tcp_affordance_alignment_change": {
                "edges": [-0.05, -0.02, -0.005, 0.005, 0.02, 0.05],
                "labels": [
                    "approach-anchor-fast", "approach-anchor-medium",
                    "approach-anchor-slow", "stable-anchor",
                    "leave-anchor-slow", "leave-anchor-medium",
                    "leave-anchor-fast",
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


def _obj_node(
    name="024_bowl",
    node_id=None,
    pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    kind="obj",
    active=True,
    mshab_id="024_bowl",
):
    node = Node(
        node_id=node_id or f"actor:{name}",
        node_type="object",
        name=name,
        pose_world=list(pose),
        attributes={"is_actor": True},
    )
    if active:
        node.attributes["is_mshab_active_target"] = True
        node.attributes["mshab_kind"] = kind
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

    def test_pose_invariance(self):
        """Same anchor under different obj poses lands on the rotated/translated location."""
        anchor = np.array([0.1, 0.0, 0.05])
        comps = [AffordanceComponent(anchor, 0.045)]
        # 180deg around z
        c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
        out = transform_anchors([0.5, 0.5, 0.5, c, 0.0, 0.0, s], comps)
        # After 180deg-Z: x -> -x is wrong; 90deg-Z: x -> y. Verify analytically.
        R = np.array([[c * c - s * s, -2 * s * c, 0],
                      [2 * s * c, c * c - s * s, 0],
                      [0, 0, 1.0]])
        expected = np.array([0.5, 0.5, 0.5]) + R @ anchor
        np.testing.assert_allclose(out[0], expected, atol=1e-9)

    def test_degenerate(self):
        comps = [AffordanceComponent(np.array([0.0, 0.0, 0.0]), 0.045)]
        self.assertIsNone(transform_anchors(None, comps))
        # Zero quaternion -> degenerate.
        self.assertIsNone(transform_anchors([0, 0, 0, 0, 0, 0, 0], comps))


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

    def test_basic_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "aff.json")
            with open(path, "w") as f:
                json.dump({
                    "_README": "x",
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
        self.assertAlmostEqual(s.by_object["024_bowl"][1].preferred_width, 0.050)

    def test_skips_bad_entries(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "aff.json")
            with open(path, "w") as f:
                json.dump({"objects": {
                    "024_bowl": {"components": [
                        {"anchor": [0.0, 0.0, 0.02], "width": 0.045},  # ok
                        {"anchor": [0.0, 0.0], "width": 0.045},        # bad shape
                        {"width": 0.045},                              # missing anchor
                        {"anchor": [0.0, 0.0, 0.02]},                  # missing width
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
        node = _obj_node(name="env-0_024_bowl-3", mshab_id="024_bowl")
        self.assertIsNotNone(lookup_components(s, node))

    def test_canonicalizes_mshab_obj_id(self):
        s = self._set()
        node = _obj_node(name="random", mshab_id="env-0_024_bowl-3")
        self.assertIsNotNone(lookup_components(s, node))

    def test_falls_back_to_name(self):
        s = self._set()
        node = _obj_node(name="env-0_024_bowl-3", mshab_id=None)
        # Constructor set the attribute; clear it to force fallback.
        node.attributes.pop("mshab_obj_id", None)
        self.assertIsNotNone(lookup_components(s, node))

    def test_missing(self):
        s = self._set()
        node = _obj_node(name="999_phantom", mshab_id="999_phantom")
        self.assertIsNone(lookup_components(s, node))


# --------------------------------------------------------------------------- #
# Edge emission tests
# --------------------------------------------------------------------------- #
class AffordanceEdgeTests(unittest.TestCase):
    def _aff_set(self):
        return AffordanceSet(by_object={
            "024_bowl": [
                AffordanceComponent(np.array([0.0, 0.0, 0.02]), 0.045),
                AffordanceComponent(np.array([0.05, 0.0, 0.01]), 0.050),
            ]
        })

    def test_emits_both_relations_for_active_obj(self):
        cfg = _cfg(self._aff_set())
        # TCP exactly on the second anchor in world frame (obj at origin).
        state = _State([0.05, 0.0, 0.01], gripper_width=0.050)
        node = _obj_node()
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        edges = affordance_edges(graph, state, cfg)
        rels = sorted(e.relation for e in edges)
        self.assertEqual(
            rels, ["gripper-width-alignment", "tcp-affordance-alignment"]
        )

    def test_records_a_star_on_node(self):
        cfg = _cfg(self._aff_set())
        state = _State([0.05, 0.0, 0.01], gripper_width=0.050)
        node = _obj_node()
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        affordance_edges(graph, state, cfg)
        self.assertEqual(node.attributes.get("affordance_a_star"), 1)

    def test_shared_a_star_drives_both_edges(self):
        cfg = _cfg(self._aff_set())
        state = _State([0.0, 0.0, 0.02], gripper_width=0.020)  # near anchor 0
        node = _obj_node()
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        edges = affordance_edges(graph, state, cfg)
        self.assertEqual(node.attributes.get("affordance_a_star"), 0)
        width_edge = [e for e in edges if e.relation == "gripper-width-alignment"][0]
        # preferred_width of component 0 is 0.045; signed err = 0.020 - 0.045 < 0
        self.assertLess(width_edge.raw_value, 0.0)
        self.assertIn(width_edge.label, ["much-tighter", "tighter"])

    def test_skips_handle(self):
        cfg = _cfg(self._aff_set())
        state = _State([0.0, 0.0, 0.0], gripper_width=0.045)
        node = _obj_node(kind="handle")
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        self.assertEqual(affordance_edges(graph, state, cfg), [])

    def test_skips_non_active_target(self):
        cfg = _cfg(self._aff_set())
        state = _State([0.0, 0.0, 0.0], gripper_width=0.045)
        node = _obj_node(active=False)
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        self.assertEqual(affordance_edges(graph, state, cfg), [])

    def test_missing_asset_entry_no_edges(self):
        cfg = _cfg(self._aff_set())
        state = _State([0.0, 0.0, 0.0], gripper_width=0.045)
        node = _obj_node(name="999_phantom", mshab_id="999_phantom")
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        self.assertEqual(affordance_edges(graph, state, cfg), [])

    def test_no_pose_no_edges(self):
        cfg = _cfg(self._aff_set())
        state = _State([0.0, 0.0, 0.0], gripper_width=0.045)
        node = _obj_node()
        node.pose_world = None
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        self.assertEqual(affordance_edges(graph, state, cfg), [])

    def test_empty_set_no_edges(self):
        cfg = _cfg(AffordanceSet())
        state = _State([0.0, 0.0, 0.0], gripper_width=0.045)
        node = _obj_node()
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        self.assertEqual(affordance_edges(graph, state, cfg), [])

    def test_no_gripper_width_omits_width_edge(self):
        cfg = _cfg(self._aff_set())
        state = _State([0.05, 0.0, 0.01], gripper_width=None)
        node = _obj_node()
        graph = Graph(0, "env", "cam", nodes=[_ee(), node])
        edges = affordance_edges(graph, state, cfg)
        rels = [e.relation for e in edges]
        self.assertIn("tcp-affordance-alignment", rels)
        self.assertNotIn("gripper-width-alignment", rels)

    def test_pose_invariant_reuse(self):
        """SAME stored anchor must reduce to the same TCP-anchor distance
        regardless of where the object currently sits."""
        cfg = _cfg(self._aff_set())
        # Place the bowl at world (1, 2, 3), no rotation; TCP at world
        # (1.05, 2.0, 3.01) -> sits on the second anchor (offset [0.05,0,0.01]).
        bowl = _obj_node(pose=(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0))
        state = _State([1.05, 2.0, 3.01], gripper_width=0.050)
        graph = Graph(0, "env", "cam", nodes=[_ee(), bowl])
        edges = affordance_edges(graph, state, cfg)
        align = [e for e in edges if e.relation == "tcp-affordance-alignment"][0]
        self.assertLess(align.raw_value, 1e-6)


# --------------------------------------------------------------------------- #
# Persistence stripping
# --------------------------------------------------------------------------- #
class PersistenceStrippingTests(unittest.TestCase):
    def test_snapshot_strips_dynamic_mshab_attrs(self):
        n = _obj_node()
        n.attributes["affordance_a_star"] = 1
        snap = _snapshot(n)
        for k in ("is_mshab_active_target", "mshab_kind", "mshab_obj_id",
                  "affordance_a_star"):
            self.assertNotIn(k, snap.attributes)
        # Non-dynamic attrs are preserved.
        self.assertIn("is_actor", snap.attributes)


# --------------------------------------------------------------------------- #
# Temporal history reset
# --------------------------------------------------------------------------- #
class TemporalAffordanceResetTests(unittest.TestCase):
    KEY = ("ee", "actor:024_bowl", "tcp-affordance-alignment")

    def _push(self, buf, frame, value, a_star=0):
        ee = _ee()
        node = _obj_node()
        if a_star is not None:
            node.attributes["affordance_a_star"] = a_star
        graph = Graph(frame, "env", "cam", nodes=[ee, node], edges=[
            Edge("ee", node.node_id, "tcp-affordance-alignment",
                 "near", float(value)),
        ])
        buf.update(graph)

    def test_history_appends_when_a_star_stable(self):
        buf = TemporalBuffer(K=3)
        self._push(buf, 0, 0.05, a_star=0)
        self._push(buf, 1, 0.04, a_star=0)
        self._push(buf, 2, 0.03, a_star=0)
        self.assertEqual(len(buf._values[self.KEY]), 3)

    def test_history_resets_on_a_star_switch(self):
        buf = TemporalBuffer(K=3)
        self._push(buf, 0, 0.05, a_star=0)
        self._push(buf, 1, 0.04, a_star=0)
        self._push(buf, 2, 0.03, a_star=1)  # switch
        # After reset, only the new sample remains.
        self.assertEqual(len(buf._values[self.KEY]), 1)
        self.assertAlmostEqual(buf._values[self.KEY][0], 0.03)

    def test_history_drops_on_edge_disappearance(self):
        buf = TemporalBuffer(K=3)
        self._push(buf, 0, 0.05, a_star=0)
        self._push(buf, 1, 0.04, a_star=0)
        # Frame 2: same node, no affordance edge.
        ee = _ee()
        node = _obj_node()
        graph = Graph(2, "env", "cam", nodes=[ee, node], edges=[])
        buf.update(graph)
        self.assertNotIn(self.KEY, buf._values)


if __name__ == "__main__":
    unittest.main()
