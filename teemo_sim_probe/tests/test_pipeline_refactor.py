"""Regression tests for the simplified whitelist/segmentation pipeline."""

import unittest

import numpy as np

from teemo_sim_probe.core.graph_builder import GraphBuilder
from teemo_sim_probe.core.node_builder import build_nodes
from teemo_sim_probe.core.schema import Edge, Graph, Node
from teemo_sim_probe.core.temporal_buffer import TemporalBuffer
from teemo_sim_probe.core.entity_identity import stable_entity_key
from teemo_sim_probe.tools.build_subtask_whitelists import _WhitelistBuilder


class OneHopWhitelistTests(unittest.TestCase):
    def test_interactions_and_direct_support_only(self):
        builder = _WhitelistBuilder("pick", "actor:024_bowl")
        builder.absorb({
            "interacted": [
                {"key": "actor:024_bowl", "kind": "actor", "name": "bowl"},
                {"key": "link:cabinet/handle", "kind": "link", "name": "handle"},
            ],
            "supports": [
                {
                    "supporter": {
                        "key": "link:cabinet/drawer", "kind": "link",
                        "name": "drawer",
                    },
                    "supported_key": "actor:024_bowl",
                },
                # Recursive cabinet -> drawer evidence must be ignored because
                # drawer is a supporter, not an interacted root.
                {
                    "supporter": {
                        "key": "link:cabinet/body", "kind": "link",
                        "name": "body",
                    },
                    "supported_key": "link:cabinet/drawer",
                },
            ],
        })
        members = builder.payload()["members"]
        self.assertIn("actor:024_bowl", members)
        self.assertIn("link:cabinet/handle", members)
        self.assertIn("link:cabinet/drawer", members)
        self.assertNotIn("link:cabinet/body", members)
        self.assertEqual(members["link:cabinet/handle"]["roles"], ["interacted"])

    def test_uninteracted_handle_is_not_admitted(self):
        builder = _WhitelistBuilder("pick", "actor:024_bowl")
        builder.absorb({"interacted": [], "supports": []})
        self.assertNotIn("link:cabinet/handle", builder.payload()["members"])
        self.assertNotIn("actor:024_bowl", builder.payload()["members"])

    def test_supporter_requires_supported_interaction(self):
        builder = _WhitelistBuilder("pick", "actor:024_bowl")
        builder.absorb({
            "interacted": [],
            "supports": [{
                "supporter": {
                    "key": "link:cabinet/drawer", "kind": "link",
                    "name": "drawer",
                },
                "supported_key": "actor:024_bowl",
            }],
        })
        self.assertNotIn("link:cabinet/drawer", builder.payload()["members"])

    def test_interaction_types_track_contact_and_grasp(self):
        builder = _WhitelistBuilder("pick", "actor:024_bowl")
        builder.absorb({
            "interacted": [
                {"key": "actor:024_bowl", "kind": "actor", "name": "bowl",
                 "grasped": True},
                {"key": "link:cabinet/handle", "kind": "link", "name": "handle"},
            ],
            "supports": [{
                "supporter": {"key": "link:cabinet/drawer", "kind": "link",
                              "name": "drawer"},
                "supported_key": "actor:024_bowl",
            }],
        })
        members = builder.payload()["members"]
        # Bowl was grasped AND participated in a support pair as the supported
        # entity, so it carries contact + grasp + support tokens.
        self.assertEqual(
            members["actor:024_bowl"]["interaction_types"],
            ["contact", "grasp", "support"],
        )
        self.assertEqual(
            members["link:cabinet/handle"]["interaction_types"], ["contact"],
        )
        # Supporters now carry the ``support`` token so the runtime can gate
        # obj-obj support-compatibility on it.
        self.assertEqual(
            members["link:cabinet/drawer"]["interaction_types"], ["support"],
        )

    def test_bin_edges_emitted_from_bin_samples(self):
        builder = _WhitelistBuilder("pick", "actor:024_bowl")
        # Repeat each sample so the 0.9 quantile equals the constant value.
        builder.absorb({
            "interacted": [
                {"key": "actor:024_bowl", "kind": "actor", "name": "bowl"},
            ],
            "supports": [],
            "bin_samples": {
                "planar_distance": [0.60] * 10,
                "height_offset": [0.30] * 10,
                "planar_distance_change": [0.10] * 10,
            },
        })
        payload = builder.payload()
        bins = payload["bin_edges"]
        # Equal-width on [0, 0.60] -> [0.20, 0.40].
        self.assertAlmostEqual(bins["planar-distance"][0], 0.20, places=6)
        self.assertAlmostEqual(bins["planar-distance"][1], 0.40, places=6)
        # Signed 3-bin on |max|=0.30 -> [-0.10, 0.10].
        self.assertAlmostEqual(bins["height-offset"][0], -0.10, places=6)
        self.assertAlmostEqual(bins["height-offset"][1], 0.10, places=6)
        # Compatibility absolute bins are always [1/3, 2/3].
        self.assertAlmostEqual(bins["grasp-compatibility"][0], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(bins["contact-compatibility"][1], 2.0 / 3.0, places=6)


class StaleEdgeTests(unittest.TestCase):
    def test_last_observed_edge_is_restored_as_stale(self):
        graph_builder = GraphBuilder.__new__(GraphBuilder)
        graph_builder._edge_history = {}
        ee = Node("ee", "ee", "end_effector")
        bowl = Node("actor:024_bowl", "object", "bowl")
        fresh = Graph(0, "env", "cam", nodes=[ee, bowl], edges=[
            Edge("ee", bowl.node_id, "contact", "contact", 2.0),
        ])
        graph_builder._attach_stale_edges(fresh, 0)

        stale_bowl = Node(
            bowl.node_id, "object", "bowl", visible=False,
            frozen_pose=True, persistent=True,
        )
        stale = Graph(3, "env", "cam", nodes=[ee, stale_bowl], edges=[])
        graph_builder._attach_stale_edges(stale, 3)
        self.assertEqual(len(stale.edges), 1)
        self.assertTrue(stale.edges[0].stale)
        self.assertEqual(stale.edges[0].observed_frame, 0)
        self.assertEqual(stale.edges[0].age, 3)

    def test_stale_edge_does_not_advance_temporal_history(self):
        # Use a continuous relation (planar-distance) since physical-state
        # predicates no longer feed the temporal buffer at all.
        buffer = TemporalBuffer(K=1)
        ee = Node("ee", "ee", "end_effector")
        bowl = Node("actor:024_bowl", "object", "bowl")
        fresh = Graph(0, "env", "cam", nodes=[ee, bowl], edges=[
            Edge("ee", bowl.node_id, "planar-distance", "near", 0.05),
        ])
        buffer.update(fresh)
        stale_bowl = Node(
            bowl.node_id, "object", "bowl", visible=False, frozen_pose=True,
        )
        stale = Graph(1, "env", "cam", nodes=[ee, stale_bowl], edges=[
            Edge(
                "ee", bowl.node_id, "planar-distance", "near", 0.10,
                stale=True, observed_frame=0, age=1,
            ),
        ])
        buffer.update(stale)
        # The fresh value gets purged when bowl becomes frozen-pose; the
        # stale-tagged edge is not re-ingested.
        self.assertNotIn(("ee", bowl.node_id, "planar-distance"), buffer._values)


class _Pose:
    p = np.array([[0.0, 0.0, 0.0]])
    q = np.array([[1.0, 0.0, 0.0, 0.0]])


class _Articulation:
    name = "env-0_cabinet-2"


class Link:
    def __init__(self, name):
        self.name = name
        self.pose = _Pose()
        self.articulation = _Articulation()


class _State:
    def __init__(self, link):
        self.env_idx = 0
        self.tcp_pose_world = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
        self.seg_id_map = {7: link}
        self.robot_links = set()
        self.ee_links = []


class SupporterMaskTests(unittest.TestCase):
    def test_visible_static_named_link_keeps_its_mask(self):
        link = Link("frl_apartment_drawer3")
        seg = np.array([[0, 7], [7, 7]], dtype=np.int32)
        nodes, masks, _cam, _rgb = build_nodes(
            {}, _State(link), seg_override=seg,
            rgb_override=np.zeros((2, 2, 3), dtype=np.uint8),
        )
        key = stable_entity_key(link)
        self.assertIn(key, nodes)
        self.assertEqual(masks.area(key), 3)
        self.assertEqual(key, "link:cabinet-2/frl_apartment_drawer3")


if __name__ == "__main__":
    unittest.main()
