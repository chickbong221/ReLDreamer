import unittest

import numpy as np

from teemo_sim_probe.core.relation_rules import object_object_edges
from teemo_sim_probe.core.schema import Graph, Node


class _State:
    def __init__(self, force_vector):
        self.force_vector = np.asarray(force_vector, dtype=float)
        self.seg_id_map = {}

    def pairwise_force_vector(self, _a, _b):
        return self.force_vector


def _node(node_id, z):
    return Node(
        node_id=node_id,
        node_type="object",
        name=node_id,
        pose_world=[0.0, 0.0, z, 1.0, 0.0, 0.0, 0.0],
    )


def _cfg():
    return {
        "contact": {"eps_force": 0.05},
        "support": {"eps_z": 0.02, "min_vertical_force_ratio": 0.5},
    }


class ObjectRelationTests(unittest.TestCase):
    def test_vertical_load_emits_only_supporter_to_supported(self):
        cube = _node("cube", 1.0)
        table = _node("table", 0.0)
        graph = Graph(0, "env", "cam", nodes=[cube, table])

        edges = object_object_edges(graph, _State([0.0, 0.0, 2.0]), _cfg())

        self.assertEqual(len(edges), 1)
        self.assertEqual(
            (edges[0].src, edges[0].dst, edges[0].relation),
            ("table", "cube", "support"),
        )

    def test_horizontal_touch_emits_contact_plus_masked_no_support(self):
        left = _node("left", 0.0)
        right = _node("right", 0.2)
        graph = Graph(0, "env", "cam", nodes=[left, right])

        edges = object_object_edges(graph, _State([2.0, 0.0, 0.1]), _cfg())

        rels = {(e.relation, e.label, e.masked) for e in edges}
        self.assertIn(("contact", "contact", False), rels)
        self.assertIn(("support", "no-support", True), rels)
        self.assertEqual(len(edges), 2)

    def test_no_touch_emits_masked_no_contact_and_no_support(self):
        a = _node("a", 0.0)
        b = _node("b", 1.0)
        graph = Graph(0, "env", "cam", nodes=[a, b])

        edges = object_object_edges(graph, _State([0.0, 0.0, 0.0]), _cfg())

        rels = {(e.relation, e.label, e.masked) for e in edges}
        self.assertIn(("contact", "no-contact", True), rels)
        self.assertIn(("support", "no-support", True), rels)
        self.assertEqual(len(edges), 2)


if __name__ == "__main__":
    unittest.main()
