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


_NEXT_SEG_ID = [0]


def _node(node_id, z):
    _NEXT_SEG_ID[0] += 1
    return Node(
        node_id=node_id,
        node_type="object",
        name=node_id,
        pose_world=[0.0, 0.0, z, 1.0, 0.0, 0.0, 0.0],
        segmentation_ids=[_NEXT_SEG_ID[0]],
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

    def test_horizontal_touch_emits_only_contact(self):
        left = _node("left", 0.0)
        right = _node("right", 0.2)
        graph = Graph(0, "env", "cam", nodes=[left, right])

        edges = object_object_edges(graph, _State([2.0, 0.0, 0.1]), _cfg())

        self.assertEqual(len(edges), 1)
        self.assertEqual(
            (edges[0].relation, edges[0].label, edges[0].masked),
            ("contact", "contact", False),
        )

    def test_no_touch_emits_no_edges(self):
        a = _node("a", 0.0)
        b = _node("b", 1.0)
        graph = Graph(0, "env", "cam", nodes=[a, b])

        edges = object_object_edges(graph, _State([0.0, 0.0, 0.0]), _cfg())

        self.assertEqual(edges, [])


class _CountingState:
    """Counts pairwise_force_vector calls so tests can assert the physics
    short-circuit really skipped the GPU query."""

    def __init__(self, force_vector):
        self.force_vector = np.asarray(force_vector, dtype=float)
        self.seg_id_map = {}
        self.pair_force_calls = 0

    def pairwise_force_vector(self, _a, _b):
        self.pair_force_calls += 1
        return self.force_vector


def _node_at(node_id, x, y, z):
    _NEXT_SEG_ID[0] += 1
    return Node(
        node_id=node_id,
        node_type="object",
        name=node_id,
        pose_world=[x, y, z, 1.0, 0.0, 0.0, 0.0],
        segmentation_ids=[_NEXT_SEG_ID[0]],
    )


class PairForceDistanceGateTests(unittest.TestCase):
    """Guard the physics-based short-circuit that skips GPU pair-force queries
    for object pairs whose centers cannot physically be in contact."""

    def test_close_pair_still_queries_force(self):
        a = _node_at("a", 0.0, 0.0, 0.0)
        b = _node_at("b", 0.3, 0.0, 0.0)  # 0.3 m -- well inside 2 m gate
        graph = Graph(0, "env", "cam", nodes=[a, b])
        state = _CountingState([1.0, 0.0, 0.1])

        edges = object_object_edges(graph, state, _cfg())

        self.assertEqual(state.pair_force_calls, 1)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].relation, "contact")

    def test_far_pair_skips_force_query_entirely(self):
        a = _node_at("a", 0.0, 0.0, 0.0)
        b = _node_at("b", 5.0, 0.0, 0.0)  # 5 m -- beyond default 2 m gate
        graph = Graph(0, "env", "cam", nodes=[a, b])
        state = _CountingState([10.0, 0.0, 0.0])  # would have been contact

        edges = object_object_edges(graph, state, _cfg())

        self.assertEqual(
            state.pair_force_calls, 0,
            "far pair must not fire the GPU pair-force query",
        )
        self.assertEqual(edges, [])

    def test_gate_can_be_disabled_by_zero_threshold(self):
        a = _node_at("a", 0.0, 0.0, 0.0)
        b = _node_at("b", 5.0, 0.0, 0.0)
        graph = Graph(0, "env", "cam", nodes=[a, b])
        state = _CountingState([1.0, 0.0, 0.0])
        cfg = _cfg()
        cfg["pair_force_max_distance"] = 0.0  # disable

        edges = object_object_edges(graph, state, cfg)

        self.assertEqual(state.pair_force_calls, 1)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].relation, "contact")

    def test_missing_pose_falls_through_to_force_query(self):
        # If pose_world is unavailable we cannot compute distance, so the
        # gate must NOT skip -- we fall back to the current behaviour.
        a = _node_at("a", 0.0, 0.0, 0.0)
        b = _node_at("b", 5.0, 0.0, 0.0)
        b.pose_world = None
        graph = Graph(0, "env", "cam", nodes=[a, b])
        state = _CountingState([1.0, 0.0, 0.0])

        edges = object_object_edges(graph, state, _cfg())

        self.assertEqual(state.pair_force_calls, 1)
        self.assertEqual(len(edges), 1)


if __name__ == "__main__":
    unittest.main()
