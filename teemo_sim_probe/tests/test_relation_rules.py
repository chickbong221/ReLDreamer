"""Object--object physical state and the pair-force short-circuit.

Every admissible pair now reports a binary state, so a far pair still emits its
facts -- it just reports ``not-holds`` without paying the GPU query.
"""

import unittest

import numpy as np

from teemo_sim_probe.core.relation_rules import (
    HOLDS, NOT_HOLDS, object_object_physical_edges,
)
from teemo_sim_probe.core.schema import Graph, Node


class _State:
    def __init__(self, force_vector):
        self.force_vector = np.asarray(force_vector, dtype=float)
        self.seg_id_map = {}
        self.pair_force_calls = 0

    def pairwise_force_vector(self, _a, _b):
        self.pair_force_calls += 1
        return self.force_vector


_NEXT_SEG_ID = [0]


def _node_at(node_id, x, y, z, types=("contact", "support")):
    _NEXT_SEG_ID[0] += 1
    return Node(
        node_id=node_id,
        node_type="object",
        name=node_id,
        visible=True,
        pose_world=[x, y, z, 1.0, 0.0, 0.0, 0.0],
        segmentation_ids=[_NEXT_SEG_ID[0]],
        attributes={"interaction_types": list(types)},
    )


def _node(node_id, z, types=("contact", "support")):
    return _node_at(node_id, 0.0, 0.0, z, types)


def _cfg(**over):
    cfg = {
        "contact": {"eps_force": 0.05},
        "support": {"min_vertical_force_ratio": 0.5},
        "affordances": {},
    }
    cfg.update(over)
    return cfg


def _labels(edges):
    return {(e.src, e.dst, e.relation): e.label for e in edges}


class ObjectRelationTests(unittest.TestCase):

    def test_vertical_load_picks_one_supporter(self):
        cube = _node("cube", 1.0)
        table = _node("table", 0.0)
        graph = Graph(0, "env", "cam", nodes=[cube, table])

        labels = _labels(object_object_physical_edges(
            graph, _State([0.0, 0.0, 2.0]), _cfg()))

        self.assertEqual(labels[("table", "cube", "support")], HOLDS)
        self.assertEqual(labels[("cube", "table", "support")], NOT_HOLDS)
        self.assertEqual(labels[("cube", "table", "contact")], HOLDS)

    def test_horizontal_touch_is_contact_without_support(self):
        left = _node("left", 0.0)
        right = _node("right", 0.2)
        graph = Graph(0, "env", "cam", nodes=[left, right])

        labels = _labels(object_object_physical_edges(
            graph, _State([2.0, 0.0, 0.1]), _cfg()))

        self.assertEqual(labels[("left", "right", "contact")], HOLDS)
        self.assertEqual(labels[("left", "right", "support")], NOT_HOLDS)
        self.assertEqual(labels[("right", "left", "support")], NOT_HOLDS)

    def test_no_touch_still_emits_negative_facts(self):
        a = _node("a", 0.0)
        b = _node("b", 1.0)
        graph = Graph(0, "env", "cam", nodes=[a, b])

        labels = _labels(object_object_physical_edges(
            graph, _State([0.0, 0.0, 0.0]), _cfg()))

        self.assertEqual(set(labels.values()), {NOT_HOLDS})
        self.assertEqual(len(labels), 3)  # one contact, two support orderings

    def test_missing_token_makes_the_pair_inadmissible(self):
        a = _node("a", 0.0, types=("contact",))
        b = _node("b", 0.05, types=("support",))
        graph = Graph(0, "env", "cam", nodes=[a, b])

        edges = object_object_physical_edges(
            graph, _State([0.0, 0.0, 2.0]), _cfg())

        self.assertEqual(edges, [])

    def test_invisible_endpoint_is_inadmissible(self):
        a = _node("a", 0.0)
        b = _node("b", 0.05)
        b.visible = False
        graph = Graph(0, "env", "cam", nodes=[a, b])

        self.assertEqual(
            object_object_physical_edges(graph, _State([0.0, 0.0, 2.0]), _cfg()),
            [],
        )


class PairForceDistanceGateTests(unittest.TestCase):
    """The short-circuit skips the GPU query for pairs that cannot touch; the
    facts still emit, as ``not-holds``."""

    def test_close_pair_still_queries_force(self):
        graph = Graph(0, "env", "cam", nodes=[
            _node_at("a", 0.0, 0.0, 0.0), _node_at("b", 0.3, 0.0, 0.0)])
        state = _State([1.0, 0.0, 0.1])

        labels = _labels(object_object_physical_edges(graph, state, _cfg()))

        self.assertEqual(state.pair_force_calls, 1)
        self.assertEqual(labels[("a", "b", "contact")], HOLDS)

    def test_far_pair_skips_force_query_entirely(self):
        graph = Graph(0, "env", "cam", nodes=[
            _node_at("a", 0.0, 0.0, 0.0), _node_at("b", 5.0, 0.0, 0.0)])
        state = _State([10.0, 0.0, 0.0])  # would have been contact

        labels = _labels(object_object_physical_edges(graph, state, _cfg()))

        self.assertEqual(
            state.pair_force_calls, 0,
            "far pair must not fire the GPU pair-force query",
        )
        self.assertEqual(labels[("a", "b", "contact")], NOT_HOLDS)

    def test_gate_can_be_disabled_by_zero_threshold(self):
        graph = Graph(0, "env", "cam", nodes=[
            _node_at("a", 0.0, 0.0, 0.0), _node_at("b", 5.0, 0.0, 0.0)])
        state = _State([1.0, 0.0, 0.0])

        labels = _labels(object_object_physical_edges(
            graph, state, _cfg(pair_force_max_distance=0.0)))

        self.assertEqual(state.pair_force_calls, 1)
        self.assertEqual(labels[("a", "b", "contact")], HOLDS)


if __name__ == "__main__":
    unittest.main()
