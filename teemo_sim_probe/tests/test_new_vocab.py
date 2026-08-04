"""Hyper-relational fact emission, vocabularies, and packing.

Covers what the new contract guarantees: physical-state relations co-fire,
admissibility is whitelist-gated, the affordance near gate picks a label rather
than deleting a fact, and delta appears only where mu^rho and history allow.
"""

import unittest

import numpy as np

from teemo_sim_probe.adapters.graph_pack import pack_graph
from teemo_sim_probe.adapters.graph_vocab import (
    GraphVocab,
    EntityVocab,
    build_absolute_vocab,
    build_relation_vocab,
    build_temporal_vocab,
)
from teemo_sim_probe.core.relation_rules import (
    ABS_LABELS,
    HOLDS,
    NOT_HOLDS,
    RELATION_TYPES,
    TEMPORAL_RELATIONS,
    UNOBSERVED,
    ee_object_physical_edges,
    ee_object_spatial_edges,
    object_object_physical_edges,
)
from teemo_sim_probe.core.schema import Graph, Node
from teemo_sim_probe.core.temporal_buffer import TemporalBuffer


BINS = {
    "planar-distance": [0.1, 0.2, 0.3, 0.4],
    "height-offset": [-0.3, -0.1, 0.1, 0.3],
    "planar-distance-change": [-0.3, -0.1, 0.1, 0.3],
}


def _cfg(**over):
    cfg = {
        "bin_edges": dict(BINS),
        "contact": {"eps_force": 0.05},
        "grasp": {"max_angle": 30.0},
        "support": {"min_vertical_force_ratio": 0.5},
        "temporal": {"K": 2},
        "affordances": {},
        "pair_force_max_distance": 2.0,
    }
    cfg.update(over)
    return cfg


def _ee(pose=(0.0, 0.0, 0.0)):
    return Node(
        node_id="ee", node_type="ee", name="end_effector", visible=True,
        pose_world=[*pose, 1.0, 0.0, 0.0, 0.0],
    )


def _obj(node_id, pose, types=("contact", "grasp"), visible=True):
    return Node(
        node_id=node_id, node_type="object", name=node_id, visible=visible,
        pose_world=[*pose, 1.0, 0.0, 0.0, 0.0],
        segmentation_ids=[abs(hash(node_id)) & 0xffff],
        attributes={"is_actor": True, "whitelist_key": f"actor:{node_id}",
                    "interaction_types": list(types)},
    )


class _StubState:
    def __init__(self, *, force_vector=(0.0, 0.0, 0.0), grasping=False,
                 contact_force=0.0, tcp=(0.0, 0.0, 0.0)):
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


def _graph(*nodes):
    return Graph(frame=0, env_id="e", camera="c", nodes=list(nodes))


class PhysicalStateTests(unittest.TestCase):

    def test_contact_and_grasp_cofire(self):
        g = _graph(_ee(), _obj("bowl", (0.05, 0.0, 0.0)))
        state = _StubState(grasping=True, contact_force=5.0)
        by_rel = {e.relation: e for e in ee_object_physical_edges(g, state, _cfg())}
        self.assertEqual(by_rel["grasp"].label, HOLDS)
        self.assertEqual(by_rel["contact"].label, HOLDS)

    def test_negative_state_is_emitted_not_dropped(self):
        g = _graph(_ee(), _obj("bowl", (0.9, 0.0, 0.0)))
        state = _StubState(grasping=False, contact_force=0.0)
        by_rel = {e.relation: e for e in ee_object_physical_edges(g, state, _cfg())}
        self.assertEqual(by_rel["grasp"].label, NOT_HOLDS)
        self.assertEqual(by_rel["contact"].label, NOT_HOLDS)

    def test_missing_token_makes_the_fact_inadmissible(self):
        g = _graph(_ee(), _obj("counter", (0.2, 0.0, 0.0), types=("contact",)))
        edges = ee_object_physical_edges(g, _StubState(), _cfg())
        self.assertEqual({e.relation for e in edges}, {"contact"})

    def test_invisible_endpoint_emits_nothing(self):
        g = _graph(_ee(), _obj("bowl", (0.05, 0.0, 0.0), visible=False))
        self.assertEqual(ee_object_physical_edges(g, _StubState(), _cfg()), [])
        self.assertEqual(ee_object_spatial_edges(g, _StubState(), _cfg()), [])

    def test_support_direction_from_force_sign(self):
        a = _obj("counter", (0.0, 0.0, 0.0), types=("contact", "support"))
        b = _obj("bowl", (0.0, 0.0, 0.1), types=("contact", "support"))
        state = _StubState(force_vector=(0.0, 0.0, -8.0))
        edges = object_object_physical_edges(_graph(a, b), state, _cfg())
        support = {(e.src, e.dst): e.label for e in edges
                   if e.relation == "support"}
        self.assertEqual(support[("counter", "bowl")], HOLDS)
        self.assertEqual(support[("bowl", "counter")], NOT_HOLDS)

    def test_support_and_contact_cofire(self):
        a = _obj("counter", (0.0, 0.0, 0.0), types=("contact", "support"))
        b = _obj("bowl", (0.0, 0.0, 0.1), types=("contact", "support"))
        state = _StubState(force_vector=(0.0, 0.0, -8.0))
        edges = object_object_physical_edges(_graph(a, b), state, _cfg())
        self.assertEqual(
            [e.label for e in edges if e.relation == "contact"], [HOLDS])


class TemporalLabelTests(unittest.TestCase):

    def _run(self, distances, cfg):
        buf = TemporalBuffer(K=cfg["temporal"]["K"])
        labels = []
        for d in distances:
            g = _graph(_ee(), _obj("bowl", (d, 0.0, 0.0)))
            g.edges.extend(ee_object_spatial_edges(g, _StubState(), cfg))
            g.edges.extend(ee_object_physical_edges(g, _StubState(), cfg))
            buf.annotate(g, cfg)
            labels.append({e.relation: e.temp_label for e in g.edges})
        return labels

    def test_physical_state_never_carries_a_change(self):
        labels = self._run([0.5, 0.4, 0.3, 0.2], _cfg())
        self.assertTrue(all(f["contact"] is None for f in labels))
        self.assertTrue(all(f["grasp"] is None for f in labels))

    def test_change_needs_k_plus_one_samples(self):
        labels = self._run([0.5, 0.4, 0.3, 0.2], _cfg())
        self.assertIsNone(labels[0]["planar-distance"])
        self.assertIsNone(labels[1]["planar-distance"])
        self.assertIsNotNone(labels[2]["planar-distance"])

    def test_approach_reads_as_a_decrease(self):
        labels = self._run([0.9, 0.7, 0.5, 0.3], _cfg())
        self.assertTrue(labels[-1]["planar-distance"].startswith("decrease"))


class VocabularyTests(unittest.TestCase):

    def test_every_relation_has_labels(self):
        for name in RELATION_TYPES:
            self.assertTrue(ABS_LABELS[name])

    def test_unobserved_is_affordance_only(self):
        for name in RELATION_TYPES:
            has = UNOBSERVED in ABS_LABELS[name]
            self.assertEqual(has, name.endswith("-compatibility"))

    def test_temporal_family_matches_mu(self):
        self.assertEqual(
            TEMPORAL_RELATIONS,
            frozenset(n for n in RELATION_TYPES
                      if n.endswith("-compatibility")
                      or n in ("planar-distance", "height-offset")),
        )

    def test_index_zero_is_pad_everywhere(self):
        for vocab in (build_relation_vocab(), build_absolute_vocab(),
                      build_temporal_vocab()):
            self.assertEqual(vocab.encode(None), 0)
            self.assertNotIn(0, vocab.token_to_id.values())


def _vocab(keys):
    entity = EntityVocab(token_to_id={k: i for i, k in enumerate(keys)})
    relation = build_relation_vocab()
    absolute = build_absolute_vocab()
    temporal = build_temporal_vocab()
    abs_valid = np.zeros((len(relation), len(absolute)), bool)
    temp_valid = np.zeros((len(relation),), bool)
    for name in RELATION_TYPES:
        rid = relation.encode(name)
        for label in ABS_LABELS[name]:
            abs_valid[rid, absolute.encode(label)] = True
        temp_valid[rid] = name in TEMPORAL_RELATIONS
    return GraphVocab(entity, relation, absolute, temporal, abs_valid, temp_valid)


class PackingTests(unittest.TestCase):

    def _packed(self, n_max=4, e_max=8):
        ee = _ee()
        obj = _obj("bowl", (0.05, 0.0, 0.0))
        ee.feat = [1.0, 2.0]
        obj.feat = [3.0, 4.0]
        g = _graph(ee, obj)
        cfg = _cfg()
        state = _StubState(grasping=True, contact_force=5.0)
        g.edges.extend(ee_object_spatial_edges(g, state, cfg))
        g.edges.extend(ee_object_physical_edges(g, state, cfg))
        TemporalBuffer(K=cfg["temporal"]["K"]).annotate(g, cfg)
        vocab = _vocab(["<pad>", "<ee>", "actor:bowl"])
        return pack_graph(g, vocab, n_max=n_max, e_max=e_max, n_feat=2)

    def test_dtypes_stay_narrow(self):
        packed = self._packed()
        self.assertEqual(packed["graph_edge_src"].dtype, np.uint8)
        self.assertEqual(packed["graph_edge_abs"].dtype, np.uint8)
        self.assertEqual(packed["graph_node_ent"].dtype, np.uint16)
        self.assertEqual(packed["graph_node_feat"].dtype, np.float32)

    def test_padding_is_masked_out(self):
        packed = self._packed()
        self.assertEqual(int(packed["graph_n_nodes"]), 2)
        self.assertTrue((packed["graph_node_valid"][:2] == 1).all())
        self.assertTrue((packed["graph_node_valid"][2:] == 0).all())
        n_edges = int(packed["graph_n_edges"])
        self.assertTrue((packed["graph_edge_valid"][n_edges:] == 0).all())

    def test_physical_rows_carry_no_temporal_mask(self):
        packed = self._packed()
        rel = build_relation_vocab()
        physical = {rel.encode(n) for n in ("contact", "grasp")}
        for i in range(int(packed["graph_n_edges"])):
            if int(packed["graph_edge_rel"][i]) in physical:
                self.assertEqual(int(packed["graph_edge_temp_mask"][i]), 0)

    def test_endpoints_must_fit_a_byte(self):
        with self.assertRaises(ValueError):
            self._packed(n_max=300)


if __name__ == "__main__":
    unittest.main()
