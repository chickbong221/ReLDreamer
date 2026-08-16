"""Relation-only contract, uid identity, and slot alignment.

The jax half skips silently without jax, so confirm the training env reports no
skips before trusting a green run.
"""

import unittest

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS, pack_graph
from scenegraph.adapters.graph_vocab import EntityVocab, GraphVocab, Vocab
from scenegraph.core.schema import Edge, Graph, Node
from scenegraph.core.selector import UID_EE, EntityRegistry

N_MAX, E_MAX = 6, 64


def _vocab(keys):
    entity = EntityVocab(token_to_id={k: i for i, k in enumerate(keys)})
    rel = Vocab(token_to_id={"contact": 1, "planar-distance": 2})
    absolute = Vocab(token_to_id={"holds": 1, "not-holds": 2, "near": 3})
    temporal = Vocab(token_to_id={"increase": 1, "decrease": 2})
    return GraphVocab(
        entity=entity, relation=rel, absolute=absolute, temporal=temporal,
        abs_valid=np.ones((len(rel), len(absolute)), bool),
        temp_valid=np.ones((len(rel),), bool))


def _node(node_id, kind, index, uid, key=None):
    node = Node(node_id=node_id, node_type=kind, name=node_id, index=index)
    node.uid = uid
    if key is not None:
        node.attributes["whitelist_key"] = key
    return node


class ObservationContract(unittest.TestCase):

    def _packed(self):
        ee = _node("ee", "ee", 0, UID_EE)
        bowl = _node("actor:bowl", "object", 1, 7, "actor:bowl")
        graph = Graph(frame=0, env_id="env0", camera="head", nodes=[ee, bowl])
        graph.edges.append(Edge(
            "ee", "actor:bowl", "contact", "holds", temp_label=None))
        graph.meta["active_target_node_id"] = "actor:bowl"
        vocab = _vocab(["<pad>", "<ee>", "actor:bowl"])
        return pack_graph(graph, vocab, n_max=N_MAX, e_max=E_MAX)

    def test_only_the_relation_only_arrays_exist(self):
        self.assertEqual(set(self._packed()), set(GRAPH_KEYS))
        for dead in ("graph_node_app", "graph_node_bbox"):
            self.assertNotIn(dead, GRAPH_KEYS)

    def test_shapes_are_fixed(self):
        packed = self._packed()
        for key in ("graph_node_ent", "graph_node_uid", "graph_node_target"):
            self.assertEqual(packed[key].shape, (N_MAX,))
        for key in GRAPH_KEYS:
            if key.startswith("graph_edge"):
                self.assertEqual(packed[key].shape, (E_MAX,))

    def test_dtypes_stay_narrow(self):
        packed = self._packed()
        self.assertEqual(packed["graph_node_uid"].dtype, np.uint16)
        self.assertEqual(packed["graph_edge_src"].dtype, np.uint8)

    def test_uid_and_target_land_on_the_owning_slot(self):
        packed = self._packed()
        self.assertEqual(int(packed["graph_node_uid"][0]), UID_EE)
        self.assertEqual(int(packed["graph_node_uid"][1]), 7)
        self.assertEqual(list(packed["graph_node_target"]), [0, 1, 0, 0, 0, 0])

    def test_an_unassigned_uid_fails_loud(self):
        ee = _node("ee", "ee", 0, UID_EE)
        bowl = _node("actor:bowl", "object", 1, 0, "actor:bowl")
        graph = Graph(frame=0, env_id="env0", camera="head", nodes=[ee, bowl])
        vocab = _vocab(["<pad>", "<ee>", "actor:bowl"])
        with self.assertRaisesRegex(ValueError, "no uid"):
            pack_graph(graph, vocab, n_max=N_MAX, e_max=E_MAX)


class UidIdentity(unittest.TestCase):

    def _registry(self, seed=0):
        return EntityRegistry(n_max=N_MAX, uid_vocab=32, seed=seed)

    def _nodes(self, *ids):
        out = {"ee": Node(node_id="ee", node_type="ee", name="ee")}
        for i in ids:
            out[i] = Node(node_id=i, node_type="object", name=i)
        return out

    def test_ee_holds_index_zero_and_the_reserved_uid(self):
        reg = self._registry()
        admitted = reg.assign(self._nodes("a"))
        self.assertEqual(admitted["ee"].index, 0)
        self.assertEqual(admitted["ee"].uid, UID_EE)

    def test_a_uid_is_stable_within_an_episode(self):
        reg = self._registry()
        first = reg.assign(self._nodes("a", "b"))
        uids = {k: v.uid for k, v in first.items()}
        second = reg.assign(self._nodes("b", "a"))
        self.assertEqual({k: v.uid for k, v in second.items()}, uids)

    def test_uids_are_permuted_between_episodes(self):
        reg = self._registry()
        seen = set()
        for _ in range(8):
            reg.reset_episode()
            seen.add(reg.assign(self._nodes("a"))["a"].uid)
        self.assertGreater(len(seen), 1)

    def test_uid_does_not_encode_category(self):
        left = self._registry(seed=1)
        right = self._registry(seed=2)
        self.assertNotEqual(
            left.assign(self._nodes("a"))["a"].uid,
            right.assign(self._nodes("a"))["a"].uid)

    def test_reset_clears_every_uid(self):
        reg = self._registry()
        reg.assign(self._nodes("a"))
        reg.reset_episode()
        self.assertEqual(reg.uid_of("a"), 0)

    def test_the_target_is_never_evicted(self):
        reg = self._registry()
        reg.assign(self._nodes("a", "b", "c", "d", "e"), protected=("a",))
        reg.assign(self._nodes("a", "b", "c", "d", "e", "f"), protected=("a",))
        self.assertIsNotNone(reg.index_of("a"))
        self.assertGreater(reg.overflow_drops, 0)

    def test_capacity_is_n_max_minus_the_ee(self):
        reg = self._registry()
        admitted = reg.assign(self._nodes(*"abcdefgh"))
        objects = [n for n in admitted.values() if n.node_type == "object"]
        self.assertEqual(len(objects), N_MAX - 1)


class SlotAlignment(unittest.TestCase):
    """The [B, N, S] map the RSSM builds, exercised without the model."""

    def setUp(self):
        try:
            import jax  # noqa: F401
        except ImportError:
            self.skipTest("jax not installed")

    def _align(self, obs_uid, slot_uid, slot_mask):
        import jax.numpy as jnp
        from dreamerv3.rssm import align_slots
        align, matched, fresh = align_slots(
            jnp.asarray([obs_uid], jnp.int32),
            jnp.asarray([slot_uid], jnp.int32),
            jnp.asarray([slot_mask], bool))
        return np.asarray(align[0]), np.asarray(matched[0]), np.asarray(fresh[0])

    def test_a_fresh_episode_seats_the_ee_in_slot_zero(self):
        align, _, fresh = self._align(
            [1, 7, 0, 0, 0, 0], [0] * 6, [False] * 6)
        self.assertTrue(align[0, 0])
        self.assertTrue(fresh[0])

    def test_a_known_uid_returns_to_its_slot(self):
        align, matched, fresh = self._align(
            [1, 9, 7, 0, 0, 0], [1, 7, 9, 0, 0, 0],
            [True, True, True, False, False, False])
        self.assertTrue(align[1, 2])
        self.assertTrue(align[2, 1])
        self.assertTrue(matched[1] and matched[2])
        self.assertFalse(fresh.any())

    def test_packed_order_does_not_move_a_slot(self):
        first, _, _ = self._align(
            [1, 7, 9, 0, 0, 0], [1, 7, 9, 0, 0, 0],
            [True, True, True, False, False, False])
        second, _, _ = self._align(
            [1, 9, 7, 0, 0, 0], [1, 7, 9, 0, 0, 0],
            [True, True, True, False, False, False])
        self.assertTrue(first[1, 1] and second[2, 1])

    def test_a_missing_uid_leaves_its_slot_unmatched(self):
        _, matched, _ = self._align(
            [1, 7, 0, 0, 0, 0], [1, 7, 9, 0, 0, 0],
            [True, True, True, False, False, False])
        self.assertTrue(matched[1])
        self.assertFalse(matched[2])

    def test_a_new_uid_takes_the_first_free_object_slot(self):
        align, _, fresh = self._align(
            [1, 7, 4, 0, 0, 0], [1, 7, 0, 0, 0, 0],
            [True, True, False, False, False, False])
        self.assertTrue(align[2, 2])
        self.assertTrue(fresh[2])
        self.assertFalse(fresh[1])

    def test_padding_vertices_claim_nothing(self):
        align, _, _ = self._align(
            [1, 0, 0, 0, 0, 0], [1, 7, 0, 0, 0, 0],
            [True, True, False, False, False, False])
        self.assertFalse(align[1:].any())


if __name__ == "__main__":
    unittest.main()
