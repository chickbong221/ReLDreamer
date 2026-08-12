"""Vertex registry and episode-scoped persistence.

Indices stay stable until overflow; a new instance then takes the oldest
resident's index.
"""

import unittest

import numpy as np

from scenegraph.core.schema import Node
from scenegraph.core.selector import EntityRegistry, NodeSelector


def _node(node_id, roles=("interacted",), visible=True):
    return Node(
        node_id=node_id, node_type="object", name=node_id, visible=visible,
        attributes={"whitelist_roles": list(roles)},
    )


def _ee():
    return Node(node_id="ee", node_type="ee", name="end_effector")


def _selector(n_max=4, k_persist=-1):
    return NodeSelector({"selection": {"n_max": n_max, "k_persist": k_persist}})


class RegistryTests(unittest.TestCase):

    def test_ee_always_holds_index_zero(self):
        reg = EntityRegistry(n_max=4)
        out = reg.assign({"ee": _ee(), "a": _node("a")})
        self.assertEqual(out["ee"].index, 0)
        self.assertEqual(out["a"].index, 1)

    def test_index_survives_disappear_and_reappear(self):
        reg = EntityRegistry(n_max=4)
        reg.assign({"ee": _ee(), "a": _node("a"), "b": _node("b")})
        b_index = reg.index_of("b")
        reg.assign({"ee": _ee(), "a": _node("a")})
        again = reg.assign({"ee": _ee(), "a": _node("a"), "b": _node("b")})
        self.assertEqual(again["b"].index, b_index)

    def test_new_entity_appends_rather_than_reorders(self):
        reg = EntityRegistry(n_max=5)
        reg.assign({"ee": _ee(), "z": _node("z")})
        out = reg.assign({"ee": _ee(), "z": _node("z"), "a": _node("a")})
        self.assertEqual(out["z"].index, 1)
        self.assertEqual(out["a"].index, 2)

    def test_overflow_evicts_oldest_even_when_it_has_higher_priority(self):
        reg = EntityRegistry(n_max=2)  # ee plus one object
        reg.assign({"ee": _ee(), "tgt": _node("tgt", roles=("interacted",))})
        out = reg.assign({
            "ee": _ee(),
            "tgt": _node("tgt", roles=("interacted",)),
            "sup": _node("sup", roles=("support",)),
        })
        self.assertNotIn("tgt", out)
        self.assertIn("sup", out)
        self.assertEqual(reg.evicted_ids, ["tgt"])
        self.assertEqual(reg.overflow_drops, 1)

    def test_overflow_keeps_newest_instances_and_reuses_oldest_index(self):
        reg = EntityRegistry(n_max=3)  # ee plus two objects
        first = reg.assign({"ee": _ee(), "a": _node("a"), "b": _node("b")})
        a_index = first["a"].index
        b_index = first["b"].index
        out = reg.assign({
            "ee": _ee(),
            "a": _node("a"), "b": _node("b"), "c": _node("c"),
        })
        self.assertNotIn("a", out)
        self.assertEqual(out["b"].index, b_index)
        self.assertEqual(out["c"].index, a_index)
        self.assertEqual(reg.evicted_ids, ["a"])
        self.assertEqual(reg.overflow_drops, 1)

    def test_evicted_old_instance_does_not_rotate_back_in(self):
        reg = EntityRegistry(n_max=3)
        reg.assign({"ee": _ee(), "a": _node("a"), "b": _node("b")})
        reg.assign({
            "ee": _ee(), "a": _node("a"), "b": _node("b"), "c": _node("c"),
        })
        out = reg.assign({
            "ee": _ee(), "a": _node("a"), "b": _node("b"), "c": _node("c"),
        })
        self.assertNotIn("a", out)
        self.assertEqual(set(out), {"ee", "b", "c"})
        self.assertEqual(reg.evicted_ids, [])


class PersistenceTests(unittest.TestCase):

    def test_retained_node_comes_back_invisible_and_unsupported(self):
        sel = _selector()
        node = _node("a")
        node.bbox = np.array([[0.1, 0.2, 0.3, 0.4]] * 2, np.float32)
        node.appearance = np.ones((2, 8), np.float32)
        sel.commit({"a": node}, frame=0)
        merged = sel.merge_persistent({}, frame=5)
        self.assertIn("a", merged)
        self.assertFalse(merged["a"].visible)
        # The registry retains identity and index only. A retained node has no
        # camera support this frame, so the box goes to zero; its appearance
        # comes back from the adapter-level cache, not from here.
        self.assertIsNone(merged["a"].bbox)
        self.assertIsNone(merged["a"].appearance)

    def test_k_persist_negative_never_evicts(self):
        sel = _selector(k_persist=-1)
        node = _node("a")
        sel.commit({"a": node}, frame=0)
        self.assertEqual(sel.evict_expired(frame=10_000), [])
        self.assertIn("a", sel.merge_persistent({}, frame=10_000))


if __name__ == "__main__":
    unittest.main()
