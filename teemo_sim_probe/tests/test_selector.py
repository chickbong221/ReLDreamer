"""Vertex registry and episode-scoped persistence.

The registry is append-only within an episode, so a node that leaves and
re-enters the view keeps its index.
"""

import unittest

import numpy as np

from teemo_sim_probe.core.schema import Node
from teemo_sim_probe.core.selector import EntityRegistry, NodeSelector


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

    def test_overflow_keeps_the_higher_priority_role(self):
        reg = EntityRegistry(n_max=2)  # ee plus one object
        reg.assign({"ee": _ee(), "sup": _node("sup", roles=("support",))})
        out = reg.assign({
            "ee": _ee(),
            "sup": _node("sup", roles=("support",)),
            "tgt": _node("tgt", roles=("interacted",)),
        })
        self.assertIn("tgt", out)
        self.assertNotIn("sup", out)

    def test_overflow_drops_the_lower_priority_newcomer(self):
        reg = EntityRegistry(n_max=2)
        reg.assign({"ee": _ee(), "tgt": _node("tgt", roles=("interacted",))})
        out = reg.assign({
            "ee": _ee(),
            "tgt": _node("tgt", roles=("interacted",)),
            "sup": _node("sup", roles=("support",)),
        })
        self.assertNotIn("sup", out)
        self.assertEqual(reg.overflow_drops, 1)


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
