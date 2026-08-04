"""Vertex registry and appearance retention.

The registry is append-only within an episode, so a node that leaves and
re-enters the view keeps its index; the selector is the single owner of the
per-camera appearance a node retains while invisible.
"""

import unittest

from teemo_sim_probe.core.schema import Node
from teemo_sim_probe.core.selector import EntityRegistry, NodeSelector


def _node(node_id, roles=("interacted",), feat=None, visible=True):
    return Node(
        node_id=node_id, node_type="object", name=node_id, visible=visible,
        feat=list(feat) if feat is not None else None,
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


class AppearanceRetentionTests(unittest.TestCase):

    def test_unseen_camera_keeps_its_last_value(self):
        sel = _selector()
        sel.update_feats({"a": _node("a", feat=[1.5, 2.5])}, 2)
        node = _node("a", feat=[None, 9.0])
        sel.update_feats({"a": node}, 2)
        self.assertEqual(node.feat, [1.5, 9.0])

    def test_never_seen_defaults_to_zero(self):
        sel = _selector()
        node = _node("a", feat=[None, None])
        sel.update_feats({"a": node}, 2)
        self.assertEqual(node.feat, [0.0, 0.0])

    def test_invisible_node_retains_appearance_across_frames(self):
        sel = _selector()
        visible = _node("a", feat=[0.8, 1.2])
        sel.update_feats({"a": visible}, 2)
        sel.commit({"a": visible}, frame=0)
        merged = sel.merge_persistent({}, frame=5)
        self.assertIn("a", merged)
        self.assertFalse(merged["a"].visible)
        self.assertEqual(merged["a"].feat, [0.8, 1.2])

    def test_k_persist_negative_never_evicts(self):
        sel = _selector(k_persist=-1)
        node = _node("a", feat=[1.0, 1.0])
        sel.update_feats({"a": node}, 2)
        sel.commit({"a": node}, frame=0)
        self.assertEqual(sel.evict_expired(frame=10_000), [])
        self.assertIn("a", sel.merge_persistent({}, frame=10_000))


if __name__ == "__main__":
    unittest.main()
