"""Tests for whitelist selection, persistence, and slot assignment."""

from __future__ import annotations

import unittest
from typing import Dict

import numpy as np

from teemo_sim_probe.core.affordance import AffordanceSet, AffordanceComponent
from teemo_sim_probe.core.schema import Node
from teemo_sim_probe.core.selector import NodeSelector
from teemo_sim_probe.core.slot_manager import SlotManager
from teemo_sim_probe.core.whitelist import Whitelist


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class _Ent:
    """seg_id_map entry that knows its own name. Identity comparison only."""

    def __init__(self, name: str):
        self.name = name


class _State:
    def __init__(self, contacts=None, grasps=None):
        # contacts / grasps are sets of seg_id_map entities.
        self.contacts = set(contacts or [])
        self.grasps = set(grasps or [])
        self.seg_id_map: Dict[int, _Ent] = {}
        self.robot_links: set = set()
        self.active_obj = None
        self.active_handle_link = None
        self.env_idx = 0

    def ee_object_contact_force(self, ent):
        return 1.0 if ent in self.contacts else 0.0

    def is_grasping(self, ent, max_angle=30):
        return ent in self.grasps

    def pairwise_force(self, a, b):
        return 0.0

    def pairwise_force_vector(self, a, b):
        return np.zeros(3)


def _cfg(n_slots=10, k_persist=5, aff_set=None):
    return {
        "contact": {"eps_force": 0.05},
        "grasp": {"max_angle": 30},
        "affordance_set": aff_set if aff_set is not None else AffordanceSet(),
        "selection": {
            "n_slots": n_slots,
            "k_persist": k_persist,
        },
    }


def _ee(xyz=(0.0, 0.0, 0.0)):
    return Node(node_id="ee", node_type="ee", name="end_effector",
                pose_world=[*xyz, 1.0, 0.0, 0.0, 0.0])


def _obj(name, xyz, *, attrs=None, seg_id=None, node_id=None) -> Node:
    base_attrs = {"is_actor": True}
    if attrs:
        base_attrs.update(attrs)
    return Node(
        node_id=node_id or f"actor:{name}",
        node_type="object", name=name,
        pose_world=[*xyz, 1.0, 0.0, 0.0, 0.0],
        attributes=base_attrs,
        segmentation_ids=[seg_id] if seg_id is not None else [],
    )


def _link(name, xyz, *, seg_id=None) -> Node:
    return Node(
        node_id=f"link:{name}",
        node_type="object", name=name,
        pose_world=[*xyz, 1.0, 0.0, 0.0, 0.0],
        attributes={"is_link": True, "is_articulation_link": True},
        segmentation_ids=[seg_id] if seg_id is not None else [],
    )


def _whitelist(members):
    """Build a Whitelist from a {key: roles} mapping."""
    return Whitelist(
        subtask="pick",
        target="024_bowl",
        by_key={k: set(v) for k, v in members.items()},
    )


# --------------------------------------------------------------------------- #
# Whitelist gate
# --------------------------------------------------------------------------- #
class WhitelistGateTests(unittest.TestCase):
    def test_drops_off_whitelist_actor(self):
        sel = NodeSelector(_cfg())
        sel.set_whitelist(_whitelist({"actor:024_bowl": {"task"}}))
        bowl = _obj("env-0_024_bowl-3", (0.1, 0, 0))   # canonicalizes to 024_bowl
        scrap = _obj("env-0_999_phantom-1", (0.2, 0, 0))
        out = sel.apply_whitelist({"ee": _ee(), bowl.node_id: bowl, scrap.node_id: scrap})
        self.assertIn(bowl.node_id, out)
        self.assertNotIn(scrap.node_id, out)
        self.assertIn("ee", out)

    def test_link_specificity(self):
        """A whitelist with `drawer3` must NOT admit `drawer1`/`drawer2`."""
        sel = NodeSelector(_cfg())
        sel.set_whitelist(_whitelist({
            "actor:024_bowl": {"task"}, "link:drawer3": {"support"},
        }))
        bowl = _obj("env-0_024_bowl-3", (0.0, 0, 0))
        d1 = _link("drawer1", (0.1, 0, 0))
        d3 = _link("drawer3", (0.2, 0, 0))
        out = sel.apply_whitelist({
            "ee": _ee(), bowl.node_id: bowl, d1.node_id: d1, d3.node_id: d3,
        })
        self.assertIn(d3.node_id, out)
        self.assertNotIn(d1.node_id, out)

    def test_fails_loud_without_whitelist(self):
        sel = NodeSelector(_cfg())
        with self.assertRaises(RuntimeError):
            sel.apply_whitelist({"ee": _ee()})


# --------------------------------------------------------------------------- #
# Role-aware overflow determinism
# --------------------------------------------------------------------------- #
class OverflowTruncationTests(unittest.TestCase):
    def test_keeps_nearest_to_ee(self):
        sel = NodeSelector(_cfg(n_slots=2))
        sel.set_whitelist(_whitelist({
            "actor:a": {"task"}, "actor:b": {"task"}, "actor:c": {"task"},
        }))
        nodes = {
            "ee": _ee(),
            "actor:a": _obj("a", (0.1, 0, 0)),
            "actor:b": _obj("b", (0.5, 0, 0)),
            "actor:c": _obj("c", (1.0, 0, 0)),
        }
        out = sel.overflow_truncate(nodes)
        self.assertEqual(out, ["actor:a", "actor:b"])

    def test_tiebreak_by_node_id(self):
        sel = NodeSelector(_cfg(n_slots=2))
        # All three at identical distance -- tiebreak by node_id ascending.
        nodes = {
            "ee": _ee(),
            "actor:c": _obj("c", (0.1, 0, 0), node_id="actor:c"),
            "actor:a": _obj("a", (0.1, 0, 0), node_id="actor:a"),
            "actor:b": _obj("b", (0.1, 0, 0), node_id="actor:b"),
        }
        out = sel.overflow_truncate(nodes)
        self.assertEqual(out, ["actor:a", "actor:b"])

    def test_role_priority_precedes_distance(self):
        sel = NodeSelector(_cfg(n_slots=2))
        interacted = _obj(
            "interacted", (2.0, 0, 0),
            attrs={"whitelist_roles": ["interacted"]},
        )
        support = _obj(
            "support", (3.0, 0, 0), attrs={"whitelist_roles": ["support"]},
        )
        nearby = _obj(
            "nearby", (0.1, 0, 0), attrs={"whitelist_roles": []},
        )
        out = sel.overflow_truncate({
            "ee": _ee(), interacted.node_id: interacted, support.node_id: support,
            nearby.node_id: nearby,
        })
        self.assertEqual(out, [interacted.node_id, support.node_id])


# --------------------------------------------------------------------------- #
# Persistence horizon (Bug P)
# --------------------------------------------------------------------------- #
class PersistenceTests(unittest.TestCase):
    def test_visible_node_kept_within_k(self):
        sel = NodeSelector(_cfg(k_persist=3))
        n0 = _obj("bowl", (0.1, 0.0, 0.0))
        sel.commit({"actor:bowl": n0}, frame=0)
        merged = sel.merge_persistent({}, frame=3)
        self.assertIn("actor:bowl", merged)
        self.assertTrue(merged["actor:bowl"].persistent)
        self.assertTrue(merged["actor:bowl"].frozen_pose)
        # 5 > k=3 -> not merged.
        self.assertNotIn("actor:bowl", sel.merge_persistent({}, frame=5))

    def test_unselected_visible_node_still_persists(self):
        """Bug P: commit() snapshots every visible object, not only selected.

        This is the heart of the persistence horizon fix -- the old code
        evicted everything unselected each frame, collapsing the window to
        ~1 frame.
        """
        sel = NodeSelector(_cfg(k_persist=4))
        bowl = _obj("bowl", (0.1, 0.0, 0.0))
        # selected_ids is EMPTY but bowl is visible.
        sel.commit({"actor:bowl": bowl}, frame=0)
        merged = sel.merge_persistent({}, frame=2)
        self.assertIn("actor:bowl", merged)

    def test_evict_expired_drops_only_aged_out(self):
        sel = NodeSelector(_cfg(k_persist=3))
        sel.commit({"actor:bowl": _obj("bowl", (0.1, 0, 0))}, frame=0)
        sel.commit({"actor:cup":  _obj("cup",  (0.2, 0, 0))}, frame=2)
        expired = sel.evict_expired(frame=4)
        # bowl: age=4, > 3 -> expired. cup: age=2, kept.
        self.assertEqual(expired, ["actor:bowl"])

    def test_k_persist_zero_disables_persistence(self):
        sel = NodeSelector(_cfg(k_persist=0))
        sel.commit({"actor:bowl": _obj("bowl", (0.1, 0, 0))}, frame=0)
        self.assertEqual(sel.merge_persistent({}, frame=1), {})


# --------------------------------------------------------------------------- #
# SlotManager (unchanged behavior, kept for regression coverage)
# --------------------------------------------------------------------------- #
class SlotManagerTests(unittest.TestCase):
    def test_sticky_assignment(self):
        sm = SlotManager(n_slots=5)
        a1 = sm.assign(["a", "b", "c"])
        self.assertEqual(a1["a"].slot_id, 0)
        self.assertEqual(a1["b"].slot_id, 1)
        self.assertEqual(a1["c"].slot_id, 2)
        a2 = sm.assign(["b", "c", "a"])
        self.assertEqual(a2["a"].slot_id, 0)
        self.assertEqual(a2["b"].slot_id, 1)
        self.assertEqual(a2["c"].slot_id, 2)
        for sa in a2.values():
            self.assertFalse(sa.reset_flag)

    def test_reset_flag_on_identity_change(self):
        sm = SlotManager(n_slots=3)
        sm.assign(["a", "b", "c"])
        out = sm.assign(["a", "d", "c"])
        self.assertEqual(out["d"].slot_id, 1)
        self.assertTrue(out["d"].reset_flag)
        self.assertFalse(out["a"].reset_flag)
        self.assertFalse(out["c"].reset_flag)

    def test_caps_to_n_slots(self):
        sm = SlotManager(n_slots=2)
        out = sm.assign(["a", "b", "c", "d"])
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
