"""Tests for NodeSelector + SlotManager (R7-R13)."""

from __future__ import annotations

import unittest
from typing import Dict

import numpy as np

from teemo_sim_probe.core.affordance import AffordanceSet, AffordanceComponent
from teemo_sim_probe.core.e_domain import EDomain
from teemo_sim_probe.core.schema import Node
from teemo_sim_probe.core.selector import NodeSelector
from teemo_sim_probe.core.slot_manager import SlotManager


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


def _cfg(weights=None, n_slots=10, n_refresh=2, k_persist=5,
         tau_dist=1.5, tau_age=3, oracle=False,
         enable_local=True, aff_set=None):
    return {
        "contact": {"eps_force": 0.05},
        "grasp": {"max_angle": 30},
        "affordance_set": aff_set if aff_set is not None else AffordanceSet(),
        "e_domain": {"enable_local_contact": enable_local},
        "selection": {
            "n_slots": n_slots, "n_refresh": n_refresh, "k_persist": k_persist,
            "oracle_force_active_target": oracle,
            "weights": weights or {
                "contact": 4.0, "grasp": 4.0, "persist": 2.0, "afford": 1.5,
                "state": 1.5, "support": 1.0, "local": 2.0, "dist": 2.0,
            },
            "tau_age": tau_age,
            "tau_dist_tabletop": tau_dist,
            "tau_dist_room_scale": tau_dist,
        },
    }


def _ee(xyz=(0.0, 0.0, 0.0)):
    return Node(node_id="ee", node_type="ee", name="end_effector",
                pose_world=[*xyz, 1.0, 0.0, 0.0, 0.0])


def _obj(name, xyz, *, attrs=None, seg_id=None) -> Node:
    return Node(
        node_id=f"actor:{name}",
        node_type="object", name=name,
        pose_world=[*xyz, 1.0, 0.0, 0.0, 0.0],
        attributes=dict(attrs or {}),
        segmentation_ids=[seg_id] if seg_id is not None else [],
    )


# --------------------------------------------------------------------------- #
# Score (R10)
# --------------------------------------------------------------------------- #
class ScoreTests(unittest.TestCase):
    def test_local_contact_beats_far_affordance(self):
        """Pinning the R10 weights against the local-physics-starvation mode."""
        aff = AffordanceSet(by_object={
            "far_bowl": [AffordanceComponent(np.zeros(3), 0.045)]
        })
        sel = NodeSelector(_cfg(aff_set=aff), EDomain(), "room_scale")
        e_near, e_far = _Ent("near_table"), _Ent("far_bowl")
        near = _obj("near_table", (0.05, 0.0, 0.0), seg_id=1)
        far = _obj("far_bowl", (2.0, 0.0, 0.0), seg_id=2)
        state = _State(contacts={e_near})
        state.seg_id_map = {1: e_near, 2: e_far}
        scores = sel.score(
            {"ee": _ee(), "actor:near_table": near, "actor:far_bowl": far}, state
        )
        self.assertGreater(scores["actor:near_table"], scores["actor:far_bowl"])

    def test_distance_decay_breaks_ties(self):
        sel = NodeSelector(_cfg(), EDomain(), "room_scale")
        scores = sel.score(
            {"ee": _ee(),
             "actor:a": _obj("a", (0.1, 0.0, 0.0)),
             "actor:b": _obj("b", (1.0, 0.0, 0.0))},
            _State(),
        )
        self.assertGreater(scores["actor:a"], scores["actor:b"])


# --------------------------------------------------------------------------- #
# Refresh quota (R11)
# --------------------------------------------------------------------------- #
class RefreshTests(unittest.TestCase):
    def test_refresh_reserves_slots_for_new_candidates(self):
        sel = NodeSelector(_cfg(n_slots=4, n_refresh=2), EDomain(), "tabletop")
        scores = {
            "old1": 10.0, "old2": 9.0, "old3": 8.0, "old4": 7.0,
            "new1": 6.5, "new2": 6.0, "new3": 5.5, "new4": 5.0,
        }
        prev = {"old1", "old2", "old3", "old4"}
        out = sel.topk_with_refresh(scores, prev)
        self.assertEqual(len(out), 4)
        # n_keep=2 -> top 2 sticky win, the other 2 slots reserved for new.
        self.assertEqual(set(out), {"old1", "old2", "new1", "new2"})

    def test_no_refresh_when_no_new_candidates(self):
        sel = NodeSelector(_cfg(n_slots=3, n_refresh=2), EDomain(), "tabletop")
        scores = {"a": 3.0, "b": 2.0, "c": 1.0}
        out = sel.topk_with_refresh(scores, {"a", "b", "c"})
        self.assertEqual(set(out), {"a", "b", "c"})

    def test_zero_refresh_falls_back_to_topk(self):
        sel = NodeSelector(_cfg(n_slots=3, n_refresh=0), EDomain(), "tabletop")
        scores = {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.5}
        out = sel.topk_with_refresh(scores, prev_selected=set())
        self.assertEqual(out, ["a", "b", "c"])


# --------------------------------------------------------------------------- #
# Persistence window (R8 / R13)
# --------------------------------------------------------------------------- #
class PersistenceTests(unittest.TestCase):
    def test_invisible_node_kept_within_k(self):
        sel = NodeSelector(_cfg(k_persist=3), EDomain(), "tabletop")
        n0 = _obj("bowl", (0.1, 0.0, 0.0))
        sel.commit(["actor:bowl"], {"actor:bowl": n0}, frame=0)
        # frame=3 -> age==3, still within window.
        merged = sel.merge_persistent({}, frame=3)
        self.assertIn("actor:bowl", merged)
        self.assertTrue(merged["actor:bowl"].persistent)
        self.assertTrue(merged["actor:bowl"].frozen_pose)
        # frame=5 -> age==5 > k=3, evicted.
        self.assertNotIn("actor:bowl", sel.merge_persistent({}, frame=5))

    def test_k_persist_zero_disables_persistence(self):
        sel = NodeSelector(_cfg(k_persist=0), EDomain(), "tabletop")
        sel.commit(["actor:bowl"], {"actor:bowl": _obj("bowl", (0.1, 0, 0))},
                   frame=0)
        self.assertEqual(sel.merge_persistent({}, frame=1), {})


# --------------------------------------------------------------------------- #
# SlotManager (R12)
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

    def test_reset_episode_clears_slots(self):
        sm = SlotManager(n_slots=3)
        sm.assign(["a", "b", "c"])
        sm.reset_episode()
        out = sm.assign(["x", "y", "z"])
        self.assertEqual(out["x"].slot_id, 0)
        self.assertFalse(out["x"].reset_flag)

    def test_caps_to_n_slots(self):
        sm = SlotManager(n_slots=2)
        out = sm.assign(["a", "b", "c", "d"])
        self.assertEqual(len(out), 2)


# --------------------------------------------------------------------------- #
# Local-contact exception (R7)
# --------------------------------------------------------------------------- #
class LocalContactTests(unittest.TestCase):
    def test_adds_ee_touching_non_domain_entity(self):
        sel = NodeSelector(_cfg(), EDomain(), "tabletop")
        ent = _Ent("scratchpad")
        state = _State(contacts={ent})
        state.seg_id_map = {7: ent}
        out = sel.expand_local_contact({"ee": _ee()}, state, prev_selected=set())
        added = [n for n in out.values()
                 if n.attributes.get("is_local_contact")]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].name, "scratchpad")

    def test_disabled_by_flag(self):
        sel = NodeSelector(_cfg(enable_local=False), EDomain(), "tabletop")
        ent = _Ent("scratchpad")
        state = _State(contacts={ent})
        state.seg_id_map = {7: ent}
        out = sel.expand_local_contact({"ee": _ee()}, state, set())
        self.assertNotIn("local:scratchpad", out)


# --------------------------------------------------------------------------- #
# Oracle ablation (R13)
# --------------------------------------------------------------------------- #
class OracleTests(unittest.TestCase):
    def test_oracle_forces_active_target_top_rank(self):
        sel = NodeSelector(_cfg(oracle=True), EDomain(), "tabletop")
        active = _obj("target", (10.0, 0.0, 0.0),
                      attrs={"is_mshab_active_target": True})
        near = _obj("near", (0.05, 0.0, 0.0))
        scores = sel.score(
            {"ee": _ee(), "actor:target": active, "actor:near": near}, _State()
        )
        self.assertGreater(scores["actor:target"], scores["actor:near"])


if __name__ == "__main__":
    unittest.main()
