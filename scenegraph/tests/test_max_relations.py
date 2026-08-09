"""``e_max`` must hold one instance of every whitelist member.

Truncation is silent: the packer drops the overflow into
``log/graph_fact_drops`` and the run continues on an incomplete graph. Those
members appear together whenever a scene shows them, so this floor is pinned
here. The all-duplicates ceiling above it is deliberately not enforced --
``e_max`` is sized from the smoke run's measured peak, and overflow past that
degrades gracefully by dropping spatial facts first.
"""

import os
import unittest

import yaml

from scenegraph.configs.loader import load_config
from scenegraph.core.whitelist import load_whitelist
from scenegraph.tools.build_union_whitelist import UNION_TARGET
from scenegraph.tools.max_relations import ceiling

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WHITELISTS = os.path.join(REPO, "scenegraph", "configs", "subtask_whitelists")
AFFORDANCES = os.path.join(REPO, "scenegraph", "configs", "affordances.json")
CONFIGS = os.path.join(REPO, "dreamerv3", "configs.yaml")


def _configured(key):
    with open(CONFIGS) as f:
        raw = yaml.safe_load(f)
    return int(raw["defaults"]["env"]["maniskill"]["graph"][key])


def _bound_whitelists():
    """Every whitelist an episode can bind for membership."""
    for name in sorted(os.listdir(WHITELISTS)):
        if not name.endswith(".json"):
            continue
        wl = load_whitelist(os.path.join(WHITELISTS, name))
        if wl.target != UNION_TARGET:
            yield name, wl


@unittest.skipUnless(
    os.path.isdir(WHITELISTS) and os.path.isfile(AFFORDANCES),
    "mined assets are absent")
class MaxRelationTests(unittest.TestCase):

    def test_every_member_fits_at_once(self):
        cfg = load_config("room_scale")
        n_max, e_max = _configured("n_max"), _configured("e_max")
        over = {
            name: int(ceiling(wl, cfg, n_max)["distinct_total"])
            for name, wl in _bound_whitelists()
        }
        self.assertTrue(over, "no per-target whitelists to check")
        over = {k: v for k, v in over.items() if v > e_max}
        self.assertFalse(
            over,
            f"e_max {e_max} cannot hold one instance of every member: {over}")

    def test_every_member_set_fits_n_max(self):
        n_max = _configured("n_max")
        over = {
            name: len(wl.by_key) + 1
            for name, wl in _bound_whitelists()
            if len(wl.by_key) + 1 > n_max
        }
        self.assertFalse(
            over,
            f"n_max {n_max} cannot hold {over} (ee plus every member), so the "
            "registry drops a vertex before any duplicate instance appears")
