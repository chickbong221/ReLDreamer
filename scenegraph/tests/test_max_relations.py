"""A graph filled to ``n_max`` must still fit ``e_max``.

Truncation is silent: the packer drops the overflow into
``log/graph_fact_drops`` and the run continues on an incomplete graph, so the
ceiling is pinned here rather than noticed in a training curve.
"""

import os
import unittest

import yaml

from scenegraph.configs.loader import load_config
from scenegraph.core.whitelist import load_whitelist
from scenegraph.tools.max_relations import ceiling

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WHITELISTS = os.path.join(REPO, "scenegraph", "configs", "subtask_whitelists")
AFFORDANCES = os.path.join(REPO, "scenegraph", "configs", "affordances.json")
CONFIGS = os.path.join(REPO, "dreamerv3", "configs.yaml")

# The union files supply bin edges only; no episode ever binds one for
# membership, so their fact count is not a shape the runtime has to hold.
UNION = "_all.json"


def _configured(key):
    with open(CONFIGS) as f:
        raw = yaml.safe_load(f)
    return int(raw["defaults"]["env"]["maniskill"]["graph"][key])


@unittest.skipUnless(
    os.path.isdir(WHITELISTS) and os.path.isfile(AFFORDANCES),
    "mined assets are absent")
class MaxRelationTests(unittest.TestCase):

    def test_a_full_graph_fits_e_max(self):
        """n_max and e_max are coupled: whatever headroom n_max grants for
        duplicate instances, e_max has to be able to pay for in pairs."""
        cfg = load_config("room_scale")
        n_max, e_max = _configured("n_max"), _configured("e_max")
        over = {}
        for name in sorted(os.listdir(WHITELISTS)):
            if not name.endswith(".json") or name.endswith(UNION):
                continue
            wl = load_whitelist(os.path.join(WHITELISTS, name))
            total = int(ceiling(wl, cfg, n_max)["total"])
            if total > e_max:
                over[name] = total
        self.assertFalse(
            over,
            f"a full {n_max}-vertex graph emits {over}, above e_max {e_max}; "
            "raise e_max or lower n_max (the pair term is quadratic in it)")
