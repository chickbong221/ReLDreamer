"""Load thresholds.yaml and flatten the chosen scale profile into ``cfg``."""

from __future__ import annotations

import os
from typing import Optional

import yaml

from ..core.affordance import load_affordance_set


def load_config(profile: str, path: Optional[str] = None) -> dict:
    """profile in {"tabletop", "room_scale"}.

    ``path`` defaults to the packaged ``thresholds.yaml``. The ``affordances``
    asset path is resolved relative to the threshold file's directory so that
    a caller passing a custom ``path`` gets a sibling-relative lookup.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "thresholds.yaml")
    cfg_dir = os.path.dirname(os.path.abspath(path))

    with open(path) as f:
        raw = yaml.safe_load(f)

    if profile not in raw["profiles"]:
        raise ValueError(
            f"unknown profile {profile!r}; have {list(raw['profiles'])}"
        )

    affordances_cfg = raw.get("affordances", {"asset_path": "affordances.json"})
    asset_path_rel = affordances_cfg.get("asset_path", "affordances.json")
    if asset_path_rel and not os.path.isabs(asset_path_rel):
        asset_path_abs = os.path.normpath(os.path.join(cfg_dir, asset_path_rel))
    else:
        asset_path_abs = asset_path_rel
    affordances_cfg = dict(affordances_cfg)
    affordances_cfg["asset_path_abs"] = asset_path_abs

    cfg = {
        "temporal": raw["temporal"],
        "contact": raw["contact"],
        "grasp": raw["grasp"],
        "support": raw["support"],
        "affordances": affordances_cfg,
        "affordance_set": load_affordance_set(asset_path_abs),
        "persistence": raw.get("persistence", {
            "n_max": 6, "w_keep": 3, "w_manip": 5,
        }),
        "profile": raw["profiles"][profile],
        "node": raw.get("node", {"min_pixels": 32, "min_area_ratio": 0.0005}),
        "profile_name": profile,
    }
    return cfg
