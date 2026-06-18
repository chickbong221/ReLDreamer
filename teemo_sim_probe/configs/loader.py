"""Load thresholds.yaml and flatten the chosen scale profile into ``cfg``."""

from __future__ import annotations

import os
from typing import Optional

import yaml


def load_config(profile: str, path: Optional[str] = None) -> dict:
    """profile in {"tabletop", "room_scale"}."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "thresholds.yaml")
    with open(path) as f:
        raw = yaml.safe_load(f)

    if profile not in raw["profiles"]:
        raise ValueError(
            f"unknown profile {profile!r}; have {list(raw['profiles'])}"
        )

    cfg = {
        "temporal": raw["temporal"],
        "contact": raw["contact"],
        "grasp": raw["grasp"],
        "support": raw["support"],
        "profile": raw["profiles"][profile],
        "node": raw.get("node", {"min_pixels": 32, "min_area_ratio": 0.0005}),
        "profile_name": profile,
    }
    return cfg
