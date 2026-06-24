"""Load thresholds.yaml and flatten the chosen scale profile into ``cfg``.

By default both mined assets are REQUIRED (R4 + R6 paper-aligned). Pass
``require_assets=False`` to bypass — only useful for unit tests that wire
their own minimal config.
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

from ..core.affordance import load_affordance_set
from ..core.e_domain import load_e_domain


_MISSING_AFFORDANCE_MSG = (
    "Affordance asset missing or empty at {path!r}.\n"
    "Mine it before running the probe:\n"
    "  python -m teemo_sim_probe.tools.build_affordances \\\n"
    "      --success-states-dir \"$MS_ASSET_DIR/robot_success_states\" \\\n"
    "      --robot fetch --subtask pick \\\n"
    "      --out teemo_sim_probe/configs/affordances.json"
)

_MISSING_EDOMAIN_MSG = (
    "E_domain asset missing or empty at {path!r}.\n"
    "Mine it before running the probe:\n"
    "  python -m teemo_sim_probe.tools.build_e_domain \\\n"
    "      --task-plans-dir \"$MS_ASSET_DIR/scene_datasets/replica_cad_dataset/rearrange/task_plans\" \\\n"
    "      --success-states-dir \"$MS_ASSET_DIR/robot_success_states\" \\\n"
    "      --out teemo_sim_probe/configs/e_domain.json"
)


def _abs_asset_path(cfg_dir: str, rel: Optional[str]) -> Optional[str]:
    if not rel:
        return None
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(cfg_dir, rel))


def load_config(
    profile: str,
    path: Optional[str] = None,
    *,
    require_assets: bool = True,
) -> dict:
    """profile in {"tabletop", "room_scale"}."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "thresholds.yaml")
    cfg_dir = os.path.dirname(os.path.abspath(path))

    with open(path) as f:
        raw = yaml.safe_load(f)

    if profile not in raw["profiles"]:
        raise ValueError(
            f"unknown profile {profile!r}; have {list(raw['profiles'])}"
        )

    affordances_cfg = dict(raw.get("affordances", {"asset_path": "affordances.json"}))
    affordances_cfg["asset_path_abs"] = _abs_asset_path(
        cfg_dir, affordances_cfg.get("asset_path")
    )

    e_domain_cfg = dict(raw.get("e_domain", {
        "asset_path": "e_domain.json",
        "enable_local_contact": True,
    }))
    e_domain_cfg["asset_path_abs"] = _abs_asset_path(
        cfg_dir, e_domain_cfg.get("asset_path")
    )

    selection_cfg = raw.get("selection", {
        "n_slots": 10, "n_refresh": 2, "k_persist": 5,
        "oracle_force_active_target": False,
        "weights": {
            "contact": 4.0, "grasp": 4.0, "persist": 2.0, "afford": 1.5,
            "state": 1.5, "support": 1.0, "local": 2.0, "dist": 2.0,
        },
        "tau_age": 3, "tau_dist_tabletop": 0.5, "tau_dist_room_scale": 1.5,
    })

    aff_set = load_affordance_set(affordances_cfg["asset_path_abs"])
    e_dom = load_e_domain(e_domain_cfg["asset_path_abs"])

    if require_assets:
        if aff_set.is_empty():
            raise FileNotFoundError(
                _MISSING_AFFORDANCE_MSG.format(path=affordances_cfg["asset_path_abs"])
            )
        if e_dom.empty:
            raise FileNotFoundError(
                _MISSING_EDOMAIN_MSG.format(path=e_domain_cfg["asset_path_abs"])
            )

    cfg = {
        "temporal": raw["temporal"],
        "contact": raw["contact"],
        "grasp": raw["grasp"],
        "support": raw["support"],
        "affordances": affordances_cfg,
        "affordance_set": aff_set,
        "e_domain": e_domain_cfg,
        "e_domain_set": e_dom,
        "selection": selection_cfg,
        "profile": raw["profiles"][profile],
        "profile_name": profile,
        "node": raw.get("node", {"min_pixels": 32, "min_area_ratio": 0.0005}),
    }
    return cfg
