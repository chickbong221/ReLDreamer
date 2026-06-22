"""CLI entry for the SAC trainer.

Examples:
    python -m sac.main --configs maniskill_state --task maniskill_PickCube-v1
    python -m sac.main --configs maniskill_rgb   --task maniskill_PickCube-v1
    python -m sac.main --configs mshab           --task maniskill_PickSubtaskTrain-v0 \
                       --env.maniskill.mshab_task pick

The structure mirrors dreamerv3/main.py: a single configs.yaml with a
``defaults`` block plus named presets selected by ``--configs <name>``, and
arbitrary dotted overrides for individual keys.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from copy import deepcopy
from typing import List, Tuple

import yaml


HERE = pathlib.Path(__file__).parent


# --------------------------------------------------------------------------- #
# Config merging
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into ``base`` (in-place on a copy)."""
    out = deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(value: str):
    """Best-effort string-to-scalar conversion for CLI override values."""
    lower = value.lower()
    if lower in ("true", "yes", "on"):
        return True
    if lower in ("false", "no", "off"):
        return False
    if lower in ("null", "none"):
        return None
    try:
        if "." in value or "e" in lower:
            return float(value)
        return int(value)
    except ValueError:
        pass
    return value


def _set_dotted(d: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _parse_argv(argv: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Return (presets, overrides). Overrides are (dotted_key, raw_value)."""
    presets: List[str] = []
    overrides: List[Tuple[str, str]] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            raise ValueError(f"Unexpected positional argument: {tok!r}")
        key = tok[2:]
        if i + 1 >= len(argv):
            raise ValueError(f"Missing value for --{key}")
        val = argv[i + 1]
        i += 2
        if key == "configs":
            presets += [s for s in val.split(",") if s]
        else:
            overrides.append((key, val))
    return presets, overrides


def load_config(argv: List[str]) -> dict:
    raw = yaml.safe_load((HERE / "configs.yaml").read_text())
    config = deepcopy(raw["defaults"])
    presets, overrides = _parse_argv(argv)
    for name in presets:
        if name not in raw:
            raise KeyError(f"Unknown preset: {name!r}")
        config = _deep_merge(config, raw[name])
    for key, value in overrides:
        _set_dotted(config, key, _coerce(value))

    # Resolve {timestamp} in logdir.
    ts = time.strftime("%Y%m%d_%H%M%S")
    config["logdir"] = str(config["logdir"]).format(timestamp=ts)
    os.makedirs(config["logdir"], exist_ok=True)

    # Snapshot the resolved config alongside the checkpoints.
    with open(os.path.join(config["logdir"], "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return config


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: List[str] | None = None) -> None:
    # Make ``sac`` importable as a top-level package when run via -m or direct.
    sys.path.insert(0, str(HERE.parent))

    argv = list(sys.argv[1:] if argv is None else argv)
    config = load_config(argv)

    print(f"[sac] logdir: {config['logdir']}")
    print(f"[sac] task:   {config['task']}")
    print(f"[sac] seed:   {config['seed']}")

    if config.get("script", "train") != "train":
        raise NotImplementedError(
            f"sac only supports script=train, got {config['script']!r}")

    from .run.train import train
    train(config)


if __name__ == "__main__":
    main()
