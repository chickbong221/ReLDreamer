"""Merge per-target whitelists into one ``<subtask>_all.json``.

A per-target whitelist answers "what may exist while picking this object". A
run over every object wants "what may exist in these scenes at all", so this
unions the members of every ``<subtask>_<target>.json`` in a directory.

Roles are unioned along with the members, which matters at runtime: every
object is ``interacted`` in its own file, so the union marks all of them that
way and the instance filter in ``NodeSelector.apply_whitelist`` has to be off
for a union to admit more than one object. ``graph.whitelist_union`` does both
together.

Bin edges are re-derived rather than copied. Each per-target file calibrates
its relation bins against the scenes that target appeared in; the union takes
the elementwise maximum of the observed statistics and runs the same
``derive_bin_edges`` the miner uses, so the merged file is calibrated for the
widest scene rather than for whichever target happened to be first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from scenegraph.core.whitelist import (
    WHITELIST_SCHEMA_VERSION,
    derive_bin_edges,
)

UNION_TARGET = 'all'


def _sources(whitelist_dir: Path, subtask: str) -> List[Path]:
    """Per-target files for one subtask, excluding a previous union."""
    return sorted(
        p for p in whitelist_dir.glob(f'{subtask}_*.json')
        if p.name != f'{subtask}_{UNION_TARGET}.json'
    )


def merge(whitelist_dir: Path, subtask: str) -> Dict:
    """Union of every per-target whitelist for ``subtask``."""
    members: Dict[str, Dict] = {}
    robust: Dict[str, float] = {}
    rollouts = 0
    sources = _sources(whitelist_dir, subtask)
    if not sources:
        raise FileNotFoundError(
            f'no {subtask}_*.json under {whitelist_dir}; mine the per-target '
            'whitelists first')

    for path in sources:
        raw = json.loads(path.read_text())
        rollouts += int(raw.get('_n_successful_rollouts') or 0)
        for key, entry in (raw.get('members') or {}).items():
            merged = members.setdefault(
                key, {'roles': set(), 'interaction_types': set(),
                      'supports': set(), 'kind': entry.get('kind')})
            merged['roles'] |= set(entry.get('roles') or ())
            merged['interaction_types'] |= set(entry.get('interaction_types') or ())
            merged['supports'] |= set(entry.get('supports') or ())
        for stat, value in (raw.get('bin_stats_robust') or {}).items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            robust[stat] = max(robust.get(stat, value), value)

    out_members = {}
    for key, entry in sorted(members.items()):
        out = {
            'roles': sorted(entry['roles']),
            'interaction_types': sorted(entry['interaction_types']),
            'kind': entry['kind'],
        }
        if entry['supports']:
            out['supports'] = sorted(entry['supports'])
        out_members[key] = out

    return {
        '_schema_version': WHITELIST_SCHEMA_VERSION,
        'subtask': subtask,
        'target': UNION_TARGET,
        'members': out_members,
        'bin_edges': derive_bin_edges(robust),
        'bin_stats_robust': robust,
        '_n_successful_rollouts': rollouts,
        '_merged_from': [p.name for p in sources],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--whitelist-dir', required=True)
    parser.add_argument('--subtask', nargs='+', default=['pick'])
    args = parser.parse_args(argv)

    whitelist_dir = Path(args.whitelist_dir)
    for subtask in args.subtask:
        try:
            data = merge(whitelist_dir, subtask)
        except FileNotFoundError as exc:
            print(f'[union] skip {subtask}: {exc}', file=sys.stderr)
            continue
        out = whitelist_dir / f'{subtask}_{UNION_TARGET}.json'
        out.write_text(json.dumps(data, indent=2, sort_keys=True))
        print(f'[union] wrote {out.name}: {len(data["members"])} members from '
              f'{len(data["_merged_from"])} targets')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
