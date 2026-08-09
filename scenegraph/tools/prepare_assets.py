"""Mine every offline asset a run needs, in one command.

Four stages, each a tool that still runs standalone: collect successful
rollouts, mine affordances, mine per-subtask whitelists, embed instructions.
Everything lands in ``scenegraph/configs/``, which is where the runtime reads
it from.

Start with ``--dry-run``: it prints the coverage report and every subcommand
without running any of them, and collection is measured in sim-hours.

``--clean`` deletes the previous artifacts first. The miners only write the
keys they mined, so without it a whitelist for an object no longer in the task
plans survives the rebuild and is still loadable at runtime.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / 'scenegraph' / 'configs'


def _plan_pairs(mshab_tasks, subtasks, splits, obj):
    """``{split: {(subtask, canonical)}}`` the task plans can produce."""
    from scenegraph.tools.build_instruction_embeddings import collect_pairs
    return collect_pairs(mshab_tasks, subtasks, splits, obj)


def _checkpointed_objects(ckpt_root: Path, subtasks: List[str]) -> Set[str]:
    """Objects the collector can drive, one per released per-object policy."""
    found = set()
    for subtask in subtasks:
        for pt in ckpt_root.glob(f'*/{subtask}/*/policy.pt'):
            found.add(pt.parent.name)
    return found


def _report(needed: Dict[str, Set[Tuple[str, str]]], ckpt_objects: Set[str],
            whitelist_dir: Path, table: Path) -> List[Tuple[str, str]]:
    """Print what the plans want against what exists. Returns uncollectable."""
    from scenegraph.core.whitelist import resolve_whitelist_path

    pairs = sorted(set().union(*needed.values())) if needed else []
    print(f'\n[prep] task plans name {len(pairs)} (subtask, object) pairs')

    have_table = set()
    if table.is_file():
        import numpy as np
        have_table = {str(k) for k in np.load(table, allow_pickle=False)['keys']}

    uncollectable = []
    for kind, canonical in pairs:
        wl = resolve_whitelist_path(str(whitelist_dir), kind, f'actor:{canonical}')
        marks = [
            'ckpt' if canonical in ckpt_objects else 'NO-CKPT',
            'wl' if wl else 'no-wl',
            'instr' if f'{kind}/actor:{canonical}' in have_table else 'no-instr',
        ]
        print(f'  {kind:6s} {canonical:28s} {" ".join(marks)}')
        if canonical not in ckpt_objects:
            uncollectable.append((kind, canonical))

    if uncollectable:
        print(f'\n[prep] {len(uncollectable)} object(s) have no per-object '
              'policy under --ckpt-root, so no rollouts can be collected for '
              'them and they will have no whitelist:')
        for kind, canonical in uncollectable:
            print(f'  {kind}/{canonical}')
    return uncollectable


def _clean(paths: List[Path], dry_run: bool) -> None:
    for path in paths:
        if not path.exists():
            continue
        print(f'  rm -r {path}')
        if dry_run:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _run(cmd: List[str], dry_run: bool) -> None:
    print('  $ ' + ' '.join(str(c) for c in cmd))
    if dry_run:
        return
    # -u because collection runs for hours and its output is usually piped to
    # a log: Python block-buffers stdout at 8KB when it is not a terminal, so
    # progress would arrive in bursts and anything still buffered when the
    # process is killed would never be written at all.
    code = subprocess.call([sys.executable, '-u', '-m', *cmd], cwd=REPO)
    if code != 0:
        raise SystemExit(f'[prep] stage failed with exit code {code}')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mshab-task', nargs='+', required=True)
    parser.add_argument('--subtask', nargs='+', default=['pick'])
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--obj', default='all')
    parser.add_argument('--ckpt-root', default='mshab_checkpoints/rl')
    parser.add_argument('--asset-dir', default=None,
                        help='data root holding robot_success_states/; '
                             'defaults to $MS_ASSET_DIR/data then ~/.maniskill/data')
    parser.add_argument('--robot', default='fetch')
    parser.add_argument('--n-success', type=int, default=30)
    parser.add_argument('--num-envs', type=int, default=35)
    parser.add_argument('--model', default='t5-base')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-collect', action='store_true')
    parser.add_argument('--skip-affordances', action='store_true')
    parser.add_argument('--skip-whitelists', action='store_true')
    parser.add_argument('--skip-instructions', action='store_true')
    args = parser.parse_args(argv)

    import os
    asset_dir = Path(args.asset_dir or (
        Path(os.environ.get('MS_ASSET_DIR', os.path.expanduser('~/.maniskill')))
        / 'data')).resolve()
    success_dir = asset_dir / 'robot_success_states'
    whitelist_dir = CONFIGS / 'subtask_whitelists'
    affordances = CONFIGS / 'affordances.json'
    table = CONFIGS / 'instructions.npz'
    control = CONFIGS / 'instructions_random.npz'
    ckpt_root = (REPO / args.ckpt_root).resolve()

    needed = _plan_pairs(args.mshab_task, args.subtask, args.splits, args.obj)
    ckpt_objects = _checkpointed_objects(ckpt_root, args.subtask)
    _report(needed, ckpt_objects, whitelist_dir, table)

    if args.clean:
        print('\n[prep] clean')
        targets = [whitelist_dir, affordances, table, control]
        targets += [success_dir / args.robot / s for s in args.subtask]
        _clean(targets, args.dry_run)

    if not args.skip_collect:
        print('\n[prep] collect')
        for subtask in args.subtask:
            cmd = [
                'scenegraph.tools.collect_robot_success_states',
                '--ckpt-root', str(ckpt_root),
                '--subtask', subtask,
                '--n-success', str(args.n_success),
                '--num-envs', str(args.num_envs),
                '--asset-dir', str(asset_dir),
                '--no-skip-done',
            ]
            # Without this the collector walks every task family under
            # ckpt-root. Objects are deduplicated by id, so the ones we need
            # cost the same, but a family we are not training on would be
            # collected too at n-success rollouts apiece.
            for task in args.mshab_task:
                cmd += ['--task', task]
            _run(cmd, args.dry_run)

    if not args.skip_affordances:
        print('\n[prep] affordances')
        for index, subtask in enumerate(args.subtask):
            cmd = [
                'scenegraph.tools.build_affordances',
                '--success-states-dir', str(success_dir),
                '--robot', args.robot,
                '--subtask', subtask,
                '--out', str(affordances),
            ]
            # The first subtask writes the file; the rest add to it.
            if index:
                cmd.append('--merge-existing')
            _run(cmd, args.dry_run)

    if not args.skip_whitelists:
        print('\n[prep] whitelists')
        _run([
            'scenegraph.tools.build_subtask_whitelists',
            '--success-states-dir', str(success_dir),
            '--out-dir', str(whitelist_dir),
            '--affordance-json', str(affordances),
        ], args.dry_run)

        # One merged file per subtask for runs whose target changes each
        # episode. Written from the per-target files, so it follows them.
        _run([
            'scenegraph.tools.build_union_whitelist',
            '--whitelist-dir', str(whitelist_dir),
            '--subtask', *args.subtask,
        ], args.dry_run)

    if not args.skip_instructions:
        print('\n[prep] instructions')
        common = [
            'scenegraph.tools.build_instruction_embeddings',
            '--mshab-task', *args.mshab_task,
            '--subtask', *args.subtask,
            '--splits', *args.splits,
            '--obj', args.obj,
        ]
        _run([*common, '--out', str(table), '--model', args.model], args.dry_run)
        _run([*common, '--out', str(control), '--random'], args.dry_run)

    if args.dry_run:
        print('\n[prep] dry run: nothing was written')
        return 0

    print('\n[prep] verify')
    missing = _report(needed, ckpt_objects, whitelist_dir, table)
    from scenegraph.core.whitelist import resolve_whitelist_path
    pairs = sorted(set().union(*needed.values())) if needed else []
    gaps = [
        f'{k}/{c}' for k, c in pairs
        if resolve_whitelist_path(str(whitelist_dir), k, f'actor:{c}') is None
    ]
    if gaps:
        print(f'[prep] FAILED: {len(gaps)} whitelist(s) still missing: '
              f'{", ".join(gaps)}', file=sys.stderr)
        return 1
    print('[prep] every (subtask, object) pair the task plans name is covered')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
