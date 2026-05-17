#!/usr/bin/env python3
"""
Sequential pretraining orchestrator for DreamerV3 + ManiSkill.

Runs DreamerV3 training phases in order, handing the checkpoint from each
phase to the next. After all training is done, runs eval_multitask.py to
produce a per-task metrics table.

Usage (from repo root):
  python sequential/train_sequential.py --config sequential/config.yaml
  python sequential/train_sequential.py --config sequential/config.yaml --run_baseline
"""

import argparse
import pathlib
import subprocess
import sys

import ruamel.yaml as yaml

# sequential/ lives one level below the repo root.
REPO = pathlib.Path(__file__).parent.parent
MAIN = REPO / 'dreamerv3' / 'main.py'
EVAL = REPO / 'sequential' / 'eval_multitask.py'


def _run(argv, label):
    print(f'\n{"=" * 60}')
    print(f'  {label}')
    print(f'{"=" * 60}\n')
    subprocess.run(argv, cwd=str(REPO), check=True)


def _train_argv(task, logdir, dreamer_configs, steps, from_checkpoint, extra_flags):
    argv = [
        sys.executable, str(MAIN),
        '--configs', *dreamer_configs,
        '--task', f'maniskill_{task}',
        '--logdir', str(logdir),
        '--run.steps', str(int(steps)),
    ]
    if from_checkpoint:
        argv += ['--run.from_checkpoint', str(from_checkpoint)]
    argv += extra_flags
    return argv


def main():
    parser = argparse.ArgumentParser(
        description='Sequential DreamerV3 pretraining + transfer evaluation.')
    parser.add_argument('--config', required=True,
                        help='Path to sequential YAML config '
                             '(e.g. sequential/config.yaml)')
    parser.add_argument('--run_baseline', action='store_true',
                        help='Also train Phase 2 from scratch for comparison '
                             '(overrides run_baseline: false in config)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.YAML(typ='safe').load(f)

    logdir_base      = pathlib.Path(cfg['logdir_base']).expanduser()
    phase1_tasks     = cfg['phase1_tasks']
    phase2_task      = cfg['phase2_task']
    steps_per_phase1 = cfg.get('steps_per_phase1_task', 2_000_000)
    phase2_steps     = cfg.get('phase2_steps', 2_000_000)
    dreamer_configs  = cfg.get('dreamer_configs', ['maniskill_rgb'])
    extra_flags      = cfg.get('extra_flags', [])
    run_baseline     = args.run_baseline or cfg.get('run_baseline', False)
    eval_episodes    = cfg.get('eval_episodes', 5)
    eval_envs        = cfg.get('eval_envs', 4)
    wandb_project    = cfg.get('wandb_project', '')
    wandb_entity     = cfg.get('wandb_entity', '')

    # ── Phase 1: sequential pretraining ─────────────────────────────────────
    prev_ckpt = None
    for task in phase1_tasks:
        task_logdir = logdir_base / 'phase1' / task
        _run(
            _train_argv(task, task_logdir, dreamer_configs,
                        steps_per_phase1, prev_ckpt, extra_flags),
            f'Phase 1 | task: {task} | from: {prev_ckpt or "scratch"}',
        )
        prev_ckpt = task_logdir / 'ckpt'

    phase1_final_ckpt = prev_ckpt  # checkpoint after last Phase 1 task

    # ── Phase 2a: pretrained branch ─────────────────────────────────────────
    pretrained_logdir = logdir_base / 'phase2' / 'pretrained'
    _run(
        _train_argv(phase2_task, pretrained_logdir, dreamer_configs,
                    phase2_steps, phase1_final_ckpt, extra_flags),
        f'Phase 2 (pretrained) | task: {phase2_task} | from: {phase1_final_ckpt}',
    )

    # ── Phase 2b: baseline branch (optional, default off) ───────────────────
    if run_baseline:
        baseline_logdir = logdir_base / 'phase2' / 'baseline'
        _run(
            _train_argv(phase2_task, baseline_logdir, dreamer_configs,
                        phase2_steps, None, extra_flags),
            f'Phase 2 (baseline / from scratch) | task: {phase2_task}',
        )

    # ── Post-training evaluation ─────────────────────────────────────────────
    # Evaluates the pretrained Phase 2 checkpoint on all Phase 1 tasks + Phase 2 task.
    eval_tasks = phase1_tasks + [phase2_task]

    def _eval_argv(ckpt_path, result_logdir, run_name):
        argv = [
            sys.executable, str(EVAL),
            '--checkpoint', str(ckpt_path),
            '--train_task', phase2_task,
            '--tasks',      *eval_tasks,
            '--configs',    *dreamer_configs,
            '--episodes',   str(eval_episodes),
            '--eval_envs',  str(eval_envs),
            '--logdir',     str(result_logdir),
        ]
        if wandb_project:
            argv += ['--wandb_project', wandb_project,
                     '--wandb_name',    run_name]
        if wandb_entity:
            argv += ['--wandb_entity', wandb_entity]
        return argv

    _run(
        _eval_argv(
            pretrained_logdir / 'ckpt',
            logdir_base / 'eval' / 'pretrained',
            f'eval_pretrained_{phase2_task}',
        ),
        'Eval: pretrained checkpoint on all tasks',
    )

    if run_baseline:
        _run(
            _eval_argv(
                baseline_logdir / 'ckpt',
                logdir_base / 'eval' / 'baseline',
                f'eval_baseline_{phase2_task}',
            ),
            'Eval: baseline checkpoint on all tasks',
        )

    print(f'\n{"=" * 60}')
    print('  Sequential pipeline complete.')
    print(f'  Results: {logdir_base / "eval"}')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()
