#!/usr/bin/env bash
set -euo pipefail

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_PushCube-v1 \
  --batch_size 16 \
  --logger.wandb_name dreamerv3-PushCube-v1-rgb-42-walltime_efficient

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_PickCube-v1 \
  --batch_size 16 \
  --logger.wandb_name dreamerv3-PickCube-v1-rgb-42-walltime_efficient

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_StackCube-v1 \
  --batch_size 16 \
  --logger.wandb_name dreamerv3-StackCube-v1-rgb-42-walltime_efficient

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_PegInsertionSide-v1 \
  --batch_size 16 \
  --logger.wandb_name dreamerv3-PegInsertionSide-v1-rgb-42-walltime_efficient

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_PushT-v1 \
  --batch_size 16 \
  --logger.wandb_name dreamerv3-PushT-v1-rgb-42-walltime_efficient

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_AnymalC-Reach-v1 \
  --batch_size 16 \
  --logger.wandb_name dreamerv3-AnymalC-Reach-v1-rgb-42-walltime_efficient

python -m dreamerv3.main \
  --configs maniskill_rgb \
  --task maniskill_UnitreeG1TransportBox-v1 \
  --batch_size 16 \
  --logger.wandb_name dreamerv3-UnitreeG1TransportBox-v1-rgb-42-walltime_efficient
