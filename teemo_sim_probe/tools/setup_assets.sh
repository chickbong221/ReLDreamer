#!/usr/bin/env bash
# One-shot setup of the two mined assets the probe REQUIRES at startup:
#   * teemo_sim_probe/configs/affordances.json   (R6)
#   * teemo_sim_probe/configs/e_domain.json      (R4)
#
# Runs the three remaining setup steps from teemo_sim_probe/README.md in order:
#   2) collect $MS_ASSET_DIR/robot_success_states/  (rollouts of per-obj SAC)
#   3) mine    teemo_sim_probe/configs/affordances.json
#   4) mine    teemo_sim_probe/configs/e_domain.json
#
# Assumes step 1 (HF checkpoint download) is already done. Re-runnable: step 2
# skips obj_ids whose pickles already have >= --n-success samples.
#
# Usage:
#   export MS_ASSET_DIR=/root/.maniskill/data
#   bash teemo_sim_probe/tools/setup_assets.sh [ckpt_root] [n_success] [n_components]
#
# Example:
#   bash teemo_sim_probe/tools/setup_assets.sh mshab_checkpoints 30 4

set -euo pipefail

CKPT_ROOT="${1:-mshab_checkpoints}"
N_SUCCESS="${2:-30}"
N_COMPONENTS="${3:-4}"

if [[ ! -d "${CKPT_ROOT}/rl" ]]; then
    echo "ERROR: ${CKPT_ROOT}/rl not found." >&2
    echo "       Pass the parent directory of 'rl/' as the first argument." >&2
    exit 1
fi

if [[ -z "${MS_ASSET_DIR:-}" ]]; then
    echo "ERROR: MS_ASSET_DIR is not set." >&2
    echo "       export MS_ASSET_DIR=/root/.maniskill/data  (or wherever)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CFG_DIR="teemo_sim_probe/configs"
AFFORD_JSON="${CFG_DIR}/affordances.json"
EDOMAIN_JSON="${CFG_DIR}/e_domain.json"

echo "================================================================"
echo " STEP 2 -- collect robot_success_states/"
echo "   MS_ASSET_DIR = ${MS_ASSET_DIR}"
echo "   ckpt root    = ${CKPT_ROOT}/rl"
echo "   target N     = ${N_SUCCESS} successes per obj"
echo "================================================================"
python -m teemo_sim_probe.tools.collect_robot_success_states \
    --ckpt-root "${CKPT_ROOT}/rl" \
    --n-success "${N_SUCCESS}"

echo ""
echo "================================================================"
echo " STEP 3 -- mine ${AFFORD_JSON} (R6)"
echo "================================================================"
python -m teemo_sim_probe.tools.build_affordances \
    --success-states-dir "${MS_ASSET_DIR}/robot_success_states" \
    --robot fetch \
    --subtask pick \
    --out "${AFFORD_JSON}" \
    --n-components "${N_COMPONENTS}"

echo ""
echo "================================================================"
echo " STEP 4 -- mine ${EDOMAIN_JSON} (R4)"
echo "================================================================"
python -m teemo_sim_probe.tools.build_e_domain \
    --task-plans-dir "${MS_ASSET_DIR}/scene_datasets/replica_cad_dataset/rearrange/task_plans" \
    --success-states-dir "${MS_ASSET_DIR}/robot_success_states" \
    --splits train \
    --out "${EDOMAIN_JSON}"

echo ""
echo "================================================================"
echo " Setup complete."
echo "   ${AFFORD_JSON}"
echo "   ${EDOMAIN_JSON}"
echo " The probe can now load_config() without FileNotFoundError."
echo "================================================================"
