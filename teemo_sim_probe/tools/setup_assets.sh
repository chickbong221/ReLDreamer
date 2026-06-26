#!/usr/bin/env bash
# One-shot setup of the mined assets the probe REQUIRES at startup:
#   * teemo_sim_probe/configs/affordances.json
#   * teemo_sim_probe/configs/subtask_whitelists/<subtask>_<target>.json
#
# Steps (in order):
#   2) collect schema-v4 successful interaction rollouts
#   3) mine    teemo_sim_probe/configs/affordances.json
#   4) mine    teemo_sim_probe/configs/subtask_whitelists/
#
# MS_ASSET_DIR follows the ManiSkill convention: it is the maniskill ROOT (parent
# of data/), NOT the data dir itself. ManiSkill internally resolves
# $MS_ASSET_DIR/data as its asset root (mani_skill.ASSET_DIR), so setting it
# with a trailing /data would cause a doubled "data/data/" segment.
#
# Usage:
#   export MS_ASSET_DIR=/root/.maniskill
#   bash teemo_sim_probe/tools/setup_assets.sh [ckpt_root] [n_success] [max_samples]
#
# Example:
#   bash teemo_sim_probe/tools/setup_assets.sh mshab_checkpoints 30 2000

set -euo pipefail

CKPT_ROOT="${1:-mshab_checkpoints}"
N_SUCCESS="${2:-30}"
MAX_SAMPLES="${3:-2000}"

if [[ ! -d "${CKPT_ROOT}/rl" ]]; then
    echo "ERROR: ${CKPT_ROOT}/rl not found." >&2
    echo "       Pass the parent directory of 'rl/' as the first argument." >&2
    exit 1
fi

if [[ -z "${MS_ASSET_DIR:-}" ]]; then
    echo "ERROR: MS_ASSET_DIR is not set." >&2
    echo "       export MS_ASSET_DIR=/root/.maniskill  (or wherever)" >&2
    echo "       Note: this is the parent of data/, NOT the data dir itself." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CFG_DIR="teemo_sim_probe/configs"
AFFORD_JSON="${CFG_DIR}/affordances.json"
WHITELIST_DIR="${CFG_DIR}/subtask_whitelists"
SUCCESS_STATES_DIR="${MS_ASSET_DIR}/data/robot_success_states"

echo "================================================================"
echo " STEP 2 -- collect robot_success_states/"
echo "   MS_ASSET_DIR = ${MS_ASSET_DIR}"
echo "   ckpt root    = ${CKPT_ROOT}/rl"
echo "   target N     = ${N_SUCCESS} successes per obj"
echo "================================================================"
python -m teemo_sim_probe.tools.collect_robot_success_states \
    --ckpt-root "${CKPT_ROOT}/rl" \
    --n-success "${N_SUCCESS}" \
    --no-skip-done

echo ""
echo "================================================================"
echo " STEP 3 -- mine ${AFFORD_JSON}"
echo "   max samples  = ${MAX_SAMPLES} affordance candidates per obj"
echo "================================================================"
python -m teemo_sim_probe.tools.build_affordances \
    --success-states-dir "${MS_ASSET_DIR}/data/robot_success_states" \
    --robot fetch \
    --subtask pick \
    --out "${AFFORD_JSON}" \
    --max-samples "${MAX_SAMPLES}"

echo ""
echo "================================================================"
echo " STEP 4 -- mine per-subtask whitelists -> ${WHITELIST_DIR}/"
echo "================================================================"
python -m teemo_sim_probe.tools.build_subtask_whitelists \
    --success-states-dir "${SUCCESS_STATES_DIR}" \
    --out-dir "${WHITELIST_DIR}"

echo ""
echo "================================================================"
echo " Setup complete."
echo "   ${AFFORD_JSON}"
echo "   ${WHITELIST_DIR}/  (one JSON per <subtask>_<canonical_target>)"
echo " The probe can now load_config() without FileNotFoundError."
echo "================================================================"
