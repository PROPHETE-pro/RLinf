#!/bin/bash
# Single-GPU PPO on RoboDojo build_tower with OpenDM dm05_robodojo.
# Extra hydra overrides after the script name are forwarded.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/kpfs_ssd/data/ruitong_gan}"
RLINF_ROOT="${RLINF_ROOT:-${WORKSPACE}/RLinf}"
SETUP_SH="${RLINF_ROOT}/examples/embodiment/setup_robodojo_rlinf_env.sh"
RUN_SH="${RLINF_ROOT}/examples/embodiment/run_robodojo.sh"
CONFIG_NAME="robodojo_build_tower_ppo_opendm_dm05_1gpu"

if [[ ! -f "${SETUP_SH}" ]]; then
  echo "[run_robodojo_build_tower_1gpu] missing ${SETUP_SH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${SETUP_SH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

cd "${RLINF_ROOT}"
exec bash "${RUN_SH}" "${CONFIG_NAME}" "$@"
