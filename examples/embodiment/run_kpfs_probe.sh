#!/bin/bash
# Launch a 4-GPU probe from JuiceFS trees. Extra hydra overrides after CONFIG_NAME.
set -euo pipefail
CONFIG_NAME="${1:?config name required}"
shift || true

export ROBOTWIN_PATH="/kpfs/data/ruitong_gan/RoboTwin"
export ASSETS_PATH="/kpfs/data/ruitong_gan/RoboTwin"
export ROBOT_PLATFORM="ALOHA"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export RAY_DEDUP_LOGS=0
export RAY_worker_register_timeout_seconds=300

cd /kpfs/data/ruitong_gan/RLinf
export EMBODIED_PATH="$(pwd)/examples/embodiment"
export REPO_PATH="$(pwd)"
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "CONFIG=${CONFIG_NAME}"
echo "OVERRIDES=$*"
echo "ROBOTWIN_PATH=${ROBOTWIN_PATH}"
echo "ASSETS_PATH=${ASSETS_PATH}"

CMD=(
  python "${EMBODIED_PATH}/train_embodied_agent.py"
  --config-path "${EMBODIED_PATH}/config/"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "env.train.assets_path=${ASSETS_PATH}"
  "$@"
)
printf '%s\n' "${CMD[@]}" | tee "${LOG_DIR}/run_embodiment.log"
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/run_embodiment.log"
