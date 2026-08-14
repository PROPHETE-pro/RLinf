#! /bin/bash
#
# Resume embodied RL training from a saved checkpoint.
#
# Usage:
#   bash run_embodiment_resume.sh <config_name> <log_dir> <checkpoint> [robot_platform]
#
# Arguments:
#   config_name     Hydra config name (without .yaml), e.g. libero_plus_10_ppo_openpi_pi05
#   log_dir         Original experiment log directory (same as runner.logger.log_path)
#   checkpoint      Either:
#                     - global step number, e.g. 200
#                     - full checkpoint dir, e.g. .../checkpoints/global_step_200
#   robot_platform  Optional. LIBERO (default), ALOHA, or BRIDGE
#
# Example:
#   cd /mnt/pfs/7wsqem/grt/RLinf
#   bash examples/embodiment/run_embodiment_resume.sh \
#     libero_plus_10_ppo_openpi_pi05 \
#     /mnt/pfs/7wsqem/grt/RLinf/logs/20260709-18:58:35-libero_plus_10_ppo_openpi_pi05 \
#     200 \
#     LIBERO
#
# Environment variables (optional):
#   LIBERO_TYPE       standard | pro | plus (auto-detected from config_name if unset)
#   LIBERO_SUFFIX     used when LIBERO_TYPE=plus (default: all)
#   LIBERO_PERTURBATION  used when LIBERO_TYPE=pro (default: all)
#   ROBOT_PLATFORM    overrides the 4th positional argument
#
set -euo pipefail

usage() {
    sed -n '3,28p' "$0" | sed 's/^# \{0,1\}//'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}

export OMNIGIBSON_NO_OMNI_LOGS=${OMNIGIBSON_NO_OMNI_LOGS:-1}
export OMNIGIBSON_DEBUG=${OMNIGIBSON_DEBUG:-0}
export OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH:-}
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}
export POLARIS_DATA_PATH=${POLARIS_DATA_PATH:-"/path/to/dataset/PolaRiS-Hub"}

CONFIG_NAME=${1:-${CONFIG_NAME:-}}
LOG_DIR=${2:-${LOG_DIR:-}}
CHECKPOINT_ARG=${3:-${RESUME_STEP:-${RESUME_DIR:-}}}
ROBOT_PLATFORM=${4:-${ROBOT_PLATFORM:-"LIBERO"}}

if [[ -z "$CONFIG_NAME" || -z "$LOG_DIR" || -z "$CHECKPOINT_ARG" ]]; then
    echo "Error: missing required arguments." >&2
    echo >&2
    usage >&2
    exit 1
fi

if [[ ! -d "$LOG_DIR" ]]; then
    echo "Error: log_dir does not exist: $LOG_DIR" >&2
    exit 1
fi

if [[ "$CHECKPOINT_ARG" =~ ^[0-9]+$ ]]; then
    RESUME_DIR="${LOG_DIR}/${CONFIG_NAME}/checkpoints/global_step_${CHECKPOINT_ARG}"
else
    RESUME_DIR="$CHECKPOINT_ARG"
fi

ACTOR_CHECKPOINT="${RESUME_DIR}/actor"
if [[ ! -d "$ACTOR_CHECKPOINT" ]]; then
    echo "Error: checkpoint actor directory not found: $ACTOR_CHECKPOINT" >&2
    echo "Hint: list available checkpoints with:" >&2
    echo "  ls ${LOG_DIR}/${CONFIG_NAME}/checkpoints/" >&2
    exit 1
fi

if [[ -z "${LIBERO_TYPE:-}" ]]; then
    if [[ "$CONFIG_NAME" == *libero_plus* ]]; then
        export LIBERO_TYPE="plus"
    elif [[ "$CONFIG_NAME" == *libero_pro* ]]; then
        export LIBERO_TYPE="pro"
    else
        export LIBERO_TYPE="standard"
    fi
fi

export ROBOT_PLATFORM

if [[ "$LIBERO_TYPE" == "pro" ]]; then
    export LIBERO_PERTURBATION=${LIBERO_PERTURBATION:-"all"}
    echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
elif [[ "$LIBERO_TYPE" == "plus" ]]; then
    export LIBERO_SUFFIX=${LIBERO_SUFFIX:-"all"}
    echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
else
    echo "Evaluation Mode: Standard LIBERO"
fi

RESUME_STEP="${RESUME_DIR##*/global_step_}"
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"
echo "Using Python at $(which python)"
echo "Config:      $CONFIG_NAME"
echo "Log dir:     $LOG_DIR"
echo "Resume from: $RESUME_DIR (global_step=${RESUME_STEP})"

MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"

CMD="python ${SRC_FILE} \
  --config-path ${EMBODIED_PATH}/config/ \
  --config-name ${CONFIG_NAME} \
  runner.logger.log_path=${LOG_DIR} \
  runner.resume_dir=${RESUME_DIR}"

{
    echo ""
    echo "===== $(date +'%Y-%m-%d %H:%M:%S') RESUME global_step_${RESUME_STEP} ====="
    echo "${CMD}"
} >> "${MEGA_LOG_FILE}"

${CMD} 2>&1 | tee -a "${MEGA_LOG_FILE}"
