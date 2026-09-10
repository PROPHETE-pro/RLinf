#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}

# Honor caller-exported trees. Do not auto-rewrite /kpfs → ~/ruitong_gan.
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/path/to/RoboTwin}"
export ASSETS_PATH="${ASSETS_PATH:-${ROBOTWIN_PATH}}"
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
echo "Using ROBOTWIN_PATH=${ROBOTWIN_PATH}"
echo "Using ASSETS_PATH=${ASSETS_PATH}"
echo "Using PYTHONPATH=${PYTHONPATH}"
# CPU-quota 开发机: allow slow worker import (torch/openpi) before Ray kills them.
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-300}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Drop K8s Service env vars so Ray runtime_env stays under Linux ARG_MAX.
# See rlinf/scheduler/cluster/node.py (filter_k8s_service_env_vars).
if command -v python3 >/dev/null 2>&1; then
  while IFS= read -r _k8s_env_name; do
    [ -n "$_k8s_env_name" ] && unset "$_k8s_env_name"
  done <<EOF
$(python3 - <<'PY'
import os, re
def drop(k, v):
    if k.startswith(("KAIC_", "KUBERNETES_")):
        return True
    if "_VPC_LB_" in k:
        return True
    if k.endswith("_SERVICE_HOST") or "_SERVICE_PORT" in k:
        return True
    if re.search(r"_PORT_\d+_(TCP|UDP)", k):
        return True
    if k.endswith("_PORT") and (v.startswith("tcp://") or v.startswith("udp://")):
        return True
    return False
print("\n".join(k for k, v in os.environ.items() if drop(k, v)))
PY
)
EOF
  unset _k8s_env_name
fi

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_NO_OMNI_LOGS=${OMNIGIBSON_NO_OMNI_LOGS:-1}
export OMNIGIBSON_DEBUG=${OMNIGIBSON_DEBUG:-0}
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

# POLARIS dataset
export POLARIS_DATA_PATH=${POLARIS_DATA_PATH:-"/path/to/dataset/PolaRiS-Hub"}

if [ -z "$1" ]; then
    CONFIG_NAME=${CONFIG_NAME:-"maniskill_ppo_openvlaoft"}
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM

# Libero variant: standard, pro, plus
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
# Forward optional overrides exported by callers (e.g. tests/parity_tests/run_all.sh).
# Sentinel: "-2" means "do not override, use YAML default". -1 is a legitimate value
# (e.g. runner.max_steps=-1 means unlimited) and is forwarded as-is.
EXTRA_OVERRIDES=""
[ -n "${STEPS:-}" ]      && [ "$STEPS"      != "-2" ] && EXTRA_OVERRIDES+=" runner.max_steps=${STEPS}"
[ -n "${SAVE_INTER:-}" ] && [ "$SAVE_INTER" != "-2" ] && EXTRA_OVERRIDES+=" runner.save_interval=${SAVE_INTER}"
[ -n "${NODES:-}" ]      && [ "$NODES"      != "-2" ] && EXTRA_OVERRIDES+=" cluster.num_nodes=${NODES}"

CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}${EXTRA_OVERRIDES}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
