#!/bin/bash
# Launch a probe from local SSD trees. Extra hydra overrides after CONFIG_NAME.
set -euo pipefail
CONFIG_NAME="${1:?config name required}"
shift || true

export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/kpfs_ssd/data/ruitong_gan/RoboTwin}"
export ASSETS_PATH="${ASSETS_PATH:-${ROBOTWIN_PATH}}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-ALOHA}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export RAY_DEDUP_LOGS=0
export RAY_worker_register_timeout_seconds=300
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Drop K8s Service env vars so Ray runtime_env stays under Linux ARG_MAX.
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

cd /kpfs_ssd/data/ruitong_gan/RLinf
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
  "env.eval.assets_path=${ASSETS_PATH}"
  "$@"
)
printf '%s\n' "${CMD[@]}" | tee "${LOG_DIR}/run_embodiment.log"
stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/run_embodiment.log"
