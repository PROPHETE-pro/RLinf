#!/bin/bash
# Launch RoboDojo post-training RL from local SSD trees.
# Extra hydra overrides after CONFIG_NAME.
#
# Before first launch in a new container, source:
#   source examples/embodiment/setup_robodojo_rlinf_env.sh --check
set -euo pipefail

WORKSPACE="${WORKSPACE:-/kpfs_ssd/data/ruitong_gan}"
export ROBODOJO_PATH="${ROBODOJO_PATH:-${WORKSPACE}/RoboDojo}"
export ROBODOJO_RUNTIME_LIBS="${ROBODOJO_RUNTIME_LIBS:-${WORKSPACE}/robodojo_runtime/runtime_libs}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-${WORKSPACE}/robodojo_runtime/caches/warp}"
export ROBODOJO_XDG_CACHE_HOME="${ROBODOJO_XDG_CACHE_HOME:-${WORKSPACE}/robodojo_runtime/caches}"
export ROBODOJO_EVAL_ROOT="${ROBODOJO_EVAL_ROOT:-${WORKSPACE}/robodojo_runtime/eval_result/RoboDojo/rlinf}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export RAY_DEDUP_LOGS=0
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-300}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -z "${ROBODOJO_PYTHON:-}" ]]; then
  for candidate in \
    "${WORKSPACE}/miniconda/envs/RoboDojo/bin/python" \
    "${WORKSPACE}/miniconda/envs/robodojo/bin/python"; do
    if [[ -x "${candidate}" ]]; then
      export ROBODOJO_PYTHON="${candidate}"
      break
    fi
  done
fi
if [[ -z "${ROBODOJO_PYTHON:-}" ]]; then
  echo "[run_robodojo] set ROBODOJO_PYTHON to the Isaac/RoboDojo conda interpreter" >&2
  exit 1
fi

mkdir -p "${WARP_CACHE_PATH}" "${ROBODOJO_EVAL_ROOT}" \
  "${ROBODOJO_XDG_CACHE_HOME}" \
  "${WORKSPACE}/robodojo_runtime/caches/ov" \
  "${WORKSPACE}/robodojo_runtime/logs"

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

cd "${WORKSPACE}/RLinf"
export EMBODIED_PATH="$(pwd)/examples/embodiment"
export REPO_PATH="$(pwd)"
# Parent process is RLinf/OpenDM python. Do not put RoboDojo on PYTHONPATH.
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_NAME="${1:-robodojo_stack_bowls_ppo_opendm_dm05}"
if [[ "${CONFIG_NAME}" == "-h" || "${CONFIG_NAME}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash examples/embodiment/run_robodojo.sh [config_name] [hydra overrides...]

Examples:
  bash examples/embodiment/run_robodojo.sh
  bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05 \
    env.train.task_config.task_name=hang_mugs env.train.max_episode_steps=800
  bash examples/embodiment/run_robodojo.sh --worker-smoke
  bash examples/embodiment/run_robodojo.sh --doctor
  bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05 \
    runner.max_epochs=1
EOF
  exit 0
fi

if [[ "${CONFIG_NAME}" == "--doctor" ]]; then
  echo "ROBODOJO_PYTHON=${ROBODOJO_PYTHON}"
  echo "ROBODOJO_PATH=${ROBODOJO_PATH}"
  missing=0
  if [[ ! -x "${ROBODOJO_PYTHON}" ]]; then
    echo "[doctor] missing executable ROBODOJO_PYTHON=${ROBODOJO_PYTHON}" >&2
    missing=1
  fi
  if [[ ! -d "${ROBODOJO_PATH}" ]]; then
    echo "[doctor] missing ROBODOJO_PATH=${ROBODOJO_PATH}" >&2
    missing=1
  fi
  if [[ ! -e "${ROBODOJO_PATH}/Assets" ]]; then
    echo "[doctor] missing Assets symlink at ${ROBODOJO_PATH}/Assets" >&2
    missing=1
  fi
  CKPT="${OPENDM_ROBODOJO_CKPT:-${WORKSPACE}/opendm/checkpoints/dm05_robodojo}"
  if [[ ! -f "${CKPT}/model.safetensors" || ! -f "${CKPT}/norm_stats.json" ]]; then
    echo "[doctor] missing OpenDM checkpoint files under ${CKPT}" >&2
    missing=1
  fi
  if [[ "${missing}" -ne 0 ]]; then
    exit 1
  fi
  echo "[doctor] ok: python / path / Assets / dm05_robodojo checkpoint"
  python3 "${REPO_PATH}/toolkits/robodojo/sync_task_horizon.py" --opendm-only | wc -l | awk '{print "[doctor] OpenDM dual_x5 tasks:", $1}'
  exit 0
fi

if [[ "${CONFIG_NAME}" == "--worker-smoke" ]]; then
  echo "ROBODOJO_PYTHON=${ROBODOJO_PYTHON}"
  echo "ROBODOJO_PATH=${ROBODOJO_PATH}"
  export LD_LIBRARY_PATH="${ROBODOJO_RUNTIME_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export XDG_CACHE_HOME="${ROBODOJO_XDG_CACHE_HOME}"
  exec "${ROBODOJO_PYTHON}" -u "${REPO_PATH}/rlinf/envs/robodojo/isaac_worker.py" \
    --standalone \
    --task_name "${ROBODOJO_SMOKE_TASK:-stack_bowls}" \
    --env_cfg_type arx_x5 \
    --device_id 0 \
    --headless \
    --enable_cameras \
    --steps "${ROBODOJO_SMOKE_STEPS:-1}"
fi

shift || true

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "CONFIG=${CONFIG_NAME}"
echo "ROBODOJO_PATH=${ROBODOJO_PATH}"
echo "ROBODOJO_PYTHON=${ROBODOJO_PYTHON}"
echo "OVERRIDES=$*"

CMD=(
  python -u "${EMBODIED_PATH}/train_embodied_agent.py"
  --config-path "${EMBODIED_PATH}/config/"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "env.train.robodojo_path=${ROBODOJO_PATH}"
  "env.eval.robodojo_path=${ROBODOJO_PATH}"
  "env.train.isaac_python=${ROBODOJO_PYTHON}"
  "env.eval.isaac_python=${ROBODOJO_PYTHON}"
  "$@"
)
export ROBODOJO_WORKER_LOG_DIR="${LOG_DIR}"
printf '%s\n' "${CMD[@]}" | tee "${LOG_DIR}/run_embodiment.log"
# Do not wrap with stdbuf: it injects LD_PRELOAD=libstdbuf.so into Isaac children.
"${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/run_embodiment.log"
