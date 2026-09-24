#!/usr/bin/env bash
# Configure the dual-interpreter env for RLinf + RoboDojo OpenDM RL.
#
# Usage (recommended, persists in the current shell):
#   source /kpfs_ssd/data/ruitong_gan/RLinf/examples/embodiment/setup_robodojo_rlinf_env.sh
#   source /kpfs_ssd/data/ruitong_gan/RLinf/examples/embodiment/setup_robodojo_rlinf_env.sh --check
#
# Then launch:
#   bash examples/embodiment/run_robodojo.sh --doctor
#   bash examples/embodiment/run_robodojo.sh --worker-smoke
#   bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05 ...
#
# Do NOT `conda activate RoboDojo` in this shell. That env is only for Isaac
# child processes (ROBODOJO_PYTHON). Parent training stays on image Python 3.12.

# Allow both `source` and `bash script.sh --check`.
_SETUP_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  _SETUP_SOURCED=1
fi

_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_RLINF_ROOT="$(cd "${_SETUP_DIR}/../.." && pwd)"
WORKSPACE="${WORKSPACE:-/kpfs_ssd/data/ruitong_gan}"

# --- paths (NFS workspace + image Python) ---
export WORKSPACE
export RLINF_ROOT="${RLINF_ROOT:-${WORKSPACE}/RLinf}"
export RLINF_PYTHON="${RLINF_PYTHON:-/usr/bin/python3}"
export ROBODOJO_PATH="${ROBODOJO_PATH:-${WORKSPACE}/RoboDojo}"
export ROBODOJO_PYTHON="${ROBODOJO_PYTHON:-${WORKSPACE}/miniconda/envs/RoboDojo/bin/python}"
export OPENDM_ROOT="${OPENDM_ROOT:-${WORKSPACE}/opendm}"
export OPENDM_ROBODOJO_CKPT="${OPENDM_ROBODOJO_CKPT:-${WORKSPACE}/opendm/checkpoints/dm05_robodojo}"

# Isaac child-only caches / GL libs. Do not overwrite parent XDG_CACHE_HOME
# (HuggingFace / torch caches live there).
export ROBODOJO_RUNTIME_LIBS="${ROBODOJO_RUNTIME_LIBS:-${WORKSPACE}/robodojo_runtime/runtime_libs}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-${WORKSPACE}/robodojo_runtime/caches/warp}"
export ROBODOJO_XDG_CACHE_HOME="${ROBODOJO_XDG_CACHE_HOME:-${WORKSPACE}/robodojo_runtime/caches}"
export ROBODOJO_EVAL_ROOT="${ROBODOJO_EVAL_ROOT:-${WORKSPACE}/robodojo_runtime/eval_result/RoboDojo/rlinf}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0}"

# Parent process hygiene
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAY_DEDUP_LOGS=0
export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-300}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"

# Parent PYTHONPATH: RLinf only. RoboDojo / XPolicyLab stay in the Isaac child.
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export REPO_PATH="${RLINF_ROOT}"
case ":${PYTHONPATH:-}:" in
  *":${RLINF_ROOT}:"*) ;;
  *) export PYTHONPATH="${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

mkdir -p "${WARP_CACHE_PATH}" "${ROBODOJO_EVAL_ROOT}" \
  "${ROBODOJO_XDG_CACHE_HOME}" \
  "${WORKSPACE}/robodojo_runtime/caches/ov" \
  "${WORKSPACE}/robodojo_runtime/logs"

_rlinf_robodojo_check() {
  local missing=0
  echo "[setup] WORKSPACE=${WORKSPACE}"
  echo "[setup] RLINF_PYTHON=${RLINF_PYTHON}"
  echo "[setup] ROBODOJO_PYTHON=${ROBODOJO_PYTHON}"
  echo "[setup] ROBODOJO_PATH=${ROBODOJO_PATH}"
  echo "[setup] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<none>}"
  echo "[setup] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "[setup] NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"

  if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
    echo "[setup] WARN: conda env '${CONDA_DEFAULT_ENV}' is active." >&2
    echo "[setup]        Parent training must stay on image Python 3.12." >&2
    echo "[setup]        Run: conda deactivate" >&2
  fi

  if [[ ! -x "${RLINF_PYTHON}" ]]; then
    echo "[setup] FAIL: RLINF_PYTHON not executable: ${RLINF_PYTHON}" >&2
    missing=1
  fi
  if [[ ! -x "${ROBODOJO_PYTHON}" ]]; then
    echo "[setup] FAIL: ROBODOJO_PYTHON not executable: ${ROBODOJO_PYTHON}" >&2
    missing=1
  fi
  if [[ ! -d "${ROBODOJO_PATH}" ]]; then
    echo "[setup] FAIL: ROBODOJO_PATH missing: ${ROBODOJO_PATH}" >&2
    missing=1
  fi
  if [[ ! -e "${ROBODOJO_PATH}/Assets" ]]; then
    echo "[setup] FAIL: Assets missing at ${ROBODOJO_PATH}/Assets" >&2
    echo "[setup]       Expected symlink to /mnt/xiaoyu_ssd_intern/RoboDojo/assets/Assets" >&2
    missing=1
  elif [[ ! -d "${ROBODOJO_PATH}/Assets/Robots" ]]; then
    echo "[setup] FAIL: Assets symlink exists but target is not readable (NFS /mnt/xiaoyu_ssd_intern?)" >&2
    missing=1
  fi
  if [[ ! -f "${OPENDM_ROBODOJO_CKPT}/model.safetensors" || ! -f "${OPENDM_ROBODOJO_CKPT}/norm_stats.json" ]]; then
    echo "[setup] FAIL: OpenDM ckpt missing under ${OPENDM_ROBODOJO_CKPT}" >&2
    missing=1
  fi
  if [[ ! -d "${ROBODOJO_RUNTIME_LIBS}" ]]; then
    echo "[setup] FAIL: runtime_libs missing: ${ROBODOJO_RUNTIME_LIBS}" >&2
    missing=1
  fi

  "${RLINF_PYTHON}" - <<'PY' || missing=1
import importlib, sys
print("[setup] parent", sys.executable, sys.version.split()[0])
need = ["torch", "hydra", "omegaconf", "ray", "gymnasium", "transformers", "opendm", "rlinf"]
bad = []
for name in need:
    try:
        mod = importlib.import_module(name)
        print(f"[setup]   OK {name} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        bad.append(f"{name}: {exc}")
        print(f"[setup]   MISS {name}: {exc}")
try:
    import torch
    print("[setup]   torch.cuda", torch.cuda.is_available(), "n", torch.cuda.device_count() if torch.cuda.is_available() else 0, "cu", getattr(torch.version, "cuda", None))
    if not torch.cuda.is_available():
        bad.append("torch.cuda unavailable")
except Exception as exc:
    bad.append(str(exc))
from rlinf.envs import SupportedEnvType, get_env_cls
assert get_env_cls("robodojo").__name__ == "RoboDojoEnv"
assert get_env_cls("robotwin").__name__ == "RoboTwinEnv"
assert SupportedEnvType.ROBODOJO.value == "robodojo"
if bad:
    raise SystemExit(1)
PY

  OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 "${ROBODOJO_PYTHON}" - <<'PY' || missing=1
import os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
print("[setup] isaac", sys.executable, sys.version.split()[0])
import isaaclab, isaacsim, torch
print("[setup]   isaaclab", getattr(isaaclab, "__version__", ""), isaaclab.__file__)
print("[setup]   isaacsim", isaacsim.__file__)
print("[setup]   torch", torch.__version__, "cuda", torch.cuda.is_available(), getattr(torch.version, "cuda", None))
from isaaclab.app import AppLauncher  # noqa: F401
print("[setup]   AppLauncher import OK (Kit not started)")
PY

  local ngpu=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    ngpu="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  fi
  echo "[setup] visible GPUs: ${ngpu}"
  if [[ "${ngpu}" -lt 1 ]]; then
    echo "[setup] FAIL: no GPU visible to nvidia-smi" >&2
    missing=1
  elif [[ "${ngpu}" -eq 1 ]]; then
    echo "[setup] NOTE: default yaml places actor/rollout on GPUs 1-7."
    echo "[setup]       This pod has 1 GPU. Override placement (printed below)."
  fi

  if [[ "${missing}" -ne 0 ]]; then
    echo "[setup] checks FAILED" >&2
    return 1
  fi
  echo "[setup] checks OK: parent=RLinf/OpenDM (py3.12), child=RoboDojo Isaac (py3.11)"
  return 0
}

_rlinf_robodojo_print_next() {
  local ngpu
  ngpu="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  cat <<EOF

Next commands (from ${RLINF_ROOT}):

  bash examples/embodiment/run_robodojo.sh --doctor
  bash examples/embodiment/run_robodojo.sh --worker-smoke

EOF
  if [[ "${ngpu}" -le 1 ]]; then
    cat <<'EOF'
  # 1-GPU smoke (Isaac + OpenDM share GPU 0). Default yaml expects 8 GPUs.
  bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05 \
    runner.max_epochs=1 \
    env.train.total_num_envs=1 \
    cluster.component_placement.actor=0 \
    cluster.component_placement.env=0 \
    cluster.component_placement.rollout=0 \
    actor.enable_offload=True \
    rollout.enable_offload=True

EOF
  elif [[ "${ngpu}" -lt 4 ]]; then
    cat <<EOF
  # ${ngpu} GPUs: keep Isaac on 0, actor/rollout on the rest.
  bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05 \
    runner.max_epochs=1 \
    env.train.total_num_envs=1 \
    cluster.component_placement.env=0 \
    cluster.component_placement.actor=1-$((ngpu - 1)) \
    cluster.component_placement.rollout=1-$((ngpu - 1))

EOF
  else
    cat <<'EOF'
  # 4+ GPUs: use the 4gpu recipe, or the default 8-GPU yaml if you have 8 cards.
  bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05_4gpu
  bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05

EOF
  fi
  cat <<'EOF'
Switch task:
  bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05 \
    env.train.task_config.task_name=hang_mugs \
    env.eval.task_config.task_name=hang_mugs

After saving this image, next boot still needs:
  1) NFS /kpfs_ssd/data  (RLinf, RoboDojo, miniconda/envs/RoboDojo, checkpoints)
  2) NFS /mnt/xiaoyu_ssd_intern  (USD Assets behind RoboDojo/Assets)
  3) Do not conda activate RoboDojo in the training shell
  4) source this script, then run_robodojo.sh
EOF
}

if [[ "${1:-}" == "--check" || "${1:-}" == "-c" ]]; then
  _rlinf_robodojo_check
  _chk=$?
  _rlinf_robodojo_print_next
  if [[ "${_SETUP_SOURCED}" -eq 0 ]]; then
    exit "${_chk}"
  fi
  return "${_chk}" 2>/dev/null || true
fi

echo "[setup] exported dual-interpreter env (parent=${RLINF_PYTHON}, isaac=${ROBODOJO_PYTHON})"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  _rlinf_robodojo_print_next
fi

if [[ "${_SETUP_SOURCED}" -eq 0 ]]; then
  echo "[setup] warning: script was executed, not sourced. Exports died with this process." >&2
  echo "[setup]          source ${BASH_SOURCE[0]}" >&2
  _rlinf_robodojo_print_next
fi
