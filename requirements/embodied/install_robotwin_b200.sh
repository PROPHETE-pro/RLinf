#!/usr/bin/env bash
# Install RLinf OpenPI + RoboTwin on NVIDIA B200 / Blackwell (sm_100+).
#
# Requires:
#   - CUDA 12.8+ driver / runtime on the target node
#   - Do NOT run on machines that only support cu126 wheels (e.g. A100/H100-only stacks)
#
# This script routes torch/torchvision/torchaudio through PyTorch's cu128 index so
# `uv sync` resolves +cu128 wheels (with sm_100 support), instead of PyPI's cu126 build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF_ROOT="${RLINF_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
CUDA_BACKEND="${CUDA_BACKEND:-cu128}"
VENV_DIR="${VENV_DIR:-.venv}"

# Optional local checkouts (override on the B200 node as needed)
export OPENPI_PATH="${OPENPI_PATH:-/mnt/pfs/7wsqem/grt/openpi}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-/mnt/pfs/7wsqem/grt/RoboTwin}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export GITHUB_PREFIX="${GITHUB_PREFIX:-https://gh-proxy.org/}"

log() { echo "[install_robotwin_b200] $*"; }

log "RLINF_ROOT=${RLINF_ROOT}"
log "Installing with torch=${TORCH_VERSION}+${CUDA_BACKEND} into ${VENV_DIR}"

CUDA_BACKEND="${CUDA_BACKEND}" \
UV_TORCH_BACKEND="${CUDA_BACKEND}" \
bash "${RLINF_ROOT}/requirements/install.sh" embodied \
  --model openpi \
  --env robotwin \
  --venv "${VENV_DIR}" \
  --torch "${TORCH_VERSION}" \
  --cuda-backend "${CUDA_BACKEND}" \
  --install-rlinf \
  "$@"

# Prefer a local OpenPI checkout when available (matches rlinf-openpi-robotwin conda flow).
if [ -d "${OPENPI_PATH}" ]; then
  log "Installing local OpenPI from ${OPENPI_PATH}"
  # shellcheck disable=SC1090
  source "${RLINF_ROOT}/${VENV_DIR}/bin/activate"
  uv pip install -e "${OPENPI_PATH}"
fi

log "Verifying torch build..."
"${RLINF_ROOT}/${VENV_DIR}/bin/python" - <<'EOF'
import sys
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("arch_list:", torch.cuda.get_arch_list())
if "+cu128" not in torch.__version__:
    print("ERROR: expected torch build with +cu128 for B200/Blackwell.", file=sys.stderr)
    sys.exit(1)
if "sm_100" not in torch.cuda.get_arch_list():
    print("WARNING: sm_100 not listed in arch_list; B200 may still fail.", file=sys.stderr)
if torch.cuda.is_available():
    x = torch.randn(2, 2, device="cuda")
    print("cuda smoke test ok:", (x @ x).shape)
EOF

log "Done. Activate with: source ${RLINF_ROOT}/${VENV_DIR}/bin/activate"
