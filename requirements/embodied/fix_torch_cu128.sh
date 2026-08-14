#!/usr/bin/env bash
# Reinstall torch/torchvision/torchaudio with +cu128 wheels in an existing uv venv.
# Use this on B200 nodes when bare `uv sync` pulled cu126 wheels from PyPI/mirrors.
#
# IMPORTANT: do NOT pass a PyPI mirror as --extra-index-url here — uv will prefer the
# mirror's generic cu126 torch wheel over the cu128 build on the PyTorch index.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF_ROOT="${RLINF_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-${RLINF_ROOT}/.venv}"

TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TV_VERSION="${TV_VERSION:-0.22.1}"
TA_VERSION="${TA_VERSION:-2.7.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "venv not found at ${VENV_DIR}; set VENV_DIR or create .venv first." >&2
  exit 1
fi

UV="${UV:-uv}"
if ! command -v "${UV}" >/dev/null 2>&1; then
  UV="${VENV_DIR}/bin/uv"
fi

echo "[fix_torch_cu128] Reinstalling torch ${TORCH_VERSION}+cu128 into ${VENV_DIR}"
echo "[fix_torch_cu128] Using index: ${TORCH_INDEX} (PyPI mirrors are ignored for this step)"

# Clear mirror/index env vars that would override --index-url and pull cu126 wheels.
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL UV_INDEX_URL UV_EXTRA_INDEX_URL \
  UV_DEFAULT_INDEX PIP_TRUSTED_HOST 2>/dev/null || true

"${UV}" pip uninstall torch torchvision torchaudio 2>/dev/null || true

# Install ONLY from the PyTorch cu128 index — no extra-index-url.
env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL \
  "${UV}" pip install --reinstall \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TV_VERSION}" \
  "torchaudio==${TA_VERSION}" \
  --index-url "${TORCH_INDEX}"

"${VENV_DIR}/bin/python" - <<'EOF'
import sys
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("arch_list:", torch.cuda.get_arch_list())
if "+cu128" not in torch.__version__:
    print("ERROR: torch is not +cu128; B200 will not work.", file=sys.stderr)
    print("Hint: ensure no PIP_INDEX_URL/UV_INDEX_URL points at a PyPI mirror.", file=sys.stderr)
    sys.exit(1)
if "sm_100" not in torch.cuda.get_arch_list():
    print("WARNING: sm_100 not in arch_list; verify GPU driver is CUDA 12.8+.", file=sys.stderr)
if torch.cuda.is_available():
    x = torch.randn(2, 2, device="cuda")
    print("cuda smoke test ok:", (x @ x).shape)
EOF

echo "[fix_torch_cu128] Done."
