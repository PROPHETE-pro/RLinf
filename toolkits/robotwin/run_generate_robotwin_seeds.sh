#!/usr/bin/env bash
# Generate RoboTwin success seeds for RLinf train/eval JSON files.
#
# Usage:
#   export ROBOTWIN_PATH=/path/to/RoboTwin
#   bash toolkits/robotwin/run_generate_robotwin_seeds.sh
#
# Optional overrides:
#   TASKS="blocks_ranking_size hanging_mug"
#   TASK_CONFIG=demo_randomized   # writes train/eval_seeds_demo_randomized.json
#   NUM_GPUS=8
#   WORKERS_PER_GPU=2
#   TRAIN_COUNT=1000
#   EVAL_COUNT=260
#   SEED_TIMEOUT=300              # kill hung seed checks after N seconds
#   QUEUE_SIZE=1                  # keep progress logs intuitive
#   MIN_NEXT_TRAIN_SEED=11309     # skip known-bad lower candidates
#   PLANNER_BACKEND=mplib         # or curobo

set -euo pipefail

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-}"
TASKS="${TASKS:-blocks_ranking_size hanging_mug open_microwave place_mouse_pad}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
TRAIN_COUNT="${TRAIN_COUNT:-1000}"
EVAL_COUNT="${EVAL_COUNT:-260}"
NUM_GPUS="${NUM_GPUS:-}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
SEED_TIMEOUT="${SEED_TIMEOUT:-180}"
QUEUE_SIZE="${QUEUE_SIZE:-}"
MIN_NEXT_TRAIN_SEED="${MIN_NEXT_TRAIN_SEED:-}"
MIN_NEXT_EVAL_SEED="${MIN_NEXT_EVAL_SEED:-}"
PLANNER_BACKEND="${PLANNER_BACKEND:-mplib}"

export ROBOT_PLATFORM="${ROBOT_PLATFORM:-ALOHA}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

if [[ -z "${ROBOTWIN_PATH}" ]]; then
  echo "ROBOTWIN_PATH is required." >&2
  exit 1
fi

CMD=(
  python "${REPO_PATH}/toolkits/robotwin/generate_robotwin_seeds.py"
  --robotwin-path "${ROBOTWIN_PATH}"
  --tasks ${TASKS}
  --task-config "${TASK_CONFIG}"
  --train-count "${TRAIN_COUNT}"
  --eval-count "${EVAL_COUNT}"
  --workers-per-gpu "${WORKERS_PER_GPU}"
  --seed-timeout "${SEED_TIMEOUT}"
  --planner-backend "${PLANNER_BACKEND}"
  --merge-existing
  --skip-existing
)

if [[ -n "${NUM_GPUS}" ]]; then
  CMD+=(--num-gpus "${NUM_GPUS}")
fi

if [[ -n "${QUEUE_SIZE}" ]]; then
  CMD+=(--queue-size "${QUEUE_SIZE}")
fi

if [[ -n "${MIN_NEXT_TRAIN_SEED}" ]]; then
  CMD+=(--min-next-train-seed "${MIN_NEXT_TRAIN_SEED}")
fi

if [[ -n "${MIN_NEXT_EVAL_SEED}" ]]; then
  CMD+=(--min-next-eval-seed "${MIN_NEXT_EVAL_SEED}")
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
