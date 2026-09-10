#!/bin/bash
# Wait until N setup_demo ok, or declare hang if log stalls.
# Usage: wait_env_init.sh LOG_FILE EXPECTED_OK [STALL_SEC=600] [MAX_SEC=1200] [WAIT_ROLLOUT=0]
set -euo pipefail
LOG="${1:?log}"
NEED="${2:?expected setup_demo ok count}"
STALL="${3:-600}"
MAX="${4:-1200}"
WAIT_ROLLOUT="${5:-0}"
start=$(date +%s)
last_size=-1
stall_since=$start
gpu_peak="0,0,0,0"

gpu_line() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
    | awk -F',' '{printf "GPU%s=%s/%sMiB(%s%%) ", $1, $2+0, $3+0, $4+0}'
}

while true; do
  now=$(date +%s)
  elapsed=$((now - start))
  mem=$(gpu_line || true)
  echo "gpu ${mem}"
  if [ ! -f "$LOG" ]; then
    if [ "$elapsed" -ge "$MAX" ]; then
      echo "VERDICT=TIMEOUT no_log elapsed=${elapsed}s"
      exit 2
    fi
    sleep 10
    continue
  fi
  ok=$(grep -c 'setup_demo ok' "$LOG" || true)
  size=$(wc -c < "$LOG")
  rollout=0
  if grep -qE 'Generating Rollout Epochs|generate_rollouts: launch|run_training: start' "$LOG"; then
    rollout=1
  fi
  if [ "$ok" -ge "$NEED" ]; then
    if [ "$WAIT_ROLLOUT" = "0" ] || [ "$rollout" = "1" ]; then
      echo "VERDICT=PASS setup_demo_ok=${ok}/${NEED} rollout=${rollout} elapsed=${elapsed}s"
      echo "gpu_final ${mem}"
      exit 0
    fi
    echo "progress init_ok=${ok}/${NEED} waiting_rollout size=${size} elapsed=${elapsed}s"
  elif [ "$rollout" = "1" ]; then
    echo "VERDICT=PASS rollout_started setup_demo_ok=${ok}/${NEED} elapsed=${elapsed}s"
    echo "gpu_final ${mem}"
    exit 0
  else
    if [ "$size" != "$last_size" ]; then
      last_size=$size
      stall_since=$now
    fi
    stalled=$((now - stall_since))
    echo "progress ok=${ok}/${NEED} size=${size} stalled=${stalled}s elapsed=${elapsed}s"
    if [ "$elapsed" -ge "$MAX" ]; then
      echo "VERDICT=TIMEOUT setup_demo_ok=${ok}/${NEED} elapsed=${elapsed}s"
      echo "gpu_final ${mem}"
      exit 2
    fi
    if [ "$ok" -gt 0 ] && [ "$stalled" -ge "$STALL" ]; then
      echo "VERDICT=HANG setup_demo_ok=${ok}/${NEED} stalled=${stalled}s last_size=${size}"
      echo "gpu_final ${mem}"
      exit 1
    fi
  fi
  if [ "$elapsed" -ge "$MAX" ]; then
    echo "VERDICT=TIMEOUT setup_demo_ok=${ok}/${NEED} rollout=${rollout} elapsed=${elapsed}s"
    echo "gpu_final ${mem}"
    exit 2
  fi
  sleep 15
done
