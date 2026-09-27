#!/usr/bin/env bash
# One single-env dense-reward probe per precision task.
# Rollout length is twice the task's official step limit, matching insert_key.
set -u
cd /kpfs_ssd/data/ruitong_gan/RLinf
cfg=robodojo_insert_key_ppo_opendm_dm05_1gpu

run_one() {
  local task="$1"
  local official="$2"
  local rollout=$((official * 2))
  echo "[probe-batch] task=${task} official=${official} rollout=${rollout}"
  python3 toolkits/robodojo/probe_first_chunk.py "${cfg}" \
    "env.train.task_config.task_name=${task}" \
    "env.train.task_config.step_lim=${official}" \
    "env.train.official_step_lim=${official}" \
    "env.train.rollout_step_lim=${rollout}" \
    "env.train.max_episode_steps=${rollout}" \
    "env.train.max_steps_per_rollout_epoch=${rollout}"
}

# insert_key and plug_in_charger already finished.
# deposit_coin is skipped: every procedural seed fails to place coin0.
run_one fasten_screws 1900
run_one insert_tubes 500
run_one build_tower 1050
run_one pour_balls_into_vase 600
run_one play_Xylophone 500
echo "[probe-batch] done"
