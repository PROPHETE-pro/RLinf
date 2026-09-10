#!/bin/bash
# Smoke-like SubEnv capacity probe. Usage: run_subenv_capacity.sh CONFIG_NAME N_ENVS
set -euo pipefail
CONFIG_NAME="${1:?config}"
N_ENVS="${2:?n_envs}"
case "${N_ENVS}" in
  4)  GBS=4;  MBS=1 ;;
  8)  GBS=8;  MBS=2 ;;
  12) GBS=12; MBS=1 ;;
  16) GBS=16; MBS=2 ;;
  20) GBS=20; MBS=1 ;;
  24) GBS=24; MBS=2 ;;
  32) GBS=32; MBS=2 ;;
  64) GBS=64; MBS=2 ;;
  *) echo "unsupported n_envs=${N_ENVS}"; exit 1 ;;
esac

exec bash /kpfs/data/ruitong_gan/RLinf/examples/embodiment/run_kpfs_probe.sh \
  "${CONFIG_NAME}" \
  "env.train.total_num_envs=${N_ENVS}" \
  "env.train.use_dense_reward=False" \
  "env.train.use_rel_reward=False" \
  "env.train.use_custom_reward=False" \
  "env.train.video_cfg.save_video=False" \
  "env.train.rollout_epoch=1" \
  "env.train.max_episode_steps=50" \
  "env.train.max_steps_per_rollout_epoch=50" \
  "env.train.task_config.step_lim=50" \
  "actor.enable_sft_co_train=False" \
  "runner.ckpt_path=null" \
  "runner.max_epochs=1" \
  "actor.global_batch_size=${GBS}" \
  "actor.micro_batch_size=${MBS}" \
  "actor.enable_offload=True" \
  "env.enable_offload=True" \
  "rollout.enable_offload=True" \
  "env.train.enable_offload=True" \
  "runner.logger.experiment_name=capacity_${CONFIG_NAME}_${N_ENVS}env"
