# Copyright 2025 The RLinf Authors.

from omegaconf import OmegaConf

from rlinf.utils.logging import should_log_progress_milestone
from rlinf.utils.robotwin_hang_diagnostics import (
    chunk_log_interval,
    configure_from_cfg,
    heartbeat_interval_sec,
    heartbeat_stuck_threshold_sec,
)


def test_configure_throttle_defaults():
    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": False},
            "env": {"train": {"env_type": "robotwin"}},
        }
    )
    configure_from_cfg(cfg)
    assert heartbeat_interval_sec() == 60.0
    assert heartbeat_stuck_threshold_sec() == 45.0
    assert chunk_log_interval() == 8


def test_milestone_throttles_per_chunk_noise():
    configure_from_cfg(
        OmegaConf.create(
            {
                "runner": {"only_eval": False},
                "env": {
                    "train": {
                        "env_type": "robotwin",
                        "hang_diagnostics_chunk_log_interval": 8,
                    }
                },
            }
        )
    )
    template = "epoch 2/4 chunk {}/32 stage=0: send_to done"
    assert should_log_progress_milestone(template.format(1))
    assert should_log_progress_milestone(template.format(8))
    assert should_log_progress_milestone(template.format(32))
    assert not should_log_progress_milestone(template.format(7))
    assert not should_log_progress_milestone(template.format(15))


def test_milestone_always_logs_critical():
    assert should_log_progress_milestone(
        "env_interact_step: chunk_step EXCEPTION AssertionError:"
    )
    assert should_log_progress_milestone("rank=1 send_rollout_trajectories: done")
    assert should_log_progress_milestone("epoch 4/4 stage=0: final bootstrap recv done")


def test_milestone_skips_predict_and_interact_micro_steps():
    assert not should_log_progress_milestone("predict: before predict_action_batch")
    assert not should_log_progress_milestone(
        "env_interact_step: before chunk_step stage=0"
    )
    assert not should_log_progress_milestone(
        "chunk 3/32 stage=0: waiting recv obs from Env"
    )
