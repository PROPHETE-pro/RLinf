# Copyright 2026 The RLinf Authors.

import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.embodied_buffer_dataset import (
    ReplayBufferDataset,
    align_mixed_batches_for_concat,
    canonicalize_mixed_sample_batch,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def _make_online_trajectory(*, t: int, b: int, h: int, d: int) -> Trajectory:
    traj = Trajectory(max_episode_length=t * h, model_weights_id="online")
    traj.actions = torch.randn(t, b, h * d)
    traj.intervene_flags = torch.zeros(t, b, h * d, dtype=torch.bool)
    traj.rewards = torch.randn(t, b, h)
    traj.terminations = torch.zeros(t, b, h, dtype=torch.bool)
    traj.truncations = torch.zeros(t, b, h, dtype=torch.bool)
    traj.dones = torch.zeros(t, b, h, dtype=torch.bool)
    traj.versions = torch.zeros(t, b, 1, dtype=torch.long)
    traj.curr_obs = {"states": torch.randn(t, b, d)}
    traj.next_obs = {"states": torch.randn(t, b, d)}
    return traj


def _make_demo_trajectory(*, t: int, b: int, h: int, d: int) -> Trajectory:
    traj = Trajectory(max_episode_length=t * h, model_weights_id="demo")
    # Legacy LeRobot conversion layout: [T, B, H, D].
    traj.actions = torch.randn(t, b, h, d)
    # Legacy demo flags are chunk-level [T, B, H], not action-flat [T, B, H*D].
    traj.intervene_flags = torch.zeros(t, b, h, dtype=torch.bool)
    traj.rewards = torch.randn(t, b, h)
    traj.terminations = torch.zeros(t, b, h, dtype=torch.bool)
    traj.truncations = torch.zeros(t, b, h, dtype=torch.bool)
    traj.dones = torch.zeros(t, b, h, dtype=torch.bool)
    traj.versions = torch.zeros(t, b, dtype=torch.long)
    traj.prev_values = torch.zeros(t, b, dtype=torch.float32)
    traj.prev_logprobs = torch.zeros(t, b, h, dtype=torch.float32)
    traj.curr_obs = {"states": torch.randn(t, b, d)}
    traj.next_obs = {"states": torch.randn(t, b, d)}
    return traj


def test_mixed_sample_concat_online_and_legacy_demo_actions():
    replay = TrajectoryReplayBuffer(
        seed=0,
        enable_cache=True,
        cache_size=32,
        sample_window_size=32,
        auto_save=False,
    )
    demo = TrajectoryReplayBuffer(
        seed=1,
        enable_cache=True,
        cache_size=32,
        sample_window_size=32,
        auto_save=False,
    )
    replay.add_trajectories([_make_online_trajectory(t=4, b=2, h=3, d=5)])
    demo.add_trajectories([_make_demo_trajectory(t=4, b=1, h=3, d=5)])

    dataset = ReplayBufferDataset(
        replay_buffer=replay,
        demo_buffer=demo,
        batch_size=8,
        min_replay_buffer_size=1,
        min_demo_buffer_size=1,
        demo_ratio=0.5,
    )
    batch = dataset._sample_mixed_batch()

    assert batch["actions"].dim() == 2
    assert batch["actions"].shape[0] == 8
    assert batch["actions"].shape[1] == 15
    assert batch["intervene_flags"].shape == batch["actions"].shape
    assert batch["versions"].shape == (8, 1)


def test_canonicalize_mixed_sample_batch_versions():
    replay = canonicalize_mixed_sample_batch(
        {"versions": torch.full((4, 50, 14), 3.0)}
    )
    demo = canonicalize_mixed_sample_batch(
        {"versions": torch.zeros(4, dtype=torch.long)}
    )
    assert replay["versions"].shape == (4, 1)
    assert demo["versions"].shape == (4, 1)


def test_align_mixed_batches_expands_demo_intervene_flags():
    replay = {
        "actions": torch.randn(4, 15),
        "intervene_flags": torch.zeros(4, 15, dtype=torch.bool),
    }
    demo = {
        "actions": torch.randn(4, 15),
        "intervene_flags": torch.zeros(4, 1, dtype=torch.bool),
    }
    replay, demo = align_mixed_batches_for_concat(replay, demo)
    assert replay["intervene_flags"].shape == (4, 15)
    assert demo["intervene_flags"].shape == (4, 15)


def test_align_mixed_batches_cross_expands_when_demo_action_width_differs():
    replay = {
        "actions": torch.randn(4, 15),
        "intervene_flags": torch.zeros(4, 15, dtype=torch.bool),
    }
    demo = {
        "actions": torch.randn(4, 1),
        "intervene_flags": torch.zeros(4, 1, dtype=torch.bool),
    }
    replay, demo = align_mixed_batches_for_concat(replay, demo)
    assert replay["actions"].shape == (4, 15)
    assert demo["actions"].shape == (4, 15)
    assert demo["intervene_flags"].shape == (4, 15)


def test_align_action_aligned_fields_expands_chunk_flags():
    flat = {
        "actions": torch.randn(4, 15),
        "intervene_flags": torch.zeros(4, 3, dtype=torch.bool),
    }
    TrajectoryReplayBuffer._align_action_aligned_fields(flat)
    assert flat["intervene_flags"].shape == (4, 15)


def test_align_mixed_batches_resizes_obs_images_to_224():
    replay = {
        "actions": torch.randn(4, 15),
        "curr_obs": {
            "main_images": torch.randint(0, 255, (4, 240, 320, 3), dtype=torch.uint8),
            "wrist_images": torch.randint(0, 255, (4, 2, 240, 320, 3), dtype=torch.uint8),
            "states": torch.randn(4, 14),
        },
        "next_obs": {
            "main_images": torch.randint(0, 255, (4, 240, 320, 3), dtype=torch.uint8),
        },
    }
    demo = {
        "actions": torch.randn(4, 15),
        "curr_obs": {
            "main_images": torch.randint(0, 255, (4, 224, 224, 3), dtype=torch.uint8),
            "wrist_images": torch.randint(0, 255, (4, 2, 224, 224, 3), dtype=torch.uint8),
            "states": torch.randn(4, 14),
        },
        "next_obs": {
            "main_images": torch.randint(0, 255, (4, 224, 224, 3), dtype=torch.uint8),
        },
    }
    replay, demo = align_mixed_batches_for_concat(replay, demo)
    assert replay["curr_obs"]["main_images"].shape == (4, 224, 224, 3)
    assert replay["curr_obs"]["wrist_images"].shape == (4, 2, 224, 224, 3)
    assert demo["curr_obs"]["main_images"].shape == (4, 224, 224, 3)
    assert replay["next_obs"]["main_images"].shape == (4, 224, 224, 3)


def test_flatten_action_samples_handles_chunked_layout():
    actions = torch.randn(6, 3, 4)
    flat = TrajectoryReplayBuffer._flatten_action_samples(
        actions.reshape(6, 3 * 4)
    )
    assert flat.shape == (6, 12)

    legacy = TrajectoryReplayBuffer._flatten_action_samples(actions)
    assert legacy.shape == (6, 12)
