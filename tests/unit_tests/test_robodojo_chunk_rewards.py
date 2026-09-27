import math
from types import SimpleNamespace

import torch

from rlinf.envs.robodojo.reward import (
    chunk_reward_from_cumulative,
    cumulative_score,
    reallocate_chunk_advantages,
)
from rlinf.envs.robodojo.robodojo_env import RoboDojoEnv


OFFICIAL = 1050
DECAY_STEPS = 525.0
TIME_COST = 0.0004


def _cumulative(**overrides):
    params = dict(
        success=False,
        step_count=50,
        official_step_lim=OFFICIAL,
        deadline_mode="hard",
        decay_steps=DECAY_STEPS,
        dense_progress=None,
        use_dense_reward=False,
        shaping_coef=0.5,
        success_reward=1.0,
    )
    params.update(overrides)
    return cumulative_score(**params)


def test_success_inside_official_limit_jumps_to_one():
    assert _cumulative(success=True, step_count=OFFICIAL) == 1.0
    reward = chunk_reward_from_cumulative(1.0, 0.0, TIME_COST, 50)
    assert math.isclose(reward, 1.0 - TIME_COST * 50)


def test_timeout_without_success_is_only_the_time_cost():
    assert _cumulative(success=False, step_count=OFFICIAL) == 0.0
    reward = chunk_reward_from_cumulative(0.0, 0.0, TIME_COST, 50)
    assert math.isclose(reward, -TIME_COST * 50)


def test_hard_deadline_drops_late_success():
    assert _cumulative(success=True, step_count=OFFICIAL + 1, deadline_mode="hard") == 0.0


def test_decay_deadline_shrinks_late_success():
    late = OFFICIAL + DECAY_STEPS
    score = _cumulative(success=True, step_count=int(late), deadline_mode="decay")
    assert math.isclose(score, math.exp(-1.0))


def test_missing_dense_progress_stays_sparse():
    assert (
        _cumulative(
            success=False,
            dense_progress=None,
            use_dense_reward=True,
        )
        == 0.0
    )
    assert (
        _cumulative(
            success=False,
            dense_progress=0.4,
            use_dense_reward=False,
        )
        == 0.0
    )


def test_dense_progress_is_a_shaped_cumulative_delta():
    first = _cumulative(dense_progress=0.2, use_dense_reward=True)
    second = _cumulative(dense_progress=0.5, use_dense_reward=True)
    assert math.isclose(first, 0.1)
    assert math.isclose(chunk_reward_from_cumulative(second, first, 0.0, 50), 0.15)


def test_late_hard_success_keeps_progress_but_not_the_terminal_bonus():
    score = _cumulative(
        success=True,
        step_count=OFFICIAL + 10,
        deadline_mode="hard",
        dense_progress=0.4,
        use_dense_reward=True,
    )
    assert math.isclose(score, 0.2)


def test_reallocate_keeps_the_chunk_sum_and_the_step_signs():
    advantages = torch.zeros(1, 1, 1)
    deltas = torch.tensor([[[0.3, 0.3, -0.2, -0.4]]], dtype=torch.float32)
    split = reallocate_chunk_advantages(advantages, deltas)
    assert torch.allclose(split.sum(dim=-1), advantages[..., 0])
    assert split[0, 0, 0] > 0
    assert split[0, 0, -1] < 0


def test_progress_trace_deltas_sum_to_the_net_change():
    env = RoboDojoEnv.__new__(RoboDojoEnv)
    env.num_envs = 1
    env.prev_step_reward = torch.tensor([0.2])
    env.prev_step_count = torch.zeros(1, dtype=torch.long)
    env.official_step_lims = [300]
    env.success_deadline_mode = "hard"
    env.success_decay_steps = 525.0
    env.use_dense_reward = True
    env.dense_shaping_coef = 1.0
    env.dense_success_reward = 1.0
    env.time_cost_per_step = 0.0
    env.cfg = SimpleNamespace(reward_coef=1.0)
    infos = {"dense_progress_trace": [[0.8, 0.2]], "step_count": [2]}
    rewards = env._rewards_from_progress_trace(torch.tensor([False]), infos, 4)
    assert rewards is not None
    assert math.isclose(float(rewards.sum()), 0.0, abs_tol=1e-6)
    assert float(rewards[0, 0]) > 0
    assert float(rewards[0, 1]) < 0
    assert float(rewards[0, 2]) == 0.0


def test_missing_progress_trace_falls_back():
    env = RoboDojoEnv.__new__(RoboDojoEnv)
    env.num_envs = 1
    assert env._rewards_from_progress_trace(torch.tensor([False]), {}, 4) is None


def test_time_cost_scales_with_executed_steps():
    full = chunk_reward_from_cumulative(0.0, 0.0, TIME_COST, 50)
    short = chunk_reward_from_cumulative(0.0, 0.0, TIME_COST, 10)
    assert math.isclose(full, 5 * short)
