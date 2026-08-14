import torch

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv


def _make_env_stub(
    *,
    use_dense_reward: bool = True,
    use_rel_reward: bool = True,
    use_custom_reward: bool = False,
    num_envs: int = 2,
):
    env = object.__new__(RoboTwinEnv)
    env.num_envs = num_envs
    env.device = torch.device("cpu")
    env.use_dense_reward = use_dense_reward
    env.use_rel_reward = use_rel_reward
    env.use_custom_reward = use_custom_reward
    env.cfg = type("Cfg", (), {"reward_coef": 1.0})()
    env.prev_step_reward = torch.zeros(num_envs, dtype=torch.float32)
    return env


def test_cal_chunk_rewards_places_dense_delta_on_last_slot():
    env = _make_env_stub()
    step_reward = torch.tensor([0.05, 0.10], dtype=torch.float32)
    terminations = torch.tensor([False, False])

    chunk_rewards = env._cal_chunk_rewards(step_reward, chunk_step=50, terminations=terminations)

    assert chunk_rewards.shape == (2, 50)
    assert chunk_rewards[:, :-1].sum() == 0.0
    assert torch.allclose(chunk_rewards[:, -1], step_reward)


def test_prepare_step_reward_dense_uses_incremental_signal():
    env = _make_env_stub(use_dense_reward=True, use_rel_reward=True)
    terminations = torch.tensor([False, False])

    first = env._prepare_step_reward([0.10, 0.20], terminations)
    second = env._prepare_step_reward([0.25, 0.30], terminations)

    assert torch.allclose(first, torch.tensor([0.10, 0.20]))
    assert torch.allclose(second, torch.tensor([0.15, 0.10]))


def test_prepare_step_reward_dense_nonzero_without_termination():
    env = _make_env_stub(use_dense_reward=True, use_rel_reward=True)
    terminations = torch.tensor([False, True])

    delta = env._prepare_step_reward([0.12, 1.00], terminations)
    chunk_rewards = env._cal_chunk_rewards(delta, chunk_step=8, terminations=terminations)

    assert torch.allclose(delta, torch.tensor([0.12, 1.00]))
    assert chunk_rewards[0, -1] == 0.12
    assert chunk_rewards[1, -1] == 1.00
    assert chunk_rewards.sum() == 1.12


def test_prepare_step_reward_sparse_custom_binary():
    env = _make_env_stub(use_dense_reward=False, use_custom_reward=True, use_rel_reward=True)
    terminations = torch.tensor([False, True], dtype=torch.bool)

    first = env._prepare_step_reward(None, terminations)
    assert torch.allclose(first, torch.tensor([0.0, 0.0]))

    second = env._prepare_step_reward(None, torch.tensor([False, True], dtype=torch.bool))
    assert torch.allclose(second, torch.tensor([0.0, 1.0]))
