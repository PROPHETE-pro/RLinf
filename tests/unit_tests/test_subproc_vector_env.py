# Copyright 2025 The RLinf Authors.

from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest

from rlinf.envs.robotwin.subproc_vector_env import (
    SubprocSubEnvWorker,
    SubprocVectorEnv,
    _make_recovery_info,
    build_robotwin_task_args,
    normalize_subenv_step_result,
)
from rlinf.envs.utils import list_of_dict_to_dict_of_list


def test_build_robotwin_task_args_keys():
    task_config = {
        "task_name": "open_microwave",
        "step_lim": 1500,
        "embodiment": ["aloha-agilex"],
        "save_path": "./data",
    }
    assets_path = "/tmp/assets"
    embodiment_types = {"aloha-agilex": {"file_path": "robots/aloha"}}
    camera_config = {"D435": {"h": 240, "w": 320}}
    embodiment_yaml = {"joint_names": ["j1"]}
    fake_global_configs = mock.MagicMock(CONFIGS_PATH="/tmp/configs/")

    with mock.patch.dict("sys.modules", {"envs._GLOBAL_CONFIGS": fake_global_configs}), mock.patch(
        "builtins.open", mock.mock_open()
    ), mock.patch(
        "rlinf.envs.robotwin.subproc_vector_env.yaml.load",
        side_effect=[
            embodiment_types,
            camera_config,
            embodiment_yaml,
            embodiment_yaml,
        ],
    ):
        args = build_robotwin_task_args(
            task_config, n_envs=2, assets_path=assets_path
        )

    assert args["task_name"] == "open_microwave"
    assert args["n_envs"] == 2
    assert args["action_dim"] == 14
    assert args["eval_mode"] is True
    assert args["render_freq"] == 0
    assert args["save_path"].endswith("/open_microwave_reward")


def _make_worker(**overrides) -> SubprocSubEnvWorker:
    worker = SubprocSubEnvWorker.__new__(SubprocSubEnvWorker)
    worker.env_id = overrides.get("env_id", 0)
    worker.task_name = "open_microwave"
    worker.args = {"task_name": "open_microwave"}
    worker.env_seed = overrides.get("env_seed", 42)
    worker.instruction_type = "seen"
    worker.step_timeout_sec = overrides.get("step_timeout_sec", 0.2)
    worker.max_respawns = overrides.get("max_respawns", 2)
    worker.on_timeout = overrides.get("on_timeout", "truncate")
    worker.respawn_count = overrides.get("respawn_count", 0)
    worker._mp_context = None
    worker.process = mock.MagicMock()
    worker.parent_conn = mock.MagicMock()
    worker._last_obs = overrides.get("_last_obs")
    return worker


def test_force_kill_and_respawn_increments_count_and_bumps_seed():
    worker = _make_worker(env_seed=42, max_respawns=2)
    worker._kill_process = mock.MagicMock()
    worker._spawn_process = mock.MagicMock()
    worker._send_recv = mock.MagicMock(return_value=None)

    worker._force_kill_and_respawn(elapsed=60.0, reason="step_timeout")

    worker._kill_process.assert_called_once()
    worker._spawn_process.assert_called_once_with(43)
    worker._send_recv.assert_called_once_with(
        "reset", 43, timeout=worker.step_timeout_sec
    )
    assert worker.respawn_count == 1


def test_force_kill_and_respawn_max_limit_raises():
    worker = _make_worker(max_respawns=0, respawn_count=0)
    worker._kill_process = mock.MagicMock()

    with pytest.raises(RuntimeError, match="subproc_max_respawns"):
        worker._force_kill_and_respawn(elapsed=60.0, reason="step_timeout")


def test_truncate_step_result_shape():
    worker = _make_worker()
    worker.respawn_count = 1
    worker._send_recv = mock.MagicMock(
        return_value={
            "full_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "left_wrist_image": None,
            "right_wrist_image": None,
            "state": np.zeros(14, dtype=np.float32),
            "instruction": "test",
        }
    )

    result = worker._truncate_step_result()

    assert result["truncated"][0] == 1
    assert result["reward"][0] == 0
    assert result["info"]["success"] is False
    assert result["info"]["subenv_timeout"] is True
    assert result["info"]["sub_env_id"] == 0
    assert result["info"]["respawn_count"] == 1
    assert result["terminated"].shape == (1,)
    assert result["truncated"].shape == (1,)
    assert result["terminated"].dtype == np.int32
    assert result["truncated"].dtype == np.int32
    worker._send_recv.assert_called_once_with(
        "get_obs", None, timeout=worker.step_timeout_sec
    )


def test_normalize_subenv_step_result_mixed_with_robotwin_chunk_step():
    normal = {
        "obs": {},
        "reward": np.array([1.0], dtype=np.float32),
        "terminated": np.array([0], dtype=np.int32),
        "truncated": np.array([0], dtype=np.int32),
        "info": {"success": False},
    }
    recovered = normalize_subenv_step_result(
        {
            "obs": {},
            "reward": 0,
            "terminated": False,
            "truncated": True,
            "info": _make_recovery_info(0, 1),
        }
    )
    merged_terminations = [normal["terminated"], recovered["terminated"]]
    merged_truncations = [normal["truncated"], recovered["truncated"]]
    term_tensor = np.array(merged_terminations).reshape(-1)
    trunc_tensor = np.array(merged_truncations).reshape(-1)
    assert term_tensor.shape == (2,)
    assert trunc_tensor.shape == (2,)
    assert trunc_tensor[1] == 1


def test_recovery_info_includes_success_key():
    info = _make_recovery_info(env_id=3, respawn_count=2)
    assert info["success"] is False
    assert info["subenv_timeout"] is True
    assert info["sub_env_id"] == 3


def test_list_of_dict_to_dict_of_list_mixed_keys():
    mixed = [
        {"success": False},
        {"success": False, "subenv_timeout": True, "sub_env_id": 0},
    ]
    out = list_of_dict_to_dict_of_list(mixed)
    assert out["success"] == [False, False]
    assert out["subenv_timeout"] == [None, True]
    assert out["sub_env_id"] == [None, 0]


def test_recovery_failed_step_result_does_not_raise():
    worker = _make_worker()
    worker.respawn_count = 1
    worker._last_obs = {
        "full_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "left_wrist_image": None,
        "right_wrist_image": None,
        "state": np.zeros(14, dtype=np.float32),
        "instruction": "cached",
    }

    result = worker._recovery_failed_step_result("spawn timed out")

    assert result["truncated"][0] == 1
    assert result["info"]["success"] is False
    assert result["info"]["subenv_recovery_failed"] is True
    assert result["info"]["recovery_error"] == "spawn timed out"


def test_subproc_vector_env_mixed_step_results():
    hang_worker = _make_worker(env_id=0)
    hang_worker.parent_conn.poll.return_value = False
    hang_worker._force_kill_and_respawn = mock.MagicMock()
    truncate_result = {
        "obs": {
            "full_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "left_wrist_image": None,
            "right_wrist_image": None,
            "state": np.zeros(14, dtype=np.float32),
            "instruction": "hang",
        },
        "reward": 0,
        "terminated": np.array([0], dtype=np.int32),
        "truncated": np.array([1], dtype=np.int32),
        "info": {
            "success": False,
            "subenv_timeout": True,
            "sub_env_id": 0,
            "respawn_count": 1,
        },
    }

    ok_worker = _make_worker(env_id=1)
    ok_worker.parent_conn.poll.return_value = True
    ok_worker.parent_conn.recv.return_value = {
        "obs": {
            "full_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "left_wrist_image": None,
            "right_wrist_image": None,
            "state": np.zeros(14, dtype=np.float32),
            "instruction": "ok",
        },
        "reward": 1.0,
        "terminated": False,
        "truncated": False,
        "info": {"success": False},
    }

    venv = SubprocVectorEnv.__new__(SubprocVectorEnv)
    venv.n_envs = 2
    venv.workers = [hang_worker, ok_worker]
    venv.step_timeout_sec = 0.2
    venv.on_timeout = "truncate"

    with mock.patch.object(
        hang_worker, "_truncate_step_result", return_value=truncate_result
    ):
        actions = np.zeros((2, 1, 14), dtype=np.float32)
        obs, rewards, terminated, truncated, infos = venv.step(actions)

    assert len(obs) == 2
    assert truncated[0] == 1
    assert rewards[0] == 0
    assert truncated[1] == 0
    assert rewards[1] == 1.0
    assert infos[0]["subenv_timeout"] is True
    assert infos[0]["success"] is False
    assert infos[1]["success"] is False
    hang_worker._force_kill_and_respawn.assert_called_once()


def test_get_obs_timeout_triggers_respawn_and_returns_obs():
    worker = _make_worker()
    worker.parent_conn.poll.return_value = False
    worker._force_kill_and_respawn = mock.MagicMock()
    recovered_obs = {
        "full_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "left_wrist_image": None,
        "right_wrist_image": None,
        "state": np.zeros(14, dtype=np.float32),
        "instruction": "recovered",
    }
    worker._obs_for_recovery = mock.MagicMock(return_value=recovered_obs)

    obs = worker.get_obs()

    worker.parent_conn.send.assert_called_once_with(("get_obs", None))
    worker._force_kill_and_respawn.assert_called_once()
    assert worker._force_kill_and_respawn.call_args.kwargs["reason"] == "get_obs_timeout"
    worker._obs_for_recovery.assert_called_once()
    assert obs is recovered_obs


def test_get_obs_timeout_recovery_failure_returns_fallback_obs():
    worker = _make_worker()
    worker.parent_conn.poll.return_value = False
    worker._force_kill_and_respawn = mock.MagicMock(
        side_effect=RuntimeError("spawn timed out")
    )
    fallback_obs = {
        "full_image": np.zeros((240, 320, 3), dtype=np.uint8),
        "left_wrist_image": None,
        "right_wrist_image": None,
        "state": np.zeros(14, dtype=np.float32),
        "instruction": "",
    }
    worker._obs_for_recovery = mock.MagicMock(return_value=fallback_obs)

    obs = worker.get_obs()

    worker._force_kill_and_respawn.assert_called_once()
    worker._obs_for_recovery.assert_called_once()
    assert obs is fallback_obs


def test_subproc_vector_env_get_obs_timeout_recovery():
    hang_worker = _make_worker(env_id=0)
    hang_worker.parent_conn.poll.return_value = False
    hang_worker._force_kill_and_respawn = mock.MagicMock()
    recovered_obs = {
        "full_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "left_wrist_image": None,
        "right_wrist_image": None,
        "state": np.zeros(14, dtype=np.float32),
        "instruction": "recovered",
    }
    hang_worker._obs_for_recovery = mock.MagicMock(return_value=recovered_obs)

    ok_worker = _make_worker(env_id=1)
    ok_worker.parent_conn.poll.return_value = True
    ok_worker.parent_conn.recv.return_value = {
        "full_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "left_wrist_image": None,
        "right_wrist_image": None,
        "state": np.zeros(14, dtype=np.float32),
        "instruction": "ok",
    }

    venv = SubprocVectorEnv.__new__(SubprocVectorEnv)
    venv.n_envs = 2
    venv.workers = [hang_worker, ok_worker]

    obs_list = venv.get_obs()

    assert len(obs_list) == 2
    assert obs_list[0]["instruction"] == "recovered"
    assert obs_list[1]["instruction"] == "ok"
    hang_worker._force_kill_and_respawn.assert_called_once()
    assert (
        hang_worker._force_kill_and_respawn.call_args.kwargs["reason"]
        == "get_obs_timeout"
    )


def test_subproc_vector_env_mixed_info_keys_compatible_with_robotwin_env():
    """Mixed normal/recovery info dicts must not crash list_of_dict_to_dict_of_list."""
    truncate_info = _make_recovery_info(env_id=0, respawn_count=1)
    normal_info = {"success": False}
    merged = list_of_dict_to_dict_of_list([normal_info, truncate_info])
    assert merged["success"] == [False, False]
    assert merged["subenv_timeout"] == [None, True]
