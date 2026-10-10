# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import numpy as np
import pytest

from rlinf.envs.robodojo.isaac_worker import (
    _capture_head_rgb,
    _step_chunk,
    resolve_layout_pack_id,
)
from rlinf.envs.robodojo.subproc_vector_env import (
    IsaacSubEnvWorker,
    SubprocVectorEnv,
    _child_env,
)


def test_child_env_drops_k8s_and_stdbuf_preload(monkeypatch):
    monkeypatch.setenv("KAIC_FOO", "drop-me")
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("FOO_SERVICE_HOST", "10.0.0.2")
    monkeypatch.setenv("FOO_SERVICE_PORT", "8080")
    monkeypatch.setenv("FOO_PORT_8080_TCP", "tcp://10.0.0.2:8080")
    monkeypatch.setenv("FOO_PORT", "tcp://10.0.0.2:8080")
    monkeypatch.setenv("LD_PRELOAD", "/usr/libexec/coreutils/libstdbuf.so")
    monkeypatch.setenv("_STDBUF_O", "L")
    monkeypatch.setenv("KEEP_ME", "yes")
    env = _child_env("/tmp/robodojo")
    assert "KAIC_FOO" not in env
    assert "KUBERNETES_SERVICE_HOST" not in env
    assert "FOO_SERVICE_HOST" not in env
    assert "FOO_SERVICE_PORT" not in env
    assert "FOO_PORT_8080_TCP" not in env
    assert "FOO_PORT" not in env
    assert "LD_PRELOAD" not in env
    assert "_STDBUF_O" not in env
    assert env["KEEP_ME"] == "yes"
    assert env["ROBODOJO_PATH"] == "/tmp/robodojo"
    assert env["PYTHONPATH"].startswith("/tmp/robodojo")


def test_child_env_drops_cuda_mask_and_records_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    env = _child_env("/tmp/robodojo")
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["ROBODOJO_ISAAC_DEVICE"] == "1"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    env = _child_env("/tmp/robodojo", extra={"CUDA_VISIBLE_DEVICES": "0"})
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["ROBODOJO_ISAAC_DEVICE"] == "0"


def test_child_env_extra_cannot_reintroduce_ld_preload():
    env = _child_env(
        "/tmp/robodojo",
        extra={"LD_PRELOAD": "/usr/libexec/coreutils/libstdbuf.so", "FOO": "1"},
    )
    assert "LD_PRELOAD" not in env
    assert env["FOO"] == "1"


def test_resolve_layout_pack_maps_missing_id(tmp_path):
    root = tmp_path / "arx_x5"
    for pack in (0, 1, 2):
        (root / str(pack)).mkdir(parents=True)
    assert resolve_layout_pack_id(root, 0) == 0
    assert resolve_layout_pack_id(root, 2) == 2
    assert resolve_layout_pack_id(root, 36044) == 36044 % 3


def test_resolve_layout_pack_missing_root_does_not_create(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="read-only"):
        resolve_layout_pack_id(missing, 0)
    assert not missing.exists()


class _ChunkEnv:
    def __init__(self, end_after: Optional[int] = None):
        self.end_flag = [False]
        self.success = [False]
        self.take_count = 0
        self.get_obs_count = 0
        self.end_after = end_after
        self.take_action_cnt = [0]
        self.step_lim = 100

    def take_action(self, action):
        del action
        self.take_count += 1
        self.take_action_cnt[0] = self.take_count

        if self.end_after is not None and self.take_count >= self.end_after:
            self.end_flag[0] = True

    def get_obs(self):
        self.get_obs_count += 1
        color = np.zeros((2, 2, 3), dtype=np.uint8)
        state = {
            "left_arm_joint_state": np.zeros(6, dtype=np.float32),
            "left_ee_joint_state": np.zeros(1, dtype=np.float32),
            "right_arm_joint_state": np.zeros(6, dtype=np.float32),
            "right_ee_joint_state": np.zeros(1, dtype=np.float32),
        }
        return {
            "vision": {
                "cam_head": {"color": color},
                "cam_left_wrist": {"color": color.copy()},
                "cam_right_wrist": {"color": color.copy()},
            },
            "state": state,
            "instruction": "task",
        }


def test_step_chunk_reads_policy_obs_once_when_video_off(monkeypatch):
    monkeypatch.delenv("ROBODOJO_SAVE_VIDEO", raising=False)
    env = _ChunkEnv()

    def _forbid_head(_env):
        raise AssertionError("head capture should stay off when video is disabled")

    monkeypatch.setattr(
        "rlinf.envs.robodojo.isaac_worker._capture_head_rgb", _forbid_head
    )
    payload = _step_chunk(env, np.zeros((4, 14), dtype=np.float32))
    assert env.take_count == 4
    assert env.get_obs_count == 1
    assert "video_frames" not in payload
    assert payload["left_wrist_image"].shape[-1] == 3


def test_step_chunk_records_head_every_action(monkeypatch):
    monkeypatch.setenv("ROBODOJO_SAVE_VIDEO", "1")
    monkeypatch.setenv("ROBODOJO_VIDEO_STRIDE", "-1")
    env = _ChunkEnv(end_after=2)
    frames = []

    def _head(_env):
        frame = np.full((2, 2, 3), len(frames) + 1, dtype=np.uint8)
        frames.append(frame)
        return frame

    monkeypatch.setattr("rlinf.envs.robodojo.isaac_worker._capture_head_rgb", _head)
    payload = _step_chunk(env, np.zeros((4, 14), dtype=np.float32))
    assert env.take_count == 2
    assert env.get_obs_count == 1
    assert len(frames) == 2
    assert len(payload["video_frames"]) == 2
    assert payload["video_frames"][0].shape == (2, 2, 3)
    assert not isinstance(payload["video_frames"][0], dict)


def test_capture_head_rgb_requests_only_head_camera():
    class _Capture:
        def __init__(self):
            self.cam_ids = None

        def step(self, env_ids=None, cam_ids=None):
            del env_ids
            self.cam_ids = list(cam_ids)
            image = np.zeros((2, 2, 4), dtype=np.uint8)
            image[..., 0] = 7
            return [{"rgb": [{"data": image}]}]

    capture = _Capture()

    class _Cameras:
        camera_names = [["cam_left_wrist", "cam_head", "cam_right_wrist"]]

    class _Obs:
        camera_manager = _Cameras()
        capture_manager = capture

    class _Env:
        obs_manager = _Obs()
        renders = 0

        def render(self):
            self.renders += 1

    env = _Env()
    frame = _capture_head_rgb(env)
    assert capture.cam_ids == [1]
    assert env.renders == 1
    assert frame.shape == (2, 2, 3)
    assert int(frame[0, 0, 0]) == 7


class _FakeProc:
    def poll(self):
        return None


class _FakeConn:
    def __init__(self, env_id, log, *, timeout=False, payload=None):
        self.env_id = env_id
        self.log = log
        self.timeout = timeout
        self.payload = payload

    def send(self, msg):
        self.log.append(("send", self.env_id, msg[0]))

    def poll(self, wait):
        del wait
        self.log.append(("poll", self.env_id))
        return not self.timeout

    def recv(self):
        self.log.append(("recv", self.env_id))
        return self.payload


def _ipc_payload():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    return {
        "full_image": image,
        "left_wrist_image": image,
        "right_wrist_image": image,
        "state": np.zeros(14, dtype=np.float32),
        "instruction": "task",
        "reward": 0,
        "terminated": 0,
        "truncated": 0,
        "success": False,
    }


def _bare_worker(env_id, conn) -> IsaacSubEnvWorker:
    worker = IsaacSubEnvWorker.__new__(IsaacSubEnvWorker)
    worker.env_id = env_id
    worker.conn = conn
    worker.process = _FakeProc()
    worker.step_timeout_sec = 0.01
    worker.ready_timeout_sec = 0.01
    worker.on_timeout = "truncate"
    worker.max_respawns = 10
    worker.respawn_count = 0
    worker._last_obs = None
    worker.env_seed = env_id
    worker.respawn_reasons = []

    def _respawn(elapsed, reason):
        del elapsed
        worker.respawn_reasons.append(reason)

    worker._force_kill_and_respawn = _respawn
    return worker


def _bare_vector_env(workers) -> SubprocVectorEnv:
    env = SubprocVectorEnv.__new__(SubprocVectorEnv)
    env.workers = workers
    env.n_envs = len(workers)
    env.extra_env = {}
    return env


def test_vector_step_sends_before_recv_and_timeout_is_local():
    log = []
    healthy = _bare_worker(0, _FakeConn(0, log, payload=_ipc_payload()))
    stalled = _bare_worker(1, _FakeConn(1, log, timeout=True))
    env = _bare_vector_env([healthy, stalled])
    obs, _reward, _terminated, truncated, _info, _video = env.step(
        np.zeros((2, 1, 14), dtype=np.float32)
    )
    send_at = [i for i, event in enumerate(log) if event[0] == "send"]
    wait_at = [i for i, event in enumerate(log) if event[0] in ("poll", "recv")]
    assert send_at and wait_at
    assert max(send_at) < min(wait_at)
    assert healthy.respawn_reasons == []
    assert stalled.respawn_reasons == ["step_timeout"]
    assert int(np.asarray(truncated[1]).reshape(-1)[0]) == 1
    assert obs[0]["instruction"] == "task"


def test_vector_reset_sends_before_recv():
    log = []
    workers = [
        _bare_worker(
            i,
            _FakeConn(
                i,
                log,
                payload={
                    "status": "ok",
                    "obs": {
                        "full_image": np.zeros((2, 2, 3), dtype=np.uint8),
                        "left_wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
                        "right_wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
                        "state": np.zeros(14, dtype=np.float32),
                        "instruction": "task",
                    },
                },
            ),
        )
        for i in range(2)
    ]
    env = _bare_vector_env(workers)
    env.reset(env_idx=[0, 1], env_seeds=[3, 4])
    send_at = [i for i, event in enumerate(log) if event[0] == "send"]
    wait_at = [i for i, event in enumerate(log) if event[0] in ("poll", "recv")]
    assert [log[i][2] for i in send_at] == ["reset", "reset"]
    assert max(send_at) < min(wait_at)
    assert workers[0].env_seed == 3
    assert workers[1].env_seed == 4
