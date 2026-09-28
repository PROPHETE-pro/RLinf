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

import pytest

from rlinf.envs.robodojo.isaac_worker import resolve_layout_pack_id
from rlinf.envs.robodojo.subproc_vector_env import _child_env


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
