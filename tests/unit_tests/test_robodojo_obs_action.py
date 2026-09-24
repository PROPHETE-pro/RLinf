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

import numpy as np

from rlinf.envs import SupportedEnvType, get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.robodojo.obs_action import (
    ACTION_DIM,
    extract_policy_obs,
    is_dual_arx5_robot_type,
    pack_joint_state,
    unpack_joint_action,
)
from rlinf.envs.robodojo.robodojo_env import RoboDojoEnv
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv


def test_pack_unpack_roundtrip():
    vec = np.linspace(0.1, 1.4, ACTION_DIM, dtype=np.float32)
    action = unpack_joint_action(vec)
    packed = pack_joint_state(action)
    np.testing.assert_allclose(packed, vec, rtol=0, atol=1e-6)
    assert action["left_arm_joint_state"].shape == (6,)
    assert action["left_ee_joint_state"].shape == (1,)
    assert action["right_arm_joint_state"].shape == (6,)
    assert action["right_ee_joint_state"].shape == (1,)


def test_extract_policy_obs_cameras():
    h, w = 12, 16
    raw = {
        "vision": {
            "cam_head": {"color": np.zeros((h, w, 3), dtype=np.uint8)},
            "cam_left_wrist": {"color": np.ones((h, w, 3), dtype=np.uint8)},
            "cam_right_wrist": {"color": np.full((h, w, 3), 2, dtype=np.uint8)},
        },
        "state": {
            "left_arm_joint_state": np.zeros(6, dtype=np.float32),
            "left_ee_joint_state": np.array([0.5], dtype=np.float32),
            "right_arm_joint_state": np.ones(6, dtype=np.float32),
            "right_ee_joint_state": np.array([0.2], dtype=np.float32),
        },
        "instruction": "stack the bowls",
    }
    obs = extract_policy_obs(raw)
    assert obs["full_image"].shape == (h, w, 3)
    assert obs["state"].shape == (ACTION_DIM,)
    assert obs["instruction"] == "stack the bowls"
    np.testing.assert_allclose(obs["state"][6], 0.5)


def test_prepare_actions_robodojo_passthrough():
    raw = np.arange(2 * 50 * 14, dtype=np.float32).reshape(2, 50, 14)
    out = prepare_actions(
        env_type="robodojo",
        model_type="opendm_dm05",
        raw_chunk_actions=raw,
        num_action_chunks=50,
        action_dim=14,
    )
    np.testing.assert_array_equal(out, raw)


def test_robotwin_env_cls_unchanged():
    assert get_env_cls("robotwin") is RoboTwinEnv
    assert SupportedEnvType.ROBOTWIN.value == "robotwin"
    assert SupportedEnvType.ROBODOJO.value == "robodojo"
    assert get_env_cls("robodojo") is RoboDojoEnv


def test_dual_arx5_robot_type_gate():
    assert is_dual_arx5_robot_type("Dual ARX5")
    assert is_dual_arx5_robot_type("dual_arx5")
    assert not is_dual_arx5_robot_type("Aloha RoboTwin2")
