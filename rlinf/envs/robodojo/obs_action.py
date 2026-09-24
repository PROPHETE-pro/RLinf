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

"""Joint-state pack/unpack for dual ARX X5 (14-D absolute joint).

Layout matches OpenDM Dual ARX5 / RoboDojo ``action_type=joint``:
``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]``.

This module is imported both from RLinf (parent) and as a sibling of
``isaac_worker.py`` (RoboDojo python). Keep it free of ``rlinf`` imports.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

ACTION_DIM = 14
DUAL_X5_DIM_INFO: dict[str, list[int]] = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
DUAL_ARX5_ROBOT_TYPES = frozenset({"Dual ARX5", "dual arx5", "dual_arx5"})
DUAL_ARX5_STATE_DESC = (
    ["joint"] * 6 + ["gripper"] + ["joint"] * 6 + ["gripper"]
)
HEAD_CAMERA_CANDIDATES = ("cam_head", "cam_high", "cam_third_view")
LEFT_WRIST_CANDIDATES = ("cam_left_wrist", "cam_left", "cam_arm_left")
RIGHT_WRIST_CANDIDATES = ("cam_right_wrist", "cam_right", "cam_arm_right")
DUMMY_IMAGE_HW = (480, 640)


def unpack_joint_action(vec: Any) -> dict[str, np.ndarray]:
    """Unpack a 14-D joint vector into a RoboDojo action dict."""
    packed = np.asarray(vec, dtype=np.float32).reshape(-1)
    if packed.size != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM}-D joint action, got {packed.shape}")
    return {
        "left_arm_joint_state": packed[0:6].copy(),
        "left_ee_joint_state": packed[6:7].copy(),
        "right_arm_joint_state": packed[7:13].copy(),
        "right_ee_joint_state": packed[13:14].copy(),
    }


def pack_joint_state(state_dict: Mapping[str, Any]) -> np.ndarray:
    """Pack a RoboDojo ``obs['state']`` dict into a 14-D vector."""
    parts = []
    for key, dim in (
        ("left_arm_joint_state", 6),
        ("left_ee_joint_state", 1),
        ("right_arm_joint_state", 6),
        ("right_ee_joint_state", 1),
    ):
        if key not in state_dict:
            raise KeyError(f"missing state key {key!r}")
        arr = np.asarray(state_dict[key], dtype=np.float32).reshape(-1)
        if arr.size != dim:
            raise ValueError(f"{key} expected dim {dim}, got {arr.shape}")
        parts.append(arr)
    return np.concatenate(parts, axis=0)


def _as_hwc_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _vision_rgb(vision: Mapping[str, Any], candidates: tuple[str, ...]) -> np.ndarray:
    for name in candidates:
        cam = vision.get(name)
        if not isinstance(cam, dict):
            continue
        color = cam.get("color", cam.get("rgb"))
        if color is not None:
            return _as_hwc_uint8(color)
    raise KeyError(f"no RGB camera among {candidates}; have {list(vision)}")


def extract_policy_obs(raw_obs: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one RoboDojo obs dict into the SubEnv wire format."""
    vision = raw_obs.get("vision") or {}
    state = raw_obs.get("state") or {}
    instruction = raw_obs.get("instruction") or ""
    if isinstance(instruction, (list, tuple)):
        instruction = instruction[0] if instruction else ""
    return {
        "full_image": _vision_rgb(vision, HEAD_CAMERA_CANDIDATES),
        "left_wrist_image": _vision_rgb(vision, LEFT_WRIST_CANDIDATES),
        "right_wrist_image": _vision_rgb(vision, RIGHT_WRIST_CANDIDATES),
        "state": pack_joint_state(state).astype(np.float32),
        "instruction": str(instruction),
    }


def minimal_obs(instruction: str = "") -> dict[str, Any]:
    dummy = np.zeros((*DUMMY_IMAGE_HW, 3), dtype=np.uint8)
    return {
        "full_image": dummy,
        "left_wrist_image": dummy.copy(),
        "right_wrist_image": dummy.copy(),
        "state": np.zeros(ACTION_DIM, dtype=np.float32),
        "instruction": instruction,
    }


def is_dual_arx5_robot_type(robot_type: Any) -> bool:
    if robot_type is None:
        return False
    return str(robot_type).strip() in DUAL_ARX5_ROBOT_TYPES
