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

"""Tests for RoboTwin multitask task_names even-split logic."""

from omegaconf import OmegaConf

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv


def test_task_names_even_split_logic():
    """Pure logic check matching RoboTwinEnv._init_task_names assignment."""
    task_names = [
        "open_microwave",
        "hanging_mug",
        "place_mouse_pad",
        "blocks_ranking_size",
    ]
    num_envs = 32
    num_tasks = len(task_names)
    assert num_envs % num_tasks == 0
    task_ids = [env_id % num_tasks for env_id in range(num_envs)]
    per_env = [task_names[i] for i in task_ids]
    for t in task_names:
        assert per_env.count(t) == num_envs // num_tasks


def test_init_task_names_on_stub(monkeypatch):
    """Call _init_task_names without constructing full env / SAPIEN."""
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "auto_reset": False,
            "use_rel_reward": False,
            "ignore_terminations": False,
            "group_size": 1,
            "use_fixed_reset_state_ids": False,
            "use_custom_reward": True,
            "use_dense_reward": False,
            "video_cfg": {},
            "total_num_envs": 8,
            "task_names": ["place_mouse_pad", "open_microwave"],
            "task_config": {"task_name": "place_mouse_pad"},
            "assets_path": "/tmp",
            "max_episode_steps": 100,
        }
    )

    env = RoboTwinEnv.__new__(RoboTwinEnv)
    env.cfg = cfg
    env.num_envs = 8
    env._init_task_names()
    assert env.num_tasks == 2
    assert env.per_env_task_names == [
        "place_mouse_pad",
        "open_microwave",
        "place_mouse_pad",
        "open_microwave",
        "place_mouse_pad",
        "open_microwave",
        "place_mouse_pad",
        "open_microwave",
    ]
