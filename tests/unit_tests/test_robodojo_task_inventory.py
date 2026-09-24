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

from omegaconf import OmegaConf

from rlinf.envs.robodojo.task_inventory import (
    COMPETITION_TASKS,
    get_task_horizon,
    list_task_records,
    opendm_supported_task_names,
)


def test_stack_bowls_horizon():
    assert get_task_horizon("stack_bowls") == 800


def test_opendm_skips_competition_tasks():
    names = set(opendm_supported_task_names())
    assert "stack_bowls" in names
    assert "hang_mugs" in names
    for skipped in COMPETITION_TASKS:
        assert skipped not in names


def test_opendm_supported_count_is_51():
    records = list_task_records()
    assert len(records) == 54
    names = opendm_supported_task_names()
    assert len(names) == 51
    assert "fasten_screws" in names
    assert get_task_horizon("fasten_screws") == 1900
    for skipped in COMPETITION_TASKS:
        match = next(r for r in records if r["name"] == skipped)
        assert match["competition"]


def test_task_names_even_split_like_robotwin():
    cfg = OmegaConf.create(
        {
            "task_config": {"task_name": "stack_bowls"},
            "task_names": ["stack_bowls", "hang_mugs"],
        }
    )
    task_names = list(cfg.task_names)
    num_envs = 4
    task_ids = [env_id % len(task_names) for env_id in range(num_envs)]
    per_env = [task_names[i] for i in task_ids]
    assert per_env.count("stack_bowls") == 2
    assert per_env.count("hang_mugs") == 2
