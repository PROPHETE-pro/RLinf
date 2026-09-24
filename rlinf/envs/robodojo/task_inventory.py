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

"""Parse RoboDojo task inventory without importing Isaac."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Optional

import yaml

COMPETITION_TASKS = frozenset(
    {
        "imitate_sorting_sequence",
        "make_kong",
        "play_tic_tac_toe",
    }
)
DEFAULT_ROBOT_CONFIG = "dual_x5"
DEFAULT_ENV_CFG = "arx_x5"


def resolve_robodojo_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_path = os.environ.get("ROBODOJO_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path("/kpfs_ssd/data/ruitong_gan/RoboDojo").resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def _step_lim_from_task_py(path: Path) -> Optional[int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Attribute) and target.attr == "step_lim"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
            return int(node.value.value)
        if isinstance(node.value, ast.Num):  # py<3.8 compat
            return int(node.value.n)
    return None


def _robot_config_for_task(task_name: str, index: dict[str, Any]) -> str:
    common = index.get("common") or {}
    tasks = index.get("tasks") or {}
    task_info = tasks.get(task_name) or {}
    return str(
        task_info.get(
            "robot_config",
            common.get("robot_config", DEFAULT_ROBOT_CONFIG),
        )
    )


def list_task_records(robodojo_path: Optional[str] = None) -> list[dict[str, Any]]:
    root = resolve_robodojo_path(robodojo_path)
    task_dir = root / "task" / "RoboDojo" / "tasks"
    config_dir = root / "task" / "RoboDojo" / "config"
    index_path = config_dir / "_task.yml"
    index = _load_yaml(index_path) if index_path.exists() else {}

    records: list[dict[str, Any]] = []
    if not task_dir.is_dir():
        return records
    for path in sorted(task_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        name = path.stem
        robot_config = _robot_config_for_task(name, index)
        config_path = config_dir / f"{name}.yml"
        records.append(
            {
                "name": name,
                "step_lim": _step_lim_from_task_py(path),
                "robot_config": robot_config,
                "opendm_supported": robot_config == DEFAULT_ROBOT_CONFIG
                and name not in COMPETITION_TASKS,
                "competition": name in COMPETITION_TASKS
                or robot_config != DEFAULT_ROBOT_CONFIG,
                "config_exists": config_path.exists(),
            }
        )
    return records


def get_task_record(
    task_name: str, robodojo_path: Optional[str] = None
) -> dict[str, Any]:
    for record in list_task_records(robodojo_path):
        if record["name"] == task_name:
            return record
    raise KeyError(f"unknown RoboDojo task {task_name!r}")


def get_task_horizon(task_name: str, robodojo_path: Optional[str] = None) -> int:
    record = get_task_record(task_name, robodojo_path)
    step_lim = record.get("step_lim")
    if step_lim is None:
        raise ValueError(f"task {task_name!r} has no parsed step_lim")
    return int(step_lim)


def opendm_supported_task_names(robodojo_path: Optional[str] = None) -> list[str]:
    return [
        record["name"]
        for record in list_task_records(robodojo_path)
        if record["opendm_supported"] and record["config_exists"]
    ]
