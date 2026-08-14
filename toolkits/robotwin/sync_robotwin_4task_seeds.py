#!/usr/bin/env python3
"""Sync RoboTwin 4-task seeds from RoboTwin eval_result into RLinf seed JSON files.

RoboTwin stores seeds as:
  {task: {demo_clean: {success_seeds: [...]}, demo_randomized: {...}}}

RLinf expects:
  {task: {task_name: task, success_seeds: [...]}}

Usage:
  python toolkits/robotwin/sync_robotwin_4task_seeds.py \\
      --robotwin-seeds /path/to/RoboTwin/eval_result/4task/eval_seeds.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_TASKS = (
    "blocks_ranking_size",
    "hanging_mug",
    "open_microwave",
    "place_mouse_pad",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _seeds_dir() -> Path:
    return _repo_root() / "rlinf/envs/robotwin/seeds"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def _extract_seeds(
    robotwin_data: dict,
    task_name: str,
    task_config: str,
) -> list[int] | None:
    task_entry = robotwin_data.get(task_name)
    if task_entry is None:
        return None
    if isinstance(task_entry, dict) and "success_seeds" in task_entry:
        return list(task_entry["success_seeds"])
    if isinstance(task_entry, dict):
        config_entry = task_entry.get(task_config)
        if isinstance(config_entry, dict):
            seeds = config_entry.get("success_seeds")
            if seeds is not None:
                return list(seeds)
    return None


def _merge_task(
    target: dict,
    task_name: str,
    success_seeds: list[int],
    *,
    overwrite: bool,
) -> bool:
    if (
        not overwrite
        and task_name in target
        and target[task_name].get("success_seeds")
    ):
        return False
    target[task_name] = {
        "task_name": task_name,
        "success_seeds": sorted(success_seeds),
    }
    return True


def sync_seeds(
    robotwin_seeds_path: Path,
    *,
    tasks: list[str],
    overwrite: bool,
) -> None:
    robotwin_data = _load_json(robotwin_seeds_path)
    seeds_dir = _seeds_dir()

    output_specs = [
        ("demo_clean", seeds_dir / "eval_seeds.json"),
        ("demo_randomized", seeds_dir / "eval_seeds_demo_randomized.json"),
        ("demo_randomized", seeds_dir / "train_seeds_demo_randomized.json"),
    ]

    for task_config, output_path in output_specs:
        payload = _load_json(output_path)
        updated_tasks: list[str] = []
        for task_name in tasks:
            success_seeds = _extract_seeds(robotwin_data, task_name, task_config)
            if success_seeds is None:
                print(
                    f"[skip] {output_path.name}: no {task_config} seeds for {task_name}"
                )
                continue
            if _merge_task(payload, task_name, success_seeds, overwrite=overwrite):
                updated_tasks.append(task_name)
        _write_json(output_path, payload)
        if updated_tasks:
            print(
                f"[write] {output_path} <= {task_config}: "
                + ", ".join(
                    f"{task}({len(payload[task]['success_seeds'])})"
                    for task in updated_tasks
                )
            )
        else:
            print(f"[keep] {output_path} unchanged")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync RoboTwin 4-task seeds into RLinf JSON seed files."
    )
    parser.add_argument(
        "--robotwin-seeds",
        default="/mnt/pfs/7wsqem/grt/RoboTwin/eval_result/4task/eval_seeds.json",
        help="Path to RoboTwin nested eval_seeds.json.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Task names to sync.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing task entries even if seeds already exist.",
    )
    args = parser.parse_args()
    sync_seeds(
        Path(args.robotwin_seeds),
        tasks=args.tasks,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
