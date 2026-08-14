#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
"""Manual smoke helpers for RoboTwin LWD (offline→online + per-task success).

Run from the RLinf repo root with your training env activated.

Examples
--------
# 1) Unit tests (no GPU / no env needed)
pytest tests/unit_tests/test_divl.py tests/unit_tests/test_qam.py -q

# 2) Env multitask split check (needs RoboTwin assets; no training)
python examples/embodiment/scripts/lwd_smoke_verify.py env-split \\
  --num-envs 32 --tasks open_microwave,hanging_mug,place_mouse_pad,blocks_ranking_size

# 3) Parse TensorBoard / JSONL logs for per-task success
python examples/embodiment/scripts/lwd_smoke_verify.py task-success \\
  --log-json /path/to/metrics.jsonl

# 4) Offline → online training smoke (Hydra overrides via run_async.sh; edit demo load_path first)
# Logs always go to RLinf/logs/<timestamp>-<config>/
# Offline (demo only):
export ROBOT_PLATFORM=ALOHA
CUDA_VISIBLE_DEVICES=0 bash examples/embodiment/run_async.sh robotwin_lwd_openpi_pi05_1task \\
  runner.lwd_stage=offline algorithm.allow_demo_only=true \\
  algorithm.demo_ratio=1.0 algorithm.replay_buffer.min_buffer_size=0 \\
  algorithm.demo_buffer.load_path=/path/to/demo_buffer \\
  runner.max_epochs=2

# Online (mixed B_off ∪ B_on):
CUDA_VISIBLE_DEVICES=0 bash examples/embodiment/run_async.sh robotwin_lwd_openpi_pi05_1task \\
  runner.lwd_stage=online algorithm.demo_ratio=0.5 \\
  algorithm.demo_buffer.load_path=/path/to/demo_buffer \\
  runner.max_epochs=5 runner.weight_sync_interval=2

# 4-task online:
bash examples/embodiment/run_async.sh robotwin_lwd_openpi_pi05_4task \\
  runner.lwd_stage=online runner.max_epochs=5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


TASK_SUITE_4 = [
    "open_microwave",
    "hanging_mug",
    "place_mouse_pad",
    "blocks_ranking_size",
]


def cmd_env_split(args: argparse.Namespace) -> int:
    """Verify env_id % N assignment without launching SAPIEN."""
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    n = len(tasks)
    num_envs = args.num_envs
    if num_envs % n != 0:
        print(f"FAIL: num_envs={num_envs} not divisible by N={n}")
        return 1
    counts = defaultdict(int)
    for env_id in range(num_envs):
        counts[tasks[env_id % n]] += 1
    expected = num_envs // n
    print(f"task_names={tasks}")
    print(f"num_envs={num_envs}, expected_per_task={expected}")
    ok = True
    for t in tasks:
        c = counts[t]
        status = "OK" if c == expected else "FAIL"
        if c != expected:
            ok = False
        print(f"  {status}  {t}: {c}")
    # Mirror RoboTwinEnv assert for total_num_envs
    if args.total_num_envs is not None and args.total_num_envs % n != 0:
        print(f"FAIL: total_num_envs={args.total_num_envs} % N != 0")
        ok = False
    return 0 if ok else 1


def cmd_task_success(args: argparse.Namespace) -> int:
    """Aggregate per-task success from JSONL metrics or episode dumps.

    Accepted line formats (one JSON object per line):
      {"success_once/open_microwave": 1.0, "task_ids": 0, ...}
      {"episode": {"success_once": true, "task_ids": 2}, "task_names": [...]}
      {"env_info": {"success_once/place_mouse_pad": [0,1,0], ...}}
    """
    path = Path(args.log_json)
    if not path.exists():
        print(f"FAIL: log file not found: {path}")
        return 1

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    stats = {t: {"success": 0, "total": 0} for t in task_names}

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            _accumulate_row(row, task_names, stats)

    print("Per-task success:")
    any_data = False
    for t, s in stats.items():
        if s["total"] == 0:
            print(f"  {t}: no episodes")
            continue
        any_data = True
        rate = s["success"] / s["total"]
        print(f"  {t}: {s['success']}/{s['total']} = {rate:.3f}")
    if not any_data:
        print(
            "WARN: no per-task success keys found. Ensure env metrics include "
            "'success_once/<task>' or 'task_ids' in logged episode info."
        )
        return 2
    return 0


def _accumulate_row(row: dict, task_names: list[str], stats: dict) -> None:
    # Flat keys: success_once/<task>
    for t in task_names:
        key = f"success_once/{t}"
        if key in row:
            val = row[key]
            if isinstance(val, list):
                for v in val:
                    stats[t]["total"] += 1
                    stats[t]["success"] += int(bool(v))
            else:
                stats[t]["total"] += 1
                stats[t]["success"] += int(bool(val))

    episode = row.get("episode") or row.get("env_info") or {}
    if isinstance(episode, dict):
        for t in task_names:
            key = f"success_once/{t}"
            if key in episode:
                val = episode[key]
                if hasattr(val, "tolist"):
                    val = val.tolist()
                if isinstance(val, list):
                    for i, v in enumerate(val):
                        # If task_ids present, only count matching envs
                        tids = episode.get("task_ids", row.get("task_ids"))
                        if tids is not None:
                            if hasattr(tids, "tolist"):
                                tids = tids.tolist()
                            if isinstance(tids, list) and i < len(tids):
                                if task_names[int(tids[i])] != t:
                                    continue
                        stats[t]["total"] += 1
                        stats[t]["success"] += int(bool(v))
                else:
                    stats[t]["total"] += 1
                    stats[t]["success"] += int(bool(val))

        # Fallback: success_once + task_ids
        if "success_once" in episode and "task_ids" in episode:
            succ = episode["success_once"]
            tids = episode["task_ids"]
            if hasattr(succ, "tolist"):
                succ = succ.tolist()
            if hasattr(tids, "tolist"):
                tids = tids.tolist()
            if not isinstance(succ, list):
                succ = [succ]
                tids = [tids]
            for s, tid in zip(succ, tids):
                t = task_names[int(tid)] if int(tid) < len(task_names) else None
                if t is None:
                    continue
                stats[t]["total"] += 1
                stats[t]["success"] += int(bool(s))


def cmd_offline_online_checklist(args: argparse.Namespace) -> int:
    """Print a copy-paste checklist for offline→online smoke."""
    demo = args.demo_path or "/path/to/demo_buffer"
    print(
        f"""
=== LWD offline → online smoke checklist ===

[A] Unit tests
  pytest tests/unit_tests/test_divl.py tests/unit_tests/test_qam.py -q

[B] Env split (N=4, 32 envs → 8 each)
  python examples/embodiment/scripts/lwd_smoke_verify.py env-split \\
    --num-envs 32

[C] Offline stage (B_off only; set demo checkpoint)
  export ROBOT_PLATFORM=ALOHA
  export DEMO_PATH={demo}
  CUDA_VISIBLE_DEVICES=0 bash examples/embodiment/run_async.sh robotwin_lwd_openpi_pi05_1task \\
    runner.lwd_stage=offline \\
    algorithm.allow_demo_only=true \\
    algorithm.demo_ratio=1.0 \\
    algorithm.replay_buffer.min_buffer_size=0 \\
    algorithm.demo_buffer.load_path={demo} \\
    runner.max_epochs=2 algorithm.update_epoch=5

  Expect: lwd/value_loss, lwd/critic_loss, lwd/qam_loss finite; no intervene_* deps.
  Logs: RLinf/logs/<timestamp>-robotwin_lwd_openpi_pi05_1task/

[D] Online stage (mix demo_ratio=0.5, weight sync)
  CUDA_VISIBLE_DEVICES=0 bash examples/embodiment/run_async.sh robotwin_lwd_openpi_pi05_1task \\
    runner.lwd_stage=online \\
    algorithm.demo_ratio=0.5 \\
    algorithm.demo_buffer.load_path={demo} \\
    runner.max_epochs=5 runner.weight_sync_interval=2

  Expect: replay_buffer/num_trajectories grows; losses remain finite.

[E] 4-task per-task success
  bash examples/embodiment/run_async.sh robotwin_lwd_openpi_pi05_4task \\
    runner.max_epochs=3
  # Dump episode metrics to JSONL from your logger, then:
  python examples/embodiment/scripts/lwd_smoke_verify.py task-success \\
    --log-json /path/to/metrics.jsonl

  Expect: success_once/<task> keys for all four tasks; counts ~even across tasks
  when each task has the same number of envs.
"""
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_split = sub.add_parser("env-split", help="Check even task assignment")
    p_split.add_argument("--num-envs", type=int, default=32)
    p_split.add_argument("--total-num-envs", type=int, default=None)
    p_split.add_argument(
        "--tasks",
        type=str,
        default=",".join(TASK_SUITE_4),
    )
    p_split.set_defaults(func=cmd_env_split)

    p_succ = sub.add_parser("task-success", help="Aggregate per-task success from JSONL")
    p_succ.add_argument("--log-json", type=str, required=True)
    p_succ.add_argument("--tasks", type=str, default=",".join(TASK_SUITE_4))
    p_succ.set_defaults(func=cmd_task_success)

    p_chk = sub.add_parser(
        "checklist", help="Print offline→online manual smoke commands"
    )
    p_chk.add_argument("--demo-path", type=str, default=None)
    p_chk.set_defaults(func=cmd_offline_online_checklist)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
