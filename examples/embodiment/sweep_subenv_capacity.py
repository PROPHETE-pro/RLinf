#!/usr/bin/env python3
"""Sweep RoboTwin env-init capacity: 3 tasks x {2,4,8,16} SubEnvs per GPU on 4 GPUs.

Training commands are sent to tmux session `rlinf`.
Success if all SubEnvs finish setup_demo within 10 minutes.
Hard stop at 15 minutes. Stall (no log growth) of 8 minutes => hang.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path("/kpfs/data/ruitong_gan/RLinf")
LOG_ROOT = REPO / "logs"
RESULT_DIR = REPO / "logs" / "capacity_sweep"
TMUX = "rlinf"
SUCCESS_SEC = 10 * 60
HARD_CAP_SEC = 15 * 60
STALL_SEC = 8 * 60
POLL_SEC = 10

TASKS = [
    ("blocks_ranking_size", "robotwin_blocks_ranking_size_ppo_openpi_pi05_cotrain_fixseed_4gpu"),
    ("hanging_mug", "robotwin_hanging_mug_ppo_openpi_pi05_cotrain_fixseed_4gpu"),
    ("place_mouse_pad", "robotwin_place_mouse_pad_ppo_openpi_pi05_cotrain_fixseed_4gpu"),
]
PER_GPU_LIST = [2, 4, 8, 16]
WORLD_SIZE = 4


def run(cmd: list[str] | str, **kwargs) -> subprocess.CompletedProcess:
    if isinstance(cmd, str):
        return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kwargs)
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def gpu_stats() -> list[tuple[int, int, int]]:
    out = run(
        "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"
    ).stdout.strip()
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            rows.append((int(parts[0]), int(float(parts[1])), int(float(parts[2]))))
    return rows


def gpu_summary(rows: list[tuple[int, int, int]]) -> str:
    if not rows:
        return "n/a"
    return " ".join(f"g{i}={u}/{t}MiB" for i, u, t in rows)


def stop_job() -> None:
    run(["tmux", "send-keys", "-t", TMUX, "C-c"])
    time.sleep(1)
    run(["tmux", "send-keys", "-t", TMUX, "C-c"])
    time.sleep(1)
    run("pkill -f train_embodied_agent.py || true")
    run("pkill -f 'ray::EmbodiedFSDPActor|ray::EnvWorker|ray::MultiStepRolloutWorker' || true")
    time.sleep(2)
    run("pkill -9 -f train_embodied_agent.py || true")
    run("ray stop --force >/dev/null 2>&1 || true")
    time.sleep(2)


def wait_prompt(timeout: float = 20) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        alive = run("pgrep -f train_embodied_agent.py || true").stdout.strip()
        if not alive:
            return
        time.sleep(1)
    stop_job()


def launch(config: str, n_envs: int) -> None:
    cmd = (
        f"cd {REPO} && bash examples/embodiment/run_subenv_capacity.sh "
        f"{config} {n_envs}"
    )
    run(["tmux", "send-keys", "-t", TMUX, "C-c"])
    time.sleep(0.3)
    run(["tmux", "send-keys", "-t", TMUX, cmd])
    run(["tmux", "send-keys", "-t", TMUX, "Enter"])


def newest_log(config: str, after: float) -> Path | None:
    pattern = f"*-{config}"
    cands = []
    for p in LOG_ROOT.glob(pattern):
        log = p / "run_embodiment.log"
        if log.is_file() and log.stat().st_mtime >= after - 2:
            cands.append(log)
    if not cands:
        return None
    return max(cands, key=lambda x: x.stat().st_mtime)


def count_ok(text: str) -> int:
    return len(re.findall(r"setup_demo ok", text))


def detect_oom(text: str) -> bool:
    keys = (
        "CUDA out of memory",
        "OutOfMemoryError",
        "out of memory",
        "CUBLAS_STATUS_ALLOC_FAILED",
    )
    low = text.lower()
    return any(k.lower() in low for k in keys)


def last_env_line(text: str) -> str:
    lines = [
        ln.strip()
        for ln in text.splitlines()
        if "setup_demo" in ln or "SAPIEN" in ln or "SubEnv" in ln or "ready" in ln
    ]
    return lines[-1][-180:] if lines else ""


def probe(task: str, config: str, per_gpu: int) -> dict:
    n_envs = per_gpu * WORLD_SIZE
    stop_job()
    wait_prompt()
    launch_t = time.time()
    launch(config, n_envs)

    log_path = None
    for _ in range(30):
        log_path = newest_log(config, launch_t)
        if log_path is not None:
            break
        time.sleep(1)
    if log_path is None:
        return {
            "task": task,
            "per_gpu": per_gpu,
            "total_envs": n_envs,
            "status": "launch_fail",
            "init_sec": "",
            "setup_demo_ok": 0,
            "peak_mem_mib": "",
            "gpu_peak": "",
            "oom": False,
            "note": "no log created",
            "log": "",
        }

    t0 = time.time()
    last_size = -1
    stall_since = t0
    peak_rows = [(i, 0, 81920) for i in range(4)]
    init_sec = None
    status = "timeout_15m"
    note = ""
    ok = 0
    oom = False

    while True:
        now = time.time()
        elapsed = now - t0
        text = log_path.read_text(errors="ignore") if log_path.exists() else ""
        ok = count_ok(text)
        oom = detect_oom(text)
        size = log_path.stat().st_size if log_path.exists() else 0
        rows = gpu_stats()
        if rows:
            peak_rows = [
                (i, max(u, peak_rows[i][1] if i < len(peak_rows) else 0), t)
                for i, u, t in rows
            ]
        if size != last_size:
            last_size = size
            stall_since = now

        if ok >= n_envs and init_sec is None:
            init_sec = elapsed
            if elapsed <= SUCCESS_SEC:
                status = "ok"
                note = "all setup_demo ok"
            else:
                status = "too_slow"
                note = f"finished but {elapsed:.0f}s > 10min"
            break

        if oom:
            status = "oom"
            note = "CUDA OOM in log"
            break

        alive = bool(run("pgrep -f train_embodied_agent.py || true").stdout.strip())
        if not alive and ok < n_envs:
            status = "oom" if oom else "crashed"
            note = last_env_line(text) or "process exited before init done"
            break

        stalled = now - stall_since
        if ok > 0 and stalled >= STALL_SEC:
            status = "hang"
            note = f"log stalled {stalled:.0f}s; last={last_env_line(text)}"
            break
        if elapsed >= SUCCESS_SEC and ok < n_envs and stalled >= 60:
            status = "hang"
            note = f"not ready in 10min ({ok}/{n_envs}); last={last_env_line(text)}"
            break
        if elapsed >= HARD_CAP_SEC:
            status = "timeout_15m"
            note = f"hard cap; {ok}/{n_envs}; last={last_env_line(text)}"
            break
        time.sleep(POLL_SEC)

    stop_job()
    wait_prompt()
    peak_mem = max((u for _, u, _ in peak_rows), default=0)
    return {
        "task": task,
        "per_gpu": per_gpu,
        "total_envs": n_envs,
        "status": status,
        "init_sec": f"{init_sec:.1f}" if init_sec is not None else "",
        "setup_demo_ok": ok,
        "peak_mem_mib": peak_mem,
        "gpu_peak": gpu_summary(peak_rows),
        "oom": oom,
        "note": note[:240],
        "log": str(log_path),
    }


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = RESULT_DIR / f"sweep_{stamp}.csv"
    fields = [
        "task",
        "per_gpu",
        "total_envs",
        "status",
        "init_sec",
        "setup_demo_ok",
        "peak_mem_mib",
        "gpu_peak",
        "oom",
        "note",
        "log",
    ]
    print(f"RESULT_CSV={csv_path}", flush=True)
    with csv_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        f.flush()
        for per_gpu in PER_GPU_LIST:
            for task, config in TASKS:
                print(f"\n===== {task} per_gpu={per_gpu} total={per_gpu * WORLD_SIZE} =====", flush=True)
                row = probe(task, config, per_gpu)
                wr.writerow(row)
                f.flush()
                print(
                    f"RESULT {task} {per_gpu}/gpu status={row['status']} "
                    f"ok={row['setup_demo_ok']}/{row['total_envs']} "
                    f"init_sec={row['init_sec']} peak={row['peak_mem_mib']}MiB "
                    f"oom={row['oom']}",
                    flush=True,
                )
    print(f"DONE {csv_path}", flush=True)


if __name__ == "__main__":
    main()
