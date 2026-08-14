#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Generate RoboTwin train/eval success seeds for RLinf.
#
# Seed validity follows the official RoboTwin data collection / eval expert-check
# protocol:
#   1. setup_demo(seed=...) succeeds (scene is physically stable)
#   2. play_once() succeeds
#   3. plan_success and check_success() are both True
#
# Train seeds match RoboTwin/script/collect_data.py (need_plan=True, is_test=False).
# Eval seeds match RoboTwin/script/eval_policy.py expert_check (need_plan=True,
# is_test=True).
#
# Example:
#   export ROBOTWIN_PATH=/path/to/RoboTwin
#   export ROBOT_PLATFORM=ALOHA
#   export MUJOCO_GL=egl
#   export PYOPENGL_PLATFORM=egl
#   cd /path/to/RLinf
#   python toolkits/robotwin/generate_robotwin_seeds.py \
#       --tasks blocks_ranking_size hanging_mug open_microwave place_mouse_pad \
#       --task-config demo_clean \
#       --train-count 1000 \
#       --eval-count 260 \
#       --num-gpus 8 \
#       --workers-per-gpu 2 \
#       --merge-existing
#
# For demo_randomized, seeds are written to separate files:
#   rlinf/envs/robotwin/seeds/train_seeds_demo_randomized.json
#   rlinf/envs/robotwin/seeds/eval_seeds_demo_randomized.json

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

SeedMode = Literal["train", "eval"]


DEFAULT_TRAIN_START = 0
DEFAULT_EVAL_START = 100_100_000
DEFAULT_TRAIN_COUNT = 1000
DEFAULT_EVAL_COUNT = 260


@dataclass(frozen=True)
class SeedJob:
    task_name: str
    mode: SeedMode
    seed: int


@dataclass
class ActiveSeedCheck:
    slot_id: int
    gpu_id: int
    job: SeedJob
    process: mp.Process
    result_queue: mp.Queue
    started_at: float = field(default_factory=time.time)


def _force_kill_process(process: mp.Process, *, grace_sec: float = 5) -> None:
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=grace_sec)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


@dataclass
class TaskModeState:
    task_name: str
    mode: SeedMode
    target: int
    start_seed: int
    success_seeds: list[int]
    next_candidate: int
    done: bool = False
    attempts: int = 0
    failures: int = 0
    timeouts: int = 0
    last_success_seed: Optional[int] = None
    last_attempt_seed: Optional[int] = None
    last_error: Optional[str] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _seeds_dir() -> Path:
    return _repo_root() / "rlinf/envs/robotwin/seeds"


def _resolve_seed_output_paths(
    task_config: str,
    *,
    output_train: Optional[str],
    output_eval: Optional[str],
) -> tuple[Path, Path]:
    """Map task config to train/eval seed JSON paths.

    demo_clean keeps the legacy filenames for backward compatibility.
    Other configs (e.g. demo_randomized) write to separate JSON files so they
    do not overwrite the demo_clean seed sets.
    """
    seeds_dir = _seeds_dir()
    if task_config == "demo_clean":
        train_path = seeds_dir / "train_seeds.json"
        eval_path = seeds_dir / "eval_seeds.json"
    else:
        train_path = seeds_dir / f"train_seeds_{task_config}.json"
        eval_path = seeds_dir / f"eval_seeds_{task_config}.json"

    if output_train is not None:
        train_path = Path(output_train)
    if output_eval is not None:
        eval_path = Path(output_eval)
    return train_path.resolve(), eval_path.resolve()


def _detect_num_gpus(explicit: Optional[int]) -> int:
    if explicit is not None:
        return max(1, explicit)
    try:
        import torch

        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
    except Exception:
        pass
    return 1


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def _merge_task_entry(
    existing: dict[str, Any],
    task_name: str,
    success_seeds: list[int],
) -> dict[str, Any]:
    merged = dict(existing)
    merged[task_name] = {
        "task_name": task_name,
        "success_seeds": sorted(success_seeds),
    }
    return merged


def _resolve_assets_path(robotwin_path: Path, assets_path: Optional[str]) -> str:
    """Resolve RoboTwin ASSETS_PATH (repo root, not the assets/ subdirectory).

    RoboTwin code joins ``ASSETS_PATH / "assets/objects/..."``. RLinf eval YAMLs
    therefore point ``assets_path`` at the RoboTwin repo root.
    """
    robotwin_root = robotwin_path.resolve()
    if assets_path is None:
        return str(robotwin_root)

    candidate = Path(assets_path).resolve()
    if (candidate / "assets" / "objects").is_dir():
        return str(candidate)
    if candidate.name == "assets" and (candidate / "objects").is_dir():
        return str(candidate.parent)
    return str(candidate)


def _validate_assets_layout(assets_path: str) -> None:
    list_json = Path(assets_path) / "assets/objects/objaverse/list.json"
    if not list_json.exists():
        raise FileNotFoundError(
            "RoboTwin assets not found at expected path: "
            f"{list_json}. Set --assets-path to the RoboTwin repo root "
            f"(not the assets/ subdirectory)."
        )


def _configure_robotwin_imports(robotwin_path: Path, assets_path: Optional[str]) -> None:
    robotwin_path = robotwin_path.resolve()
    if not robotwin_path.exists():
        raise FileNotFoundError(f"RoboTwin path does not exist: {robotwin_path}")

    os.chdir(robotwin_path)
    if str(robotwin_path) not in sys.path:
        sys.path.insert(0, str(robotwin_path))

    if assets_path is not None:
        assets_path = str(Path(assets_path).resolve())
        os.environ["ASSETS_PATH"] = assets_path


def _build_task_args(task_name: str, task_config_name: str) -> dict[str, Any]:
    import yaml
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    config_path = Path(CONFIGS_PATH) / f"{task_config_name}.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config_name
    args["render_freq"] = 0
    args["save_data"] = False
    args["collect_data"] = False
    args["eval_video_log"] = False

    embodiment_type = args.get("embodiment")
    embodiment_config_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"
    with embodiment_config_path.open("r", encoding="utf-8") as handle:
        embodiment_types = yaml.load(handle.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment: str) -> str:
        robot_file = embodiment_types[embodiment]["file_path"]
        if robot_file is None:
            raise ValueError(f"Missing embodiment file for {embodiment}")
        return robot_file

    def get_embodiment_config(robot_file: str) -> dict[str, Any]:
        robot_config_file = Path(robot_file) / "config.yml"
        with robot_config_file.open("r", encoding="utf-8") as handle:
            return yaml.load(handle.read(), Loader=yaml.FullLoader)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        embodiment_name = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        embodiment_name = f"{embodiment_type[0]}+{embodiment_type[1]}"
    else:
        raise ValueError("embodiment must contain 1 or 3 entries")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = embodiment_name
    args["save_path"] = os.path.join(args.get("save_path", "./data"), task_name, task_config_name)
    args.setdefault("planner_backend", "mplib")
    return args


def _check_seed(
    task_name: str,
    args: dict[str, Any],
    seed: int,
    *,
    is_eval: bool,
) -> tuple[bool, Optional[str]]:
    import importlib

    from envs.utils.create_actor import UnStableError

    module = importlib.import_module(f"envs.{task_name}")
    task_cls = getattr(module, task_name)
    task_env = task_cls()

    setup_kwargs = dict(args)
    setup_kwargs["need_plan"] = True

    try:
        if is_eval:
            task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **setup_kwargs)
        else:
            task_env.setup_demo(now_ep_num=0, seed=seed, **setup_kwargs)
        task_env.play_once()
        success = bool(task_env.plan_success and task_env.check_success())
        task_env.close_env()
        if task_env.render_freq:
            task_env.viewer.close()
        return success, None
    except UnStableError:
        try:
            task_env.close_env()
        except Exception:
            pass
        return False, "unstable"
    except Exception as exc:
        try:
            task_env.close_env()
        except Exception:
            pass
        return False, f"{type(exc).__name__}: {exc}"


def _check_seed_isolated(
    result_queue: mp.Queue,
    gpu_id: int,
    robotwin_path: str,
    assets_path: Optional[str],
    task_name: str,
    task_config_name: str,
    planner_backend: str,
    seed: int,
    is_eval: bool,
) -> None:
    """Run a single seed check in a child process (can be killed on timeout)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("ROBOT_PLATFORM", "ALOHA")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="warp")

    try:
        _configure_robotwin_imports(Path(robotwin_path), assets_path)
        args = _build_task_args(task_name, task_config_name)
        args["planner_backend"] = planner_backend
        success, error = _check_seed(task_name, args, seed, is_eval=is_eval)
        result_queue.put((success, error))
    except Exception as exc:
        result_queue.put((False, f"{type(exc).__name__}: {exc}"))


def _launch_seed_check(
    ctx: mp.context.BaseContext,
    *,
    slot_id: int,
    gpu_id: int,
    robotwin_path: str,
    assets_path: Optional[str],
    task_config_name: str,
    planner_backend: str,
    job: SeedJob,
) -> ActiveSeedCheck:
    result_queue: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=_check_seed_isolated,
        args=(
            result_queue,
            gpu_id,
            robotwin_path,
            assets_path,
            job.task_name,
            task_config_name,
            planner_backend,
            job.seed,
            job.mode == "eval",
        ),
        daemon=False,
    )
    process.start()
    return ActiveSeedCheck(
        slot_id=slot_id,
        gpu_id=gpu_id,
        job=job,
        process=process,
        result_queue=result_queue,
    )


def _finish_seed_check(check: ActiveSeedCheck) -> tuple[bool, Optional[str]]:
    try:
        success, error = check.result_queue.get(timeout=1)
        return success, error
    except queue.Empty:
        exit_code = check.process.exitcode
        if exit_code not in (0, None):
            return False, f"crashed(exit={exit_code})"
        return False, "no_result"


def _apply_seed_result(
    states: dict[tuple[str, SeedMode], TaskModeState],
    job: SeedJob,
    *,
    success: bool,
    error: Optional[str],
    state_lock: threading.Lock,
) -> None:
    key = (job.task_name, job.mode)
    with state_lock:
        state = states[key]
        state.last_attempt_seed = job.seed
        state.last_error = error
        if success:
            if len(state.success_seeds) < state.target:
                state.success_seeds.append(job.seed)
                state.last_success_seed = job.seed
                if len(state.success_seeds) >= state.target:
                    state.done = True
        else:
            state.failures += 1
            if error and error.startswith("timeout"):
                state.timeouts += 1
                print(
                    f"  [timeout] {state.task_name}/{state.mode} "
                    f"seed={job.seed} ({error})",
                    flush=True,
                )
            elif error and error.startswith("crashed"):
                print(
                    f"  [crashed] {state.task_name}/{state.mode} "
                    f"seed={job.seed} ({error})",
                    flush=True,
                )


def _reap_seed_checks(
    active_checks: list[ActiveSeedCheck],
    *,
    states: dict[tuple[str, SeedMode], TaskModeState],
    state_lock: threading.Lock,
    seed_timeout: float,
    active_jobs: dict[int, dict[str, Any]],
    force: bool = False,
) -> None:
    now = time.time()
    grace_sec = max(15.0, seed_timeout * 0.1) if seed_timeout > 0 else 0.0
    for check in list(active_checks):
        elapsed = now - check.started_at
        timed_out = force or (
            seed_timeout > 0 and elapsed > seed_timeout + grace_sec
        )
        if check.process.is_alive() and not timed_out:
            continue

        if check.process.is_alive():
            print(
                f"  [watchdog] force-killing seed={check.job.seed} "
                f"({check.job.task_name}/{check.job.mode}) elapsed={elapsed:.0f}s",
                flush=True,
            )
            _force_kill_process(check.process)
            success, error = False, f"timeout>{seed_timeout}s(watchdog)"
        else:
            success, error = _finish_seed_check(check)
        _apply_seed_result(
            states,
            check.job,
            success=success,
            error=error,
            state_lock=state_lock,
        )
        active_checks.remove(check)
        active_jobs.pop(check.slot_id, None)


def _init_task_states(
    tasks: list[str],
    *,
    train_count: int,
    eval_count: int,
    train_start: int,
    eval_start: int,
    train_output: Path,
    eval_output: Path,
    merge_existing: bool,
    skip_existing: bool,
    min_next_train_seed: Optional[int],
    min_next_eval_seed: Optional[int],
) -> dict[tuple[str, SeedMode], TaskModeState]:
    train_data = _load_json(train_output) if merge_existing or skip_existing else {}
    eval_data = _load_json(eval_output) if merge_existing or skip_existing else {}
    states: dict[tuple[str, SeedMode], TaskModeState] = {}

    for task_name in tasks:
        existing_train = train_data.get(task_name, {}).get("success_seeds", [])
        if skip_existing and len(existing_train) >= train_count:
            train_seeds = list(existing_train[:train_count])
            train_done = True
            train_next = max(existing_train) + 1 if existing_train else train_start
        else:
            train_seeds = list(existing_train)
            train_done = len(train_seeds) >= train_count
            train_next = max(train_seeds) + 1 if train_seeds else train_start

        if min_next_train_seed is not None and not train_done:
            train_next = max(train_next, min_next_train_seed)

        states[(task_name, "train")] = TaskModeState(
            task_name=task_name,
            mode="train",
            target=train_count,
            start_seed=train_start,
            success_seeds=train_seeds,
            next_candidate=train_next,
            done=train_done,
        )

        existing_eval = eval_data.get(task_name, {}).get("success_seeds", [])
        if skip_existing and len(existing_eval) >= eval_count:
            eval_seeds = list(existing_eval[:eval_count])
            eval_done = True
            eval_next = max(existing_eval) + 1 if existing_eval else eval_start
        else:
            eval_seeds = list(existing_eval)
            eval_done = len(eval_seeds) >= eval_count
            eval_next = max(eval_seeds) + 1 if eval_seeds else eval_start

        if min_next_eval_seed is not None and not eval_done:
            eval_next = max(eval_next, min_next_eval_seed)

        states[(task_name, "eval")] = TaskModeState(
            task_name=task_name,
            mode="eval",
            target=eval_count,
            start_seed=eval_start,
            success_seeds=eval_seeds,
            next_candidate=eval_next,
            done=eval_done,
        )

    return states


def _save_outputs(
    states: dict[tuple[str, SeedMode], TaskModeState],
    *,
    train_output: Path,
    eval_output: Path,
    merge_existing: bool,
) -> None:
    train_payload = _load_json(train_output) if merge_existing else {}
    eval_payload = _load_json(eval_output) if merge_existing else {}

    for state in states.values():
        if state.mode == "train":
            train_payload = _merge_task_entry(train_payload, state.task_name, state.success_seeds)
        else:
            eval_payload = _merge_task_entry(eval_payload, state.task_name, state.success_seeds)

    _write_json(train_output, train_payload)
    _write_json(eval_output, eval_payload)


def _dispatch_jobs(
    states: dict[tuple[str, SeedMode], TaskModeState],
    job_queue: mp.Queue,
    *,
    state_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        with state_lock:
            if all(state.done for state in states.values()):
                break

        dispatched = False
        with state_lock:
            for state in states.values():
                if state.done:
                    continue
                if len(state.success_seeds) >= state.target:
                    state.done = True
                    continue

                job = SeedJob(
                    task_name=state.task_name,
                    mode=state.mode,
                    seed=state.next_candidate,
                )
                state.next_candidate += 1
                state.attempts += 1
                try:
                    job_queue.put(
                        {
                            "task_name": job.task_name,
                            "mode": job.mode,
                            "seed": job.seed,
                        },
                        timeout=0.05,
                    )
                    dispatched = True
                except queue.Full:
                    break

        if not dispatched:
            time.sleep(0.05)


def _print_progress(
    states: dict[tuple[str, SeedMode], TaskModeState],
    *,
    active_jobs: dict[int, dict[str, Any]],
    total_workers: int,
) -> None:
    for state in sorted(states.values(), key=lambda item: (item.task_name, item.mode)):
        status = "done" if state.done else "running"
        last_success = (
            str(state.last_success_seed)
            if state.last_success_seed is not None
            else "-"
        )
        max_success = max(state.success_seeds) if state.success_seeds else None
        max_success_text = str(max_success) if max_success is not None else "-"
        last_attempt = (
            str(state.last_attempt_seed)
            if state.last_attempt_seed is not None
            else "-"
        )
        last_error = state.last_error or "-"
        print(
            f"[{status}] {state.task_name}/{state.mode}: "
            f"{len(state.success_seeds)}/{state.target} success, "
            f"attempts={state.attempts}, dispatched_next={state.next_candidate}, "
            f"last_success_seed={last_success}, max_success_seed={max_success_text}, "
            f"last_attempt_seed={last_attempt}, timeouts={state.timeouts}, "
            f"last_error={last_error}",
            flush=True,
        )

    if active_jobs:
        now = time.time()
        for slot_id in sorted(active_jobs):
            info = active_jobs[slot_id]
            elapsed = now - info["started_at"]
            print(
                f"  [slot {slot_id}] checking seed={info['seed']} "
                f"({info['task_name']}/{info['mode']}, gpu={info['gpu_id']}) "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
    else:
        print(
            f"  [pool] 0/{total_workers} seed checks running",
            flush=True,
        )


def run_seed_generation(args: argparse.Namespace) -> None:
    robotwin_path = Path(args.robotwin_path).resolve()
    train_output = Path(args.output_train).resolve()
    eval_output = Path(args.output_eval).resolve()

    num_gpus = _detect_num_gpus(args.num_gpus)
    workers_per_gpu = max(1, args.workers_per_gpu)
    total_workers = num_gpus * workers_per_gpu
    if args.queue_size is None:
        queue_size = max(1, total_workers)
    else:
        queue_size = max(1, args.queue_size)

    states = _init_task_states(
        args.tasks,
        train_count=args.train_count,
        eval_count=args.eval_count,
        train_start=args.train_start,
        eval_start=args.eval_start,
        train_output=train_output,
        eval_output=eval_output,
        merge_existing=args.merge_existing,
        skip_existing=args.skip_existing,
        min_next_train_seed=args.min_next_train_seed,
        min_next_eval_seed=args.min_next_eval_seed,
    )

    pending = [state for state in states.values() if not state.done]
    if not pending:
        print("All requested task/mode seed sets are already complete. Nothing to do.")
        _save_outputs(
            states,
            train_output=train_output,
            eval_output=eval_output,
            merge_existing=args.merge_existing,
        )
        return

    print(
        f"RoboTwin seed generation: tasks={args.tasks}, config={args.task_config}, "
        f"gpus={num_gpus}, workers_per_gpu={workers_per_gpu}, total_workers={total_workers}, "
        f"queue_size={queue_size}, seed_timeout={args.seed_timeout}s, "
        f"planner={args.planner_backend}",
        flush=True,
    )
    print(f"Train output: {train_output}", flush=True)
    print(f"Eval output: {eval_output}", flush=True)
    _print_progress(
        states,
        active_jobs={},
        total_workers=total_workers,
    )

    ctx = mp.get_context("spawn")
    job_queue: mp.Queue = ctx.Queue(maxsize=queue_size)
    state_lock = threading.Lock()
    active_checks: list[ActiveSeedCheck] = []
    active_jobs: dict[int, dict[str, Any]] = {}
    next_slot_id = 0

    dispatcher_stop = threading.Event()
    dispatcher = threading.Thread(
        target=_dispatch_jobs,
        kwargs={
            "states": states,
            "job_queue": job_queue,
            "state_lock": state_lock,
            "stop_event": dispatcher_stop,
        },
        daemon=True,
    )
    dispatcher.start()

    last_save = time.time()
    last_report = time.time()

    def _try_launch_checks() -> None:
        nonlocal next_slot_id
        while len(active_checks) < total_workers:
            if all(state.done for state in states.values()):
                break
            try:
                payload = job_queue.get_nowait()
            except queue.Empty:
                break
            job = SeedJob(**payload)
            slot_id = next_slot_id
            next_slot_id += 1
            gpu_id = len(active_checks) % num_gpus
            check = _launch_seed_check(
                ctx,
                slot_id=slot_id,
                gpu_id=gpu_id,
                robotwin_path=str(robotwin_path),
                assets_path=args.assets_path,
                task_config_name=args.task_config,
                planner_backend=args.planner_backend,
                job=job,
            )
            active_checks.append(check)
            active_jobs[slot_id] = {
                "task_name": job.task_name,
                "mode": job.mode,
                "seed": job.seed,
                "gpu_id": gpu_id,
                "started_at": check.started_at,
            }

    try:
        while True:
            if all(state.done for state in states.values()) and not active_checks:
                break

            _reap_seed_checks(
                active_checks,
                states=states,
                state_lock=state_lock,
                seed_timeout=args.seed_timeout,
                active_jobs=active_jobs,
            )
            _try_launch_checks()

            now = time.time()
            if now - last_report >= args.report_interval:
                _print_progress(
                    states,
                    active_jobs=active_jobs,
                    total_workers=total_workers,
                )
                last_report = now

            if now - last_save >= args.save_interval:
                _save_outputs(
                    states,
                    train_output=train_output,
                    eval_output=eval_output,
                    merge_existing=args.merge_existing,
                )
                last_save = now

            if args.max_attempts_per_task_mode is not None:
                exceeded = any(
                    (not state.done)
                    and state.attempts >= args.max_attempts_per_task_mode
                    for state in states.values()
                )
                if exceeded:
                    print("Reached --max-attempts-per-task-mode limit; stopping.", flush=True)
                    break

            if not all(state.done for state in states.values()) or active_checks:
                time.sleep(0.2)
    finally:
        dispatcher_stop.set()
        _reap_seed_checks(
            active_checks,
            states=states,
            state_lock=state_lock,
            seed_timeout=args.seed_timeout,
            active_jobs=active_jobs,
            force=True,
        )

        _save_outputs(
            states,
            train_output=train_output,
            eval_output=eval_output,
            merge_existing=args.merge_existing,
        )

    print("\nFinal status:", flush=True)
    _print_progress(
        states,
        active_jobs=active_jobs,
        total_workers=total_workers,
    )
    print(f"Wrote train seeds to {train_output}", flush=True)
    print(f"Wrote eval seeds to {eval_output}", flush=True)

    incomplete = [state for state in states.values() if not state.done]
    if incomplete:
        details = ", ".join(
            f"{state.task_name}/{state.mode} ({len(state.success_seeds)}/{state.target})"
            for state in incomplete
        )
        raise RuntimeError(f"Incomplete seed generation for: {details}")


def _build_arg_parser() -> argparse.ArgumentParser:
    default_robotwin = os.environ.get("ROBOTWIN_PATH", "")

    parser = argparse.ArgumentParser(
        description="Generate RoboTwin success seeds for RLinf train/eval JSON files.",
    )
    parser.add_argument(
        "--robotwin-path",
        default=default_robotwin,
        help="Path to RoboTwin repo root (RLinf_support branch recommended).",
    )
    parser.add_argument(
        "--assets-path",
        default=None,
        help=(
            "RoboTwin ASSETS_PATH root. Defaults to <robotwin-path>. "
            "Must be the repo root (where assets/ lives), not assets/ itself."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="RoboTwin task names to process.",
    )
    parser.add_argument(
        "--task-config",
        default="demo_clean",
        help=(
            "Task config yaml stem under RoboTwin/task_config/. "
            "demo_clean writes train_seeds.json/eval_seeds.json; other configs "
            "write train_seeds_<config>.json/eval_seeds_<config>.json."
        ),
    )
    parser.add_argument(
        "--planner-backend",
        default="mplib",
        choices=["mplib", "curobo"],
        help=(
            "Motion planner backend used during expert play_once checks. "
            "mplib is recommended for seed generation; curobo can hang or crash "
            "when repeatedly spawned."
        ),
    )
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--eval-count", type=int, default=DEFAULT_EVAL_COUNT)
    parser.add_argument("--train-start", type=int, default=DEFAULT_TRAIN_START)
    parser.add_argument("--eval-start", type=int, default=DEFAULT_EVAL_START)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument(
        "--queue-size",
        type=int,
        default=None,
        help=(
            "Pending seed-check job queue size. Defaults to total worker count. "
            "Use 1 for the most intuitive progress logs."
        ),
    )
    parser.add_argument(
        "--seed-timeout",
        type=float,
        default=180.0,
        help=(
            "Seconds before force-killing a single seed check subprocess. "
            "Set 0 to disable watchdog (not recommended)."
        ),
    )
    parser.add_argument(
        "--min-next-train-seed",
        type=int,
        default=None,
        help="Optional lower bound for the next train candidate seed.",
    )
    parser.add_argument(
        "--min-next-eval-seed",
        type=int,
        default=None,
        help="Optional lower bound for the next eval candidate seed.",
    )
    parser.add_argument(
        "--max-attempts-per-task-mode",
        type=int,
        default=None,
        help="Optional safety cap on candidate seeds tested per task/mode.",
    )
    parser.add_argument(
        "--output-train",
        default=None,
        help="Override train seed JSON path (default depends on --task-config).",
    )
    parser.add_argument(
        "--output-eval",
        default=None,
        help="Override eval seed JSON path (default depends on --task-config).",
    )
    parser.add_argument(
        "--merge-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep other tasks in output JSON and merge current task results.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip task/mode generation when enough seeds already exist in output JSON.",
    )
    parser.add_argument("--save-interval", type=float, default=30.0)
    parser.add_argument("--report-interval", type=float, default=5.0)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.robotwin_path:
        parser.error("--robotwin-path is required (or set ROBOTWIN_PATH).")

    if args.assets_path is None:
        args.assets_path = _resolve_assets_path(Path(args.robotwin_path), None)
    else:
        args.assets_path = _resolve_assets_path(
            Path(args.robotwin_path),
            args.assets_path,
        )
    _validate_assets_layout(args.assets_path)
    print(f"ASSETS_PATH: {args.assets_path}", flush=True)

    train_output, eval_output = _resolve_seed_output_paths(
        args.task_config,
        output_train=args.output_train,
        output_eval=args.output_eval,
    )
    args.output_train = str(train_output)
    args.output_eval = str(eval_output)

    mp.set_start_method("spawn", force=True)
    run_seed_generation(args)


if __name__ == "__main__":
    main()
