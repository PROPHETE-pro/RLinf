# Copyright 2025 The RLinf Authors.
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

"""Subprocess vectorized environment for RoboTwin.

Each SubEnv runs in an isolated child process so native SAPIEN/mplib hangs can
be killed without SIGKILL-ing the entire EnvWorker Ray actor.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections import defaultdict
from multiprocessing import connection
from typing import Any, Literal, Optional

import gymnasium as gym
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

OnTimeoutPolicy = Literal["truncate", "fail"]
_KILL_GRACE_SEC = 5.0
_SPAWN_READY_TIMEOUT_SEC = 600.0


def _make_recovery_info(
    env_id: int,
    respawn_count: int,
    *,
    recovery_failed: bool = False,
    recovery_error: Optional[str] = None,
) -> dict[str, Any]:
    """Info dict aligned with RoboTwin SubEnv step info plus recovery metadata."""
    info: dict[str, Any] = {
        "success": False,
        "subenv_timeout": True,
        "sub_env_id": env_id,
        "respawn_count": respawn_count,
    }
    if recovery_failed:
        info["subenv_recovery_failed"] = True
        if recovery_error is not None:
            info["recovery_error"] = recovery_error
    return info


def _minimal_obs(instruction: str = "") -> dict[str, Any]:
    """Placeholder observation when a SubEnv cannot serve get_obs after recovery."""
    return {
        "full_image": np.zeros((240, 320, 3), dtype=np.uint8),
        "left_wrist_image": None,
        "right_wrist_image": None,
        "state": np.zeros(14, dtype=np.float32),
        "instruction": instruction,
    }


def _to_subenv_scalar_array(value: Any, *, dtype: np.dtype) -> np.ndarray:
    """Match RoboTwin SubEnv step fields: length-1 numpy vectors."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return np.array([value.item()], dtype=dtype)
        flat = value.reshape(-1)
        if flat.size == 1:
            return np.array([flat[0]], dtype=dtype)
        return flat.astype(dtype, copy=False)
    if isinstance(value, (bool, np.bool_)):
        return np.array([int(value)], dtype=dtype)
    return np.array([value], dtype=dtype)


def normalize_subenv_step_result(result: dict[str, Any]) -> dict[str, Any]:
    """Align recovery/truncate step dict with live SubEnv ``step()`` conventions."""
    normalized = dict(result)
    normalized["reward"] = _to_subenv_scalar_array(
        result.get("reward", 0), dtype=np.float32
    )
    normalized["terminated"] = _to_subenv_scalar_array(
        result.get("terminated", 0), dtype=np.int32
    )
    normalized["truncated"] = _to_subenv_scalar_array(
        result.get("truncated", 0), dtype=np.int32
    )
    return normalized


def build_robotwin_task_args(
    task_config: dict,
    n_envs: int,
    assets_path: Optional[str] = None,
) -> dict:
    """Build RoboTwin SubEnv args dict (mirrors VectorEnv.__init__)."""
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    assets_path = assets_path or os.getenv("ASSETS_PATH")
    if assets_path is None:
        raise ValueError("ASSETS_PATH must be set for RoboTwin SubprocVectorEnv")

    head_camera_type = "D435"
    rdt_step = 10
    args = dict(task_config)

    args["planner_backend"] = args.get("planner_backend", "curobo")
    args["clear_cache_freq"] = max(1, int(args.get("clear_cache_freq", 8)))

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    def get_embodiment_file(emb_type: str) -> str:
        robot_file = _embodiment_types[emb_type]["file_path"]
        if robot_file is None:
            raise ValueError("No embodiment files")
        return robot_file

    def get_embodiment_config(robot_file: str) -> dict:
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    if len(embodiment_type) == 1:
        args["left_robot_file"] = os.path.join(
            assets_path, get_embodiment_file(embodiment_type[0])
        )
        args["right_robot_file"] = os.path.join(
            assets_path, get_embodiment_file(embodiment_type[0])
        )
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = os.path.join(
            assets_path, get_embodiment_file(embodiment_type[0])
        )
        args["right_robot_file"] = os.path.join(
            assets_path, get_embodiment_file(embodiment_type[1])
        )
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "_" + str(embodiment_type[1])

    args["embodiment_name"] = embodiment_name
    args["rdt_step"] = rdt_step
    args["save_path"] += f"/{args['task_name']}_reward"
    args["n_envs"] = n_envs
    args["action_dim"] = 14
    args["eval_mode"] = True
    args["eval_video_log"] = False
    args["render_freq"] = 0
    return args


def _subproc_worker(
    conn: connection.Connection,
    env_id: int,
    task_name: str,
    args: dict,
    env_seed: Optional[int],
    instruction_type: str,
) -> None:
    """Child process entry: one RoboTwin SubEnv + Pipe command loop."""
    from robotwin.envs.vector_env import SubEnv

    global_lock = threading.Lock()
    sub_env = SubEnv(
        env_id=env_id,
        task_name=task_name,
        args=args,
        env_seed=env_seed,
        instruction_type=instruction_type,
        global_lock=global_lock,
    )
    sub_env.setup_task()
    conn.send({"status": "ready"})

    try:
        while True:
            try:
                cmd, data = conn.recv()
            except EOFError:
                conn.close()
                break

            if cmd == "step":
                conn.send(sub_env.step(data))
            elif cmd == "reset":
                sub_env.reset(env_seed=data)
                conn.send(None)
            elif cmd == "get_obs":
                conn.send(sub_env.get_obs())
            elif cmd == "close":
                sub_env.close(clear_cache=data)
                conn.send(None)
                conn.close()
                break
            elif cmd == "check_seed":
                conn.send(sub_env.check_seed(data))
            else:
                conn.close()
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        conn.close()


class SubprocSubEnvWorker:
    """Manage a single RoboTwin SubEnv subprocess."""

    def __init__(
        self,
        env_id: int,
        task_name: str,
        args: dict,
        env_seed: Optional[int],
        instruction_type: str = "seen",
        step_timeout_sec: float = 60.0,
        max_respawns: int = 10,
        on_timeout: OnTimeoutPolicy = "truncate",
        mp_context=None,
    ) -> None:
        self.env_id = env_id
        self.task_name = task_name
        self.args = args
        self.env_seed = env_seed if env_seed is not None else env_id
        self.instruction_type = instruction_type
        self.step_timeout_sec = step_timeout_sec
        self.max_respawns = max_respawns
        self.on_timeout = on_timeout
        self.respawn_count = 0
        self._mp_context = mp_context
        self.process = None
        self.parent_conn: Optional[connection.Connection] = None
        self._last_obs: Optional[dict[str, Any]] = None
        self._spawn_process(self.env_seed)

    def _spawn_process(self, env_seed: int) -> None:
        ctx = self._mp_context
        parent_conn, child_conn = ctx.Pipe()
        process = ctx.Process(
            target=_subproc_worker,
            args=(
                child_conn,
                self.env_id,
                self.task_name,
                self.args,
                env_seed,
                self.instruction_type,
            ),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self.process = process
        self.parent_conn = parent_conn
        self.env_seed = env_seed
        self._wait_ready()

    def _wait_ready(self, timeout: float = _SPAWN_READY_TIMEOUT_SEC) -> None:
        if self.parent_conn is None:
            raise RuntimeError(f"SubEnv {self.env_id}: pipe not initialized")
        if not self.parent_conn.poll(timeout):
            self._kill_process()
            raise RuntimeError(
                f"SubEnv {self.env_id} failed to start within {timeout:.1f}s"
            )
        msg = self.parent_conn.recv()
        if not isinstance(msg, dict) or msg.get("status") != "ready":
            raise RuntimeError(
                f"SubEnv {self.env_id} unexpected ready message: {msg!r}"
            )

    def _kill_process(self) -> None:
        if self.process is None:
            return
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=_KILL_GRACE_SEC)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=_KILL_GRACE_SEC)
        if self.parent_conn is not None:
            try:
                self.parent_conn.close()
            except OSError:
                pass
        self.process = None
        self.parent_conn = None

    def _send_recv(
        self,
        cmd: str,
        data: Any,
        timeout: Optional[float] = None,
    ) -> Any:
        if self.parent_conn is None or self.process is None:
            raise RuntimeError(f"SubEnv {self.env_id} process is not running")
        if not self.process.is_alive():
            raise RuntimeError(f"SubEnv {self.env_id} process died unexpectedly")

        self.parent_conn.send((cmd, data))
        effective_timeout = (
            self.step_timeout_sec if timeout is None else timeout
        )
        if not self.parent_conn.poll(effective_timeout):
            raise TimeoutError(
                f"SubEnv {self.env_id} {cmd} timed out after {effective_timeout:.1f}s"
            )
        return self.parent_conn.recv()

    def _force_kill_and_respawn(self, elapsed: float, reason: str) -> None:
        old_seed = self.env_seed
        self._kill_process()
        self.respawn_count += 1
        if self.respawn_count > self.max_respawns:
            raise RuntimeError(
                f"SubEnv {self.env_id} exceeded subproc_max_respawns="
                f"{self.max_respawns} after {reason}"
            )
        new_seed = old_seed + 1
        logger.warning(
            "[SUBENV_RECOVER] sub_env=%s elapsed=%.1fs action=kill+respawn "
            "reason=%s seed=%s->%s respawn_count=%s",
            self.env_id,
            elapsed,
            reason,
            old_seed,
            new_seed,
            self.respawn_count,
        )
        self._spawn_process(new_seed)
        try:
            self._send_recv("reset", new_seed, timeout=self.step_timeout_sec)
        except TimeoutError as exc:
            raise RuntimeError(
                f"SubEnv {self.env_id} reset timed out after respawn"
            ) from exc

    def _remember_obs(self, obs: Optional[dict[str, Any]]) -> None:
        if isinstance(obs, dict):
            self._last_obs = obs

    def _obs_for_recovery(self) -> dict[str, Any]:
        if self._last_obs is not None:
            return self._last_obs
        try:
            obs = self._send_recv("get_obs", None, timeout=self.step_timeout_sec)
            self._remember_obs(obs)
            return obs
        except Exception as exc:
            logger.warning(
                "[SUBENV_RECOVER] sub_env=%s get_obs failed after respawn: %s",
                self.env_id,
                exc,
            )
            return _minimal_obs()

    def _truncate_step_result(
        self,
        *,
        recovery_failed: bool = False,
        recovery_error: Optional[str] = None,
    ) -> dict:
        obs = (
            self._last_obs
            if recovery_failed and self._last_obs is not None
            else self._obs_for_recovery()
        )
        return normalize_subenv_step_result(
            {
                "obs": obs,
                "reward": np.array([0], dtype=np.float32),
                "terminated": np.array([0], dtype=np.int32),
                "truncated": np.array([1], dtype=np.int32),
                "info": _make_recovery_info(
                    self.env_id,
                    self.respawn_count,
                    recovery_failed=recovery_failed,
                    recovery_error=recovery_error,
                ),
            }
        )

    def _recovery_failed_step_result(self, reason: str) -> dict:
        logger.error(
            "[SUBENV_RECOVER] sub_env=%s recovery failed: %s",
            self.env_id,
            reason,
        )
        return self._truncate_step_result(
            recovery_failed=True,
            recovery_error=reason,
        )

    def step_with_timeout(self, actions: np.ndarray) -> dict:
        start = time.time()
        self.parent_conn.send(("step", actions))
        if not self.parent_conn.poll(self.step_timeout_sec):
            elapsed = time.time() - start
            if self.on_timeout == "fail":
                self._kill_process()
                raise RuntimeError(
                    f"SubEnv {self.env_id} step timed out after {elapsed:.1f}s"
                )
            try:
                self._force_kill_and_respawn(elapsed, reason="step_timeout")
            except Exception as exc:
                return self._recovery_failed_step_result(str(exc))
            return self._truncate_step_result()
        result = self.parent_conn.recv()
        self._remember_obs(result.get("obs"))
        return normalize_subenv_step_result(result)

    def reset(self, env_seed: Optional[int] = None) -> None:
        if env_seed is not None:
            self.env_seed = env_seed
        start = time.time()
        try:
            self._send_recv("reset", env_seed, timeout=self.step_timeout_sec)
        except TimeoutError:
            elapsed = time.time() - start
            if self.on_timeout == "fail":
                self._kill_process()
                raise RuntimeError(
                    f"SubEnv {self.env_id} reset timed out after {elapsed:.1f}s"
                )
            self._force_kill_and_respawn(elapsed, reason="reset_timeout")

    def get_obs(self) -> dict:
        if self.parent_conn is None or self.process is None:
            raise RuntimeError(f"SubEnv {self.env_id} process is not running")
        self.parent_conn.send(("get_obs", None))
        return self._recv_get_obs_with_recovery()

    def _recv_get_obs_with_recovery(self) -> dict:
        start = time.time()
        if not self.parent_conn.poll(self.step_timeout_sec):
            elapsed = time.time() - start
            if self.on_timeout == "fail":
                self._kill_process()
                raise RuntimeError(
                    f"SubEnv {self.env_id} get_obs timed out after {elapsed:.1f}s"
                )
            try:
                self._force_kill_and_respawn(elapsed, reason="get_obs_timeout")
            except Exception as exc:
                logger.warning(
                    "[SUBENV_RECOVER] sub_env=%s get_obs recovery failed: %s",
                    self.env_id,
                    exc,
                )
                obs = self._obs_for_recovery()
                self._remember_obs(obs)
                return obs
            obs = self._obs_for_recovery()
            self._remember_obs(obs)
            return obs
        obs = self.parent_conn.recv()
        self._remember_obs(obs)
        return obs

    def check_seed(self, seed: int) -> dict:
        return self._send_recv("check_seed", seed, timeout=self.step_timeout_sec)

    def close(self, clear_cache: bool = True) -> None:
        if self.parent_conn is not None and self.process is not None:
            if self.process.is_alive():
                try:
                    self.parent_conn.send(("close", clear_cache))
                    if self.parent_conn.poll(self.step_timeout_sec):
                        self.parent_conn.recv()
                except (BrokenPipeError, OSError):
                    pass
        self._kill_process()


class SubprocVectorEnv(gym.Env):
    """Drop-in replacement for RoboTwin VectorEnv using subprocess isolation."""

    def __init__(
        self,
        task_config: dict,
        n_envs: int,
        env_seeds: Optional[list[int]] = None,
        instruction_type: str = "seen",
        step_timeout_sec: float = 60.0,
        max_respawns: int = 10,
        on_timeout: OnTimeoutPolicy = "truncate",
        mp_context=None,
        per_env_task_names: Optional[list[str]] = None,
    ) -> None:
        import torch.multiprocessing as mp

        self.env_seeds = env_seeds
        if self.env_seeds is not None:
            assert len(self.env_seeds) == n_envs
        self.task_name = task_config.get("task_name")
        self.n_envs = n_envs
        self.instruction_type = instruction_type
        self.step_timeout_sec = step_timeout_sec
        self.max_respawns = max_respawns
        self.on_timeout = on_timeout
        self._mp_context = mp_context or mp.get_context("spawn")

        if per_env_task_names is not None:
            assert len(per_env_task_names) == n_envs, (
                f"per_env_task_names length ({len(per_env_task_names)}) "
                f"must equal n_envs ({n_envs})"
            )
            self.per_env_task_names = [str(name) for name in per_env_task_names]
        else:
            self.per_env_task_names = [self.task_name] * n_envs

        # Shared embodiment/camera args; task_name overridden per worker.
        base_task_config = dict(task_config)
        base_task_config["task_name"] = self.per_env_task_names[0]
        self.args = build_robotwin_task_args(base_task_config, n_envs)
        self.workers: list[SubprocSubEnvWorker] = []
        self._init_workers()

    def _build_args_for_task(self, task_name: str) -> dict:
        """Clone shared args with a task-specific name and save_path suffix."""
        args = dict(self.args)
        # build_robotwin_task_args appends f"/{task_name}_reward" once; replace
        # the suffix when the shared args were built for a different task.
        save_path = str(args.get("save_path", "./data"))
        # Strip a trailing "/{any}_reward" then re-append for this task.
        if save_path.endswith("_reward"):
            parent = save_path.rsplit("/", 1)[0]
            args["save_path"] = f"{parent}/{task_name}_reward"
        else:
            args["save_path"] = f"{save_path}/{task_name}_reward"
        args["task_name"] = task_name
        return args

    def _init_workers(self) -> None:
        self.workers = []
        for i in range(self.n_envs):
            seed = self.env_seeds[i] if self.env_seeds else None
            task_name = self.per_env_task_names[i]
            self.workers.append(
                SubprocSubEnvWorker(
                    env_id=i,
                    task_name=task_name,
                    args=self._build_args_for_task(task_name),
                    env_seed=seed,
                    instruction_type=self.instruction_type,
                    step_timeout_sec=self.step_timeout_sec,
                    max_respawns=self.max_respawns,
                    on_timeout=self.on_timeout,
                    mp_context=self._mp_context,
                )
            )

    @staticmethod
    def transform(results: list[dict]) -> tuple:
        res_dict = defaultdict(list)
        for res in results:
            for k, v in res.items():
                res_dict[k].append(v)
        res_dict = dict(res_dict)
        return (
            res_dict["obs"],
            res_dict["reward"],
            res_dict["terminated"],
            res_dict["truncated"],
            res_dict["info"],
        )

    def step(self, actions: np.ndarray) -> tuple:
        if len(self.workers) == 0:
            self._init_workers()

        # Dispatch all steps first so healthy envs can run in parallel.
        for i, worker in enumerate(self.workers):
            worker.parent_conn.send(("step", actions[i]))

        results = []
        for i, worker in enumerate(self.workers):
            start = time.time()
            if not worker.parent_conn.poll(self.step_timeout_sec):
                elapsed = time.time() - start
                if self.on_timeout == "fail":
                    worker._kill_process()
                    raise RuntimeError(
                        f"SubEnv {i} step timed out after {elapsed:.1f}s"
                    )
                try:
                    worker._force_kill_and_respawn(elapsed, reason="step_timeout")
                    results.append(worker._truncate_step_result())
                except Exception as exc:
                    results.append(worker._recovery_failed_step_result(str(exc)))
            else:
                result = worker.parent_conn.recv()
                worker._remember_obs(result.get("obs"))
                results.append(normalize_subenv_step_result(result))

        return self.transform(results)

    def reset(
        self,
        env_idx: Optional[list[int] | int] = None,
        env_seeds: Optional[list[int]] = None,
    ) -> None:
        if len(self.workers) == 0:
            self._init_workers()

        if env_idx is None:
            env_idx = list(range(self.n_envs))
        elif isinstance(env_idx, (list, tuple)):
            env_idx = list(env_idx)
        elif isinstance(env_idx, torch.Tensor):
            env_idx = env_idx.tolist()
        else:
            env_idx = [env_idx]

        for idx in env_idx:
            if 0 <= idx < self.n_envs:
                seed = None
                if env_seeds is not None and len(env_seeds) == len(env_idx):
                    seed_idx = env_idx.index(idx)
                    seed = env_seeds[seed_idx]
                self.workers[idx].reset(env_seed=seed)

    def get_obs(self) -> list[dict]:
        if len(self.workers) == 0:
            self._init_workers()

        for worker in self.workers:
            worker.parent_conn.send(("get_obs", None))

        return [worker._recv_get_obs_with_recovery() for worker in self.workers]

    def close(self, clear_cache: bool = True) -> None:
        for worker in self.workers:
            worker.close(clear_cache=clear_cache)
        self.workers = []
        if clear_cache:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def check_seeds(self, seeds: list[int]) -> list[dict]:
        assert len(seeds) == self.n_envs
        return [self.workers[i].check_seed(seeds[i]) for i in range(self.n_envs)]
