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

"""Cross-interpreter SubprocVectorEnv for RoboDojo Isaac workers."""

from __future__ import annotations

import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from collections import defaultdict
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Literal, Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.robodojo.obs_action import minimal_obs

logger = logging.getLogger(__name__)

OnTimeoutPolicy = Literal["truncate", "fail"]
_KILL_GRACE_SEC = 10.0
_SPAWN_READY_TIMEOUT_SEC = 900.0
_WORKER_SCRIPT = Path(__file__).resolve().parent / "isaac_worker.py"
_K8S_NUMBERED_PORT_RE = re.compile(r"_PORT_\d+_(TCP|UDP)")
_CHILD_DROP_KEYS = frozenset({"LD_PRELOAD", "_STDBUF_E", "_STDBUF_O"})
_ACCEPT_POLL_SEC = 1.0
_LOG_TAIL_BYTES = 16384


def _to_scalar_array(value: Any, *, dtype: np.dtype) -> np.ndarray:
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


def normalize_step_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    obs = result.get("obs")
    if obs is None:
        obs = {
            "full_image": result.get("full_image"),
            "left_wrist_image": result.get("left_wrist_image"),
            "right_wrist_image": result.get("right_wrist_image"),
            "state": result.get("state"),
            "instruction": result.get("instruction", ""),
        }
        if obs["full_image"] is None:
            obs = minimal_obs(str(result.get("instruction", "")))
    normalized["obs"] = obs
    normalized["reward"] = _to_scalar_array(result.get("reward", 0), dtype=np.float32)
    normalized["terminated"] = _to_scalar_array(
        result.get("terminated", 0), dtype=np.int32
    )
    normalized["truncated"] = _to_scalar_array(
        result.get("truncated", 0), dtype=np.int32
    )
    info = result.get("info") or {}
    info.setdefault("success", bool(result.get("success", False)))
    info.setdefault("sub_env_id", result.get("sub_env_id"))
    normalized["info"] = info
    return normalized


def _drop_child_env_var(key: str, value: str) -> bool:
    """Drop vars that make Isaac execve fail (K8s ARG_MAX) or crash Kit (stdbuf)."""
    if key in _CHILD_DROP_KEYS:
        return True
    if key.startswith(("KAIC_", "KUBERNETES_")):
        return True
    if "_VPC_LB_" in key:
        return True
    if key.endswith("_SERVICE_HOST") or "_SERVICE_PORT" in key:
        return True
    if _K8S_NUMBERED_PORT_RE.search(key):
        return True
    if key.endswith("_PORT") and (
        value.startswith("tcp://") or value.startswith("udp://")
    ):
        return True
    return False


def _child_env(robodojo_path: str, extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not _drop_child_env_var(key, value)
    }
    env["ROBODOJO_PATH"] = robodojo_path
    env["OMNI_KIT_ACCEPT_EULA"] = env.get("OMNI_KIT_ACCEPT_EULA", "YES")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    xpolicy = os.path.join(robodojo_path, "XPolicyLab")
    env["PYTHONPATH"] = os.pathsep.join([robodojo_path, xpolicy])
    if extra:
        env.update(
            {
                key: value
                for key, value in extra.items()
                if not _drop_child_env_var(key, value)
            }
        )
        # Keep RoboDojo first even if extra injects a PYTHONPATH suffix.
        extra_pp = extra.get("PYTHONPATH")
        if extra_pp:
            env["PYTHONPATH"] = os.pathsep.join(
                [robodojo_path, xpolicy, extra_pp]
            )
    env.pop("LD_PRELOAD", None)
    return env


def _open_worker_log(env_id: int) -> tuple[str, Any]:
    log_dir = os.environ.get("ROBODOJO_WORKER_LOG_DIR") or os.path.join(
        tempfile.gettempdir(), "rlinf_robodojo_logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"isaac_worker_{env_id}.log")
    return path, open(path, "ab")  # noqa: SIM115


def _tail_file(path: Optional[str], nbytes: int = _LOG_TAIL_BYTES) -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - nbytes))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


class IsaacSubEnvWorker:
    """One RoboDojo python + Isaac Kit process."""

    def __init__(
        self,
        env_id: int,
        task_name: str,
        env_cfg_type: str,
        env_seed: int,
        isaac_python: str,
        robodojo_path: str,
        step_timeout_sec: float = 180.0,
        ready_timeout_sec: float = _SPAWN_READY_TIMEOUT_SEC,
        max_respawns: int = 10,
        on_timeout: OnTimeoutPolicy = "truncate",
        extra_env: Optional[dict[str, str]] = None,
    ) -> None:
        self.env_id = env_id
        self.task_name = task_name
        self.env_cfg_type = env_cfg_type
        self.env_seed = int(env_seed)
        self.isaac_python = isaac_python
        self.robodojo_path = robodojo_path
        self.step_timeout_sec = float(step_timeout_sec)
        self.ready_timeout_sec = float(ready_timeout_sec)
        self.max_respawns = int(max_respawns)
        self.on_timeout = on_timeout
        self.extra_env = extra_env or {}
        self.respawn_count = 0
        self.process: Optional[subprocess.Popen] = None
        self.conn = None
        self._listener: Optional[Listener] = None
        self._ipc_path: Optional[str] = None
        self._log_path: Optional[str] = None
        self._log_file = None
        self._last_obs: Optional[dict[str, Any]] = None
        self._spawn_process(self.env_seed)

    def _spawn_process(self, env_seed: int) -> None:
        self._cleanup_ipc()
        sock_dir = tempfile.mkdtemp(prefix=f"rlinf_robodojo_{self.env_id}_")
        self._ipc_path = os.path.join(sock_dir, "worker.sock")
        authkey = os.urandom(32)
        self._listener = Listener(self._ipc_path, family="AF_UNIX", authkey=authkey)
        cmd = [
            self.isaac_python,
            "-u",
            str(_WORKER_SCRIPT),
            "--ipc-path",
            self._ipc_path,
            "--auth-key",
            authkey.hex(),
            "--task_name",
            self.task_name,
            "--env_cfg_type",
            self.env_cfg_type,
            "--device_id",
            "0",
            "--num_envs",
            "1",
            "--seed",
            str(int(env_seed)),
            "--headless",
            "--enable_cameras",
        ]
        layout_mode = str(self.extra_env.get("ROBODOJO_LAYOUT_MODE") or "").strip()
        if layout_mode:
            cmd.extend(["--layout_mode", layout_mode])
        self._log_path, self._log_file = _open_worker_log(self.env_id)
        logger.info(
            "RoboDojo SubEnv %s isaac_worker log: %s", self.env_id, self._log_path
        )
        self.process = subprocess.Popen(
            cmd,
            cwd=self.robodojo_path,
            env=_child_env(self.robodojo_path, self.extra_env),
            stdout=self._log_file,
            stderr=self._log_file,
        )
        self.conn = self._accept_ready()
        self.env_seed = int(env_seed)

    def _worker_died_error(self) -> RuntimeError:
        rc = None if self.process is None else self.process.poll()
        if getattr(self, "_log_file", None) is not None:
            try:
                self._log_file.flush()
            except OSError:
                pass
        tail = _tail_file(getattr(self, "_log_path", None)).strip()
        detail = f"\n--- isaac_worker log tail ({getattr(self, '_log_path', None)}) ---\n{tail}"
        if tail:
            detail += "\n--- end log tail ---"
        return RuntimeError(
            f"RoboDojo SubEnv {self.env_id} isaac_worker exited rc={rc} "
            f"before IPC ready.{detail}"
        )

    def _accept_ready(self):
        if self._listener is None:
            raise RuntimeError(f"RoboDojo SubEnv {self.env_id} missing IPC listener")
        self._listener._listener._socket.settimeout(_ACCEPT_POLL_SEC)
        deadline = time.time() + self.ready_timeout_sec
        conn = None
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                err = self._worker_died_error()
                self._kill_process()
                raise err
            try:
                conn = self._listener.accept()
                break
            except (TimeoutError, socket.timeout):
                continue
            except OSError as exc:
                if self.process is not None and self.process.poll() is not None:
                    err = self._worker_died_error()
                    self._kill_process()
                    raise err from exc
                self._kill_process()
                raise RuntimeError(
                    f"RoboDojo SubEnv {self.env_id} failed to accept IPC "
                    f"within {self.ready_timeout_sec:.1f}s"
                ) from exc
        if conn is None:
            if self.process is not None and self.process.poll() is not None:
                err = self._worker_died_error()
                self._kill_process()
                raise err
            self._kill_process()
            raise RuntimeError(
                f"RoboDojo SubEnv {self.env_id} failed to accept IPC "
                f"within {self.ready_timeout_sec:.1f}s"
            )
        remaining = max(1.0, deadline - time.time())
        poll_deadline = time.time() + remaining
        while time.time() < poll_deadline:
            if self.process is not None and self.process.poll() is not None:
                err = self._worker_died_error()
                self._kill_process()
                raise err
            if conn.poll(min(_ACCEPT_POLL_SEC, max(0.1, poll_deadline - time.time()))):
                break
        else:
            self._kill_process()
            raise RuntimeError(
                f"RoboDojo SubEnv {self.env_id} failed to start within "
                f"{self.ready_timeout_sec:.1f}s"
            )
        msg = conn.recv()
        if not isinstance(msg, dict) or msg.get("status") != "ready":
            self._kill_process()
            raise RuntimeError(
                f"RoboDojo SubEnv {self.env_id} unexpected ready message: {msg!r}"
            )
        return conn

    def _cleanup_ipc(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
            self.conn = None
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        if self._ipc_path:
            sock_dir = os.path.dirname(self._ipc_path)
            try:
                shutil.rmtree(sock_dir, ignore_errors=True)
            except OSError:
                pass
            self._ipc_path = None

    def _kill_process(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=_KILL_GRACE_SEC)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=_KILL_GRACE_SEC)
            else:
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
        self.process = None
        if getattr(self, "_log_file", None) is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None
        self._cleanup_ipc()

    def _force_kill_and_respawn(self, elapsed: float, reason: str) -> None:
        old_seed = self.env_seed
        self._kill_process()
        self.respawn_count += 1
        if self.respawn_count > self.max_respawns:
            raise RuntimeError(
                f"RoboDojo SubEnv {self.env_id} exceeded subproc_max_respawns="
                f"{self.max_respawns} after {reason}"
            )
        new_seed = old_seed + 1
        logger.warning(
            "[ROBODOJO_RECOVER] sub_env=%s elapsed=%.1fs reason=%s seed=%s->%s "
            "respawn_count=%s",
            self.env_id,
            elapsed,
            reason,
            old_seed,
            new_seed,
            self.respawn_count,
        )
        self._spawn_process(new_seed)
        self.reset(new_seed)

    def _truncate_result(self, recovery_error: Optional[str] = None) -> dict:
        obs = self._last_obs if self._last_obs is not None else minimal_obs()
        info = {
            "success": False,
            "subenv_timeout": True,
            "sub_env_id": self.env_id,
            "respawn_count": self.respawn_count,
        }
        if recovery_error:
            info["recovery_error"] = recovery_error
            info["subenv_recovery_failed"] = True
        return normalize_step_result(
            {
                "obs": obs,
                "reward": 0,
                "terminated": 0,
                "truncated": 1,
                "success": False,
                "info": info,
            }
        )

    def _remember_obs(self, payload: dict[str, Any]) -> None:
        if "full_image" in payload:
            self._last_obs = {
                "full_image": payload["full_image"],
                "left_wrist_image": payload["left_wrist_image"],
                "right_wrist_image": payload["right_wrist_image"],
                "state": payload["state"],
                "instruction": payload.get("instruction", ""),
            }
        elif "obs" in payload and isinstance(payload["obs"], dict):
            self._last_obs = payload["obs"]

    def _send_recv(self, cmd: str, data: Any, timeout: Optional[float] = None) -> Any:
        if self.conn is None or self.process is None or self.process.poll() is not None:
            raise RuntimeError(f"RoboDojo SubEnv {self.env_id} is not running")
        self.conn.send((cmd, data))
        wait = self.step_timeout_sec if timeout is None else timeout
        if not self.conn.poll(wait):
            raise TimeoutError(
                f"RoboDojo SubEnv {self.env_id} {cmd} timed out after {wait:.1f}s"
            )
        return self.conn.recv()

    def step_with_timeout(self, actions: np.ndarray) -> dict:
        start = time.time()
        try:
            result = self._send_recv("step", actions)
        except (TimeoutError, EOFError, BrokenPipeError, ConnectionResetError) as exc:
            elapsed = time.time() - start
            if self.on_timeout == "fail":
                self._kill_process()
                raise RuntimeError(
                    f"RoboDojo SubEnv {self.env_id} step timed out after {elapsed:.1f}s"
                ) from exc
            try:
                self._force_kill_and_respawn(elapsed, reason="step_timeout")
            except Exception as recover_exc:
                return self._truncate_result(str(recover_exc))
            return self._truncate_result()
        self._remember_obs(result)
        result["obs"] = self._last_obs
        result["info"] = {
            "success": bool(result.get("success", False)),
            "sub_env_id": self.env_id,
        }
        return normalize_step_result(result)

    def reset(self, env_seed: Optional[int] = None) -> dict[str, Any]:
        if env_seed is not None:
            self.env_seed = int(env_seed)
        start = time.time()
        try:
            result = self._send_recv(
                "reset",
                self.env_seed,
                timeout=max(self.step_timeout_sec, self.ready_timeout_sec),
            )
        except (TimeoutError, EOFError, BrokenPipeError, ConnectionResetError):
            elapsed = time.time() - start
            if self.on_timeout == "fail":
                self._kill_process()
                raise RuntimeError(
                    f"RoboDojo SubEnv {self.env_id} reset timed out after {elapsed:.1f}s"
                )
            self._force_kill_and_respawn(elapsed, reason="reset_timeout")
            return self.get_obs()
        if isinstance(result, dict) and "obs" in result:
            self._remember_obs(result["obs"])
            return result["obs"]
        if isinstance(result, dict) and "full_image" in result:
            self._remember_obs(result)
            return self._last_obs
        return self.get_obs()

    def get_obs(self) -> dict[str, Any]:
        try:
            result = self._send_recv("get_obs", None)
        except Exception:
            if self._last_obs is not None:
                return self._last_obs
            return minimal_obs()
        self._remember_obs(result)
        return self._last_obs if self._last_obs is not None else minimal_obs()

    def close(self) -> None:
        if self.conn is not None and self.process is not None and self.process.poll() is None:
            try:
                self.conn.send(("close", None))
                self.conn.poll(min(30.0, self.step_timeout_sec))
            except (BrokenPipeError, OSError, EOFError):
                pass
        self._kill_process()


class SubprocVectorEnv(gym.Env):
    def __init__(
        self,
        task_config: dict,
        n_envs: int,
        env_seeds: Optional[list[int]] = None,
        isaac_python: str = "",
        robodojo_path: str = "",
        env_cfg_type: str = "arx_x5",
        step_timeout_sec: float = 180.0,
        ready_timeout_sec: float = _SPAWN_READY_TIMEOUT_SEC,
        max_respawns: int = 10,
        on_timeout: OnTimeoutPolicy = "truncate",
        per_env_task_names: Optional[list[str]] = None,
        extra_env: Optional[dict[str, str]] = None,
    ) -> None:
        if not isaac_python or not os.path.isfile(isaac_python):
            raise ValueError(
                f"isaac_python must be a RoboDojo conda interpreter, got {isaac_python!r}"
            )
        if not robodojo_path or not os.path.isdir(robodojo_path):
            raise ValueError(f"robodojo_path is not a directory: {robodojo_path!r}")
        self.n_envs = n_envs
        self.env_seeds = env_seeds
        if self.env_seeds is not None:
            assert len(self.env_seeds) == n_envs
        self.task_name = task_config.get("task_name")
        if per_env_task_names is not None:
            assert len(per_env_task_names) == n_envs
            self.per_env_task_names = [str(name) for name in per_env_task_names]
        else:
            self.per_env_task_names = [self.task_name] * n_envs
        self.isaac_python = isaac_python
        self.robodojo_path = robodojo_path
        self.env_cfg_type = env_cfg_type
        self.step_timeout_sec = step_timeout_sec
        self.ready_timeout_sec = ready_timeout_sec
        self.max_respawns = max_respawns
        self.on_timeout = on_timeout
        self.extra_env = extra_env or {}
        self.workers: list[IsaacSubEnvWorker] = []
        self._init_workers()

    def _init_workers(self) -> None:
        self.workers = []
        for i in range(self.n_envs):
            seed = self.env_seeds[i] if self.env_seeds else i
            self.workers.append(
                IsaacSubEnvWorker(
                    env_id=i,
                    task_name=self.per_env_task_names[i],
                    env_cfg_type=self.env_cfg_type,
                    env_seed=int(seed),
                    isaac_python=self.isaac_python,
                    robodojo_path=self.robodojo_path,
                    step_timeout_sec=self.step_timeout_sec,
                    ready_timeout_sec=self.ready_timeout_sec,
                    max_respawns=self.max_respawns,
                    on_timeout=self.on_timeout,
                    extra_env=self.extra_env,
                )
            )

    @staticmethod
    def transform(results: list[dict]) -> tuple:
        video_frames = [res.pop("video_frames", None) or [] for res in results]
        res_dict = defaultdict(list)
        for res in results:
            for key, value in res.items():
                res_dict[key].append(value)
        return (
            res_dict["obs"],
            res_dict["reward"],
            res_dict["terminated"],
            res_dict["truncated"],
            res_dict["info"],
            video_frames,
        )

    def step(self, actions: np.ndarray) -> tuple:
        results = []
        for i, worker in enumerate(self.workers):
            results.append(worker.step_with_timeout(actions[i]))
        return self.transform(results)

    def reset(
        self,
        env_idx: Optional[list[int] | int] = None,
        env_seeds: Optional[list[int]] = None,
    ) -> None:
        # enable_offload closes Isaac after init so the actor can use the GPU.
        # The next reset must spawn the workers again; an empty list is not a
        # valid env index and previously raised IndexError.
        if len(self.workers) == 0:
            logger.info(
                "RoboDojo Isaac workers were closed; respawning %s workers "
                "before reset (layout_mode=%s)",
                self.n_envs,
                self.extra_env.get("ROBODOJO_LAYOUT_MODE"),
            )
            self._init_workers()
        if env_idx is None:
            env_idx = list(range(self.n_envs))
        elif isinstance(env_idx, (list, tuple)):
            env_idx = list(env_idx)
        else:
            env_idx = [int(env_idx)]
        for i in env_idx:
            if not 0 <= int(i) < len(self.workers):
                raise IndexError(
                    f"RoboDojo reset env_idx={i} is outside "
                    f"0..{len(self.workers) - 1}"
                )
            seed = None
            if env_seeds is not None:
                seed = env_seeds[i] if i < len(env_seeds) else env_seeds[0]
            self.workers[i].reset(seed)

    def get_obs(self) -> list[dict[str, Any]]:
        return [worker.get_obs() for worker in self.workers]

    def close(self, clear_cache: bool = True) -> None:
        del clear_cache
        for worker in self.workers:
            worker.close()
        self.workers = []
