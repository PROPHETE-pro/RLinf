#!/usr/bin/env python3
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

"""Isaac Sim worker process for RoboDojo RL (runs under RoboDojo python).

Standalone smoke::

    python isaac_worker.py --standalone --task_name stack_bowls \\
        --env_cfg_type arx_x5 --headless --enable_cameras --steps 2

IPC worker (started by SubprocVectorEnv)::

    python isaac_worker.py --ipc-path /tmp/xxx.sock --auth-key HEX ...
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from obs_action import (  # noqa: E402
    DUAL_X5_DIM_INFO,
    extract_policy_obs,
    pack_joint_state,
    unpack_joint_action,
)


class DummyModelClient:
    """No-op stand-in so EvalEnv never opens a WebSocket policy server."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def call(self, func_name=None, obs=None, **kwargs):
        del obs, kwargs
        if func_name in ("get_action", "get_action_batch"):
            raise RuntimeError("RL Isaac worker must not query a policy server")
        return None

    def close(self):
        return None


def _episode_flags(env) -> tuple[float, bool, bool, bool]:
    # EvalEnv.success starts True ("not failed yet") and is only meaningful
    # together with end_flag after RewardManager / is_episode_end.
    ended = bool(env.end_flag[0])
    success = bool(ended and env.success[0])
    truncated = bool(ended and not env.success[0])
    terminated = success
    reward = 1.0 if success else 0.0
    return float(reward), terminated, truncated, success


def _pack_state(env, raw_obs: dict[str, Any]) -> np.ndarray:
    try:
        from utils.process_data import pack_robot_state

        dim_info = getattr(env, "robot_action_dim_info", None) or DUAL_X5_DIM_INFO
        packed = pack_robot_state(
            raw_obs, action_type="joint", robot_action_dim_info=dim_info
        )
        return np.asarray(packed, dtype=np.float32).reshape(-1)
    except Exception:
        return pack_joint_state(raw_obs.get("state") or {}).astype(np.float32)


def _obs_payload(env) -> dict[str, Any]:
    raw = env.get_obs()
    payload = extract_policy_obs(raw)
    payload["state"] = _pack_state(env, raw)
    reward, terminated, truncated, success = _episode_flags(env)
    payload.update(
        {
            "reward": np.array([reward], dtype=np.float32),
            "terminated": np.array([int(terminated)], dtype=np.int32),
            "truncated": np.array([int(truncated)], dtype=np.int32),
            "success": success,
            "step_count": int(env.take_action_cnt[0]),
            "step_lim": int(env.step_lim),
        }
    )
    return payload


def resolve_layout_pack_id(layout_root: str | Path, requested: int) -> int:
    """Pick an existing Eval_Layout pack directory (read-only Assets; never create)."""
    root = Path(layout_root)
    if not root.is_dir():
        raise FileNotFoundError(
            "Eval_Layout pack root is missing (Assets is read-only, do not create "
            f"it): {root}"
        )
    packs = sorted(
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if not packs:
        raise FileNotFoundError(
            f"no numeric Eval_Layout packs under {root} (Assets is read-only)"
        )
    requested = int(requested)
    if requested in packs:
        return requested
    mapped = packs[requested % len(packs)]
    print(
        f"[isaac_worker] layout pack {requested} is not on disk; "
        f"using existing pack {mapped} under {root}",
        flush=True,
    )
    return mapped


def _n_layouts(env) -> int:
    seed_info = getattr(getattr(env, "seed_manager", None), "seed_info", None) or {}
    return max(1, len(seed_info))


def _layout_mode(env) -> str:
    return str(getattr(env, "layout_mode", None) or os.environ.get("ROBODOJO_LAYOUT_MODE", "eval_json"))


def _reset_with_retries(env, seed: int, max_tries: int = 20):
    last_error: Optional[BaseException] = None
    layout_mode = _layout_mode(env)
    n_layouts = _n_layouts(env)
    base = int(seed)
    try:
        from utils.cluttered_generator import UnStableError
    except Exception:
        UnStableError = type("UnStableError", (Exception,), {})  # type: ignore[misc,assignment]
    if layout_mode == "procedural":
        tries = max(1, max_tries)
    else:
        tries = min(max(1, max_tries), n_layouts)
    for offset in range(tries):
        if layout_mode == "procedural":
            try_seed = base + offset
        else:
            try_seed = (base + offset) % n_layouts
        try:
            env.reset(seed=[try_seed])
            if hasattr(env, "run_reward"):
                env.run_reward()
            return try_seed
        except UnStableError as exc:
            last_error = exc
            print(
                f"[isaac_worker] reset seed={try_seed} unstable: {exc}",
                flush=True,
            )
        except Exception:
            raise
    raise RuntimeError(f"reset failed after {tries} layout seeds") from last_error


def select_video_indices(n_steps: int, stride: int) -> list[int]:
    """Indices of executed actions to keep for the head-camera video.

    ``stride <= 0`` (YAML default ``-1``) keeps every step. A positive stride
    keeps one frame every N steps and always keeps the last executed step.
    """
    if n_steps <= 0:
        return []
    every = 1 if int(stride) <= 0 else int(stride)
    indices = list(range(0, n_steps, every))
    if indices[-1] != n_steps - 1:
        indices.append(n_steps - 1)
    return indices


def _video_record_every() -> Optional[int]:
    """Head-video stride, or None when save_video is off.

    ``ROBODOJO_VIDEO_STRIDE <= 0`` records every executed action.
    """
    save = os.environ.get("ROBODOJO_SAVE_VIDEO", "0").strip().lower()
    if save not in ("1", "true", "yes"):
        return None
    raw = os.environ.get("ROBODOJO_VIDEO_STRIDE", "-1").strip()
    try:
        stride = int(raw)
    except ValueError:
        stride = -1
    return 1 if stride <= 0 else stride


def _step_chunk(env, actions: np.ndarray) -> dict[str, Any]:
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2 or chunk.shape[-1] != 14:
        raise ValueError(f"expected (H, 14) actions, got {chunk.shape}")
    every = _video_record_every()
    last_payload = None
    head_frames: list[np.ndarray] = []
    for row in chunk:
        if env.end_flag[0]:
            break
        env.take_action(unpack_joint_action(row))
        last_payload = _obs_payload(env)
        if every is not None:
            head_frames.append(np.array(last_payload["full_image"], copy=True))
    if last_payload is None:
        last_payload = _obs_payload(env)
    if every is not None and head_frames:
        kept = select_video_indices(len(head_frames), -1 if every == 1 else every)
        last_payload = dict(last_payload)
        last_payload["video_frames"] = [head_frames[i] for i in kept]
    return last_payload


def _build_env(args_cli):
    from omegaconf import OmegaConf

    from env.global_configs import ASSETS_PATH, BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    from src.eval_client import eval_env as eval_env_mod
    from src.eval_client.eval_env import create_eval_env
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    eval_env_mod.WsModelClient = DummyModelClient

    task_name = args_cli.task_name
    eval_cfg_name = args_cli.env_cfg_type
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, eval_cfg_name + ".yml"))
    eval_cfg["task_name"] = task_name
    eval_cfg["num_envs"] = 1
    eval_cfg["device_id"] = args_cli.device_id
    eval_cfg["eval_batch"] = False
    eval_cfg["policy_name"] = "OpenDM"
    eval_cfg["additional_info"] = "rlinf_rl"
    eval_cfg["physx_monitor_enabled"] = False
    eval_cfg["config_name"] = eval_cfg.get("config_name", eval_cfg_name)
    layout_mode = str(
        getattr(args_cli, "layout_mode", None)
        or os.environ.get("ROBODOJO_LAYOUT_MODE")
        or "eval_json"
    )
    eval_cfg["layout_mode"] = layout_mode
    # SeedManager.init_eval uses eval_cfg.seed as the pack directory name under
    # Assets/Eval_Layout/<benchmark>/<config_name>/<seed>. That tree is a
    # read-only symlink; procedural training uses seed as RNG instead.
    if layout_mode == "eval_json":
        if getattr(args_cli, "layout_pack", None) is not None:
            os.environ["ROBODOJO_LAYOUT_PACK"] = str(int(args_cli.layout_pack))
        requested_pack = int(os.environ.get("ROBODOJO_LAYOUT_PACK", "0"))
        layout_root = os.path.join(
            ASSETS_PATH, "Eval_Layout", BENCHMARK, eval_cfg["config_name"]
        )
        eval_cfg["seed"] = resolve_layout_pack_id(layout_root, requested_pack)
        print(
            f"[isaac_worker] layout_mode=eval_json pack={eval_cfg['seed']} "
            f"requested={requested_pack} root={layout_root}",
            flush=True,
        )
    else:
        eval_cfg["seed"] = int(getattr(args_cli, "seed", 0) or 0)
        print(
            f"[isaac_worker] layout_mode={layout_mode} rng_seed={eval_cfg['seed']}",
            flush=True,
        )
    every = _video_record_every()
    if every is None:
        print("[isaac_worker] head video: off", flush=True)
    elif every == 1:
        print("[isaac_worker] head video: every action", flush=True)
    else:
        print(f"[isaac_worker] head video: every {every} actions", flush=True)

    deploy_cfg = {
        "policy_name": "OpenDM",
        "port": 0,
        "host": "127.0.0.1",
        "protocol": "ws",
        "policy_server_url": "ws://127.0.0.1:0",
        "evaluation_id": os.environ.get("ROBODOJO_RUN_ID", "rlinf-rl"),
        "trial_id": f"{task_name}-rlinf-rl",
        "action_case_id": f"{task_name}_rl",
        "repeat_index": None,
    }

    import importlib

    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    benchmark_path = os.path.join(ROOT_DIR, "task", BENCHMARK)
    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(
                os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")
            ),
            "scene": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml"
                )
            ),
            "camera": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "camera",
                    eval_cfg["config"]["camera"] + ".yml",
                )
            ),
            "robot": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml"
                )
            ),
            "task_env": load_yaml(
                task_registry.task_config_path(
                    os.path.join(benchmark_path, "config"), task_name
                )
            ),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        }
    )
    OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, eval_num = process_config(env_cfg, task_name=task_name)
    eval_cfg["eval_num"] = int(eval_num)
    OmegaConf.update(env_cfg, "eval_cfg.eval_num", int(eval_num), force_add=True)
    OmegaConf.update(
        env_cfg,
        "camera.default_frequency",
        eval_cfg.get("observation", {}).get("collect_freq", 0),
        force_add=True,
    )
    env_cfg.sim.seed = [int(args_cli.seed)]

    simulation_app = args_cli._simulation_app
    env = create_eval_env(env_cfg, app=simulation_app)
    # RL does not persist eval videos; skip ffmpeg streams and last-frame dumps.
    env._stream_vision = lambda *args, **kwargs: None  # type: ignore[method-assign]
    env.save_video = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return env, simulation_app


def _serve_ipc(env, ipc_path: str, auth_key: bytes) -> None:
    from multiprocessing.connection import Client

    conn = Client(ipc_path, family="AF_UNIX", authkey=auth_key)
    conn.send({"status": "ready"})
    try:
        while True:
            try:
                cmd, data = conn.recv()
            except EOFError:
                break
            if cmd == "reset":
                seed = int(data) if data is not None else 0
                used = _reset_with_retries(env, seed)
                print(
                    f"[isaac_worker] reset done layout_mode={_layout_mode(env)} "
                    f"seed={used}",
                    flush=True,
                )
                conn.send({"status": "ok", "seed": used, "obs": _obs_payload(env)})
            elif cmd == "step":
                payload = _step_chunk(env, data)
                conn.send(payload)
            elif cmd == "get_obs":
                conn.send(_obs_payload(env))
            elif cmd == "close":
                conn.send({"status": "ok"})
                conn.close()
                break
            else:
                conn.send({"status": "error", "error": f"unknown command {cmd}"})
    finally:
        try:
            env.close()
        except Exception:
            traceback.print_exc()


def _run_standalone(env, steps: int, seed: int) -> None:
    used = _reset_with_retries(env, seed)
    obs = _obs_payload(env)
    print(
        f"[isaac_worker] reset seed={used} "
        f"head={obs['full_image'].shape} state={obs['state'].shape} "
        f"instruction={obs['instruction']!r}",
        flush=True,
    )
    hold = obs["state"]
    for i in range(max(0, steps)):
        payload = _step_chunk(env, hold[None, :])
        print(
            f"[isaac_worker] step={i + 1} reward={payload['reward']} "
            f"term={payload['terminated']} trunc={payload['truncated']} "
            f"success={payload['success']}",
            flush=True,
        )
    env.close()


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="RoboDojo Isaac RL worker")
    parser.add_argument("--task_name", "--task", dest="task_name", type=str, default="stack_bowls")
    parser.add_argument(
        "--env_cfg_type", "--env-cfg", dest="env_cfg_type", type=str, default="arx_x5"
    )
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout_pack", type=int, default=None)
    parser.add_argument(
        "--layout_mode",
        type=str,
        default=None,
        choices=["procedural", "eval_json"],
        help="procedural: sample task YAML; eval_json: replay Eval_Layout packs",
    )
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--ipc-path", type=str, default="")
    parser.add_argument("--auth-key", type=str, default="")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if not args.headless:
        args.headless = True
    return args


def main(argv: Optional[list[str]] = None) -> int:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ROBODOJO_RUN_ID", "rlinf-rl")
    robodojo_path = os.environ.get("ROBODOJO_PATH")
    if robodojo_path:
        os.chdir(robodojo_path)
        xpolicy = os.path.join(robodojo_path, "XPolicyLab")
        # RoboDojo `utils` must precede XPolicyLab `utils`.
        for path in (xpolicy, robodojo_path):
            while path in sys.path:
                sys.path.remove(path)
        sys.path.insert(0, robodojo_path)
        sys.path.insert(1, xpolicy)

    args = parse_args(argv)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    args._simulation_app = app_launcher.app
    env = None
    try:
        env, simulation_app = _build_env(args)
        if args.standalone or not args.ipc_path:
            _run_standalone(env, steps=args.steps, seed=args.seed)
        else:
            auth = bytes.fromhex(args.auth_key)
            _serve_ipc(env, args.ipc_path, auth)
        try:
            simulation_app.close()
        except Exception:
            traceback.print_exc()
        return 0
    except Exception:
        traceback.print_exc()
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            args._simulation_app.close()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
