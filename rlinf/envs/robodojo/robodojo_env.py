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

"""RLinf gym wrapper for RoboDojo (Isaac Sim) post-training RL."""

from __future__ import annotations

import json
import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from rlinf.envs.robodojo.obs_action import ACTION_DIM
from rlinf.envs.robodojo.subproc_vector_env import SubprocVectorEnv
from rlinf.envs.robodojo.task_inventory import COMPETITION_TASKS, get_task_horizon
from rlinf.envs.utils import center_crop_image, list_of_dict_to_dict_of_list

__all__ = ["RoboDojoEnv"]


class RoboDojoEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.base_seed = env_seed
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.use_custom_reward = cfg.use_custom_reward
        self.use_dense_reward = bool(cfg.get("use_dense_reward", False))
        self.video_cfg = cfg.video_cfg
        self.cfg = cfg
        self.record_metrics = record_metrics
        self._is_start = True
        self.center_crop = cfg.get("center_crop", False)
        self._init_task_names()
        self._init_task_horizons()
        self._init_reset_state_ids()
        self._init_env()
        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        if self.record_metrics:
            self._init_metrics()
            self._elapsed_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

    def _init_task_names(self):
        task_names_raw = OmegaConf.select(self.cfg, "task_names", default=None)
        if task_names_raw is None:
            task_names_raw = OmegaConf.select(
                self.cfg, "task_config.task_names", default=None
            )
        if task_names_raw is not None:
            task_names = OmegaConf.to_container(task_names_raw, resolve=True)
            if not isinstance(task_names, list):
                task_names = [task_names]
        else:
            task_names = [self.cfg.task_config.task_name]
        self.task_names = [str(name) for name in task_names]
        self.num_tasks = len(self.task_names)
        assert self.num_tasks >= 1, "task_names must be non-empty"
        assert self.num_envs % self.num_tasks == 0, (
            f"num_envs ({self.num_envs}) must be divisible by num_tasks ({self.num_tasks})"
        )
        self.task_ids = [env_id % self.num_tasks for env_id in range(self.num_envs)]
        self.task_ids_tensor = torch.as_tensor(
            self.task_ids, dtype=torch.long, device=self.device
        )
        self.per_env_task_names = [self.task_names[i] for i in self.task_ids]
        self.task_name = self.task_names[0]
        if not OmegaConf.select(self.cfg, "task_config.task_name", default=None):
            OmegaConf.update(
                self.cfg, "task_config.task_name", self.task_name, merge=False
            )
        unsupported = [name for name in self.task_names if name in COMPETITION_TASKS]
        if unsupported:
            raise ValueError(
                "OpenDM Dual ARX5 does not support Franka competition tasks: "
                + ", ".join(unsupported)
            )

    def _init_task_horizons(self):
        robodojo_path = self._resolve_robodojo_path()
        horizons: list[int] = []
        for name in self.task_names:
            try:
                horizons.append(int(get_task_horizon(name, robodojo_path)))
            except Exception:
                fallback = int(self.cfg.get("max_episode_steps", 800) or 800)
                horizons.append(fallback)
        self.task_horizons = {
            name: horizon for name, horizon in zip(self.task_names, horizons)
        }
        per_env = [self.task_horizons[name] for name in self.per_env_task_names]
        self.per_env_horizons = torch.as_tensor(
            per_env, dtype=torch.long, device=self.device
        )
        max_horizon = max(horizons) if horizons else int(self.cfg.max_episode_steps)
        if bool(self.cfg.get("auto_task_horizon", True)):
            self.cfg.max_episode_steps = max_horizon
            if OmegaConf.select(self.cfg, "max_steps_per_rollout_epoch", default=None) is not None:
                self.cfg.max_steps_per_rollout_epoch = max(
                    int(self.cfg.max_steps_per_rollout_epoch), max_horizon
                )
            OmegaConf.update(
                self.cfg, "task_config.step_lim", max_horizon, merge=False
            )

    def _resolve_isaac_python(self) -> str:
        path = self.cfg.get("isaac_python") or os.environ.get("ROBODOJO_PYTHON")
        if not path:
            raise ValueError(
                "RoboDojoEnv requires env.train.isaac_python or ROBODOJO_PYTHON"
            )
        return str(path)

    def _resolve_robodojo_path(self) -> str:
        path = (
            self.cfg.get("robodojo_path")
            or os.environ.get("ROBODOJO_PATH")
            or self.cfg.get("assets_path")
        )
        if not path:
            raise ValueError(
                "RoboDojoEnv requires env.train.robodojo_path or ROBODOJO_PATH"
            )
        return str(path)

    def _init_env(self):
        env_seeds = self.reset_state_ids.tolist()
        task_config = OmegaConf.to_container(self.cfg.task_config, resolve=True)
        extra_env = {}
        extra_env["OMNI_KIT_ACCEPT_EULA"] = os.environ.get(
            "OMNI_KIT_ACCEPT_EULA", "YES"
        )
        extra_env["PYTHONNOUSERSITE"] = "1"
        runtime_libs = self.cfg.get("runtime_libs") or os.environ.get(
            "ROBODOJO_RUNTIME_LIBS"
        )
        if runtime_libs:
            extra_env["LD_LIBRARY_PATH"] = (
                f"{runtime_libs}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            )
        warp_cache = self.cfg.get("warp_cache_path") or os.environ.get("WARP_CACHE_PATH")
        if warp_cache:
            extra_env["WARP_CACHE_PATH"] = str(warp_cache)
        xdg_cache = os.environ.get("ROBODOJO_XDG_CACHE_HOME") or os.environ.get(
            "XDG_CACHE_HOME"
        )
        if xdg_cache:
            extra_env["XDG_CACHE_HOME"] = str(xdg_cache)
        eval_root = self.cfg.get("eval_root") or os.environ.get("ROBODOJO_EVAL_ROOT")
        if eval_root:
            extra_env["ROBODOJO_EVAL_ROOT"] = str(eval_root)
        layout_mode = self.cfg.get("layout_mode")
        if layout_mode in (None, "", "null"):
            layout_mode = "eval_json" if bool(self.cfg.get("is_eval", False)) else "procedural"
        extra_env["ROBODOJO_LAYOUT_MODE"] = str(layout_mode)
        if str(layout_mode) == "eval_json":
            layout_pack = self.cfg.get("layout_pack_seed")
            if layout_pack is None:
                layout_pack = os.environ.get("ROBODOJO_LAYOUT_PACK", 0)
            extra_env["ROBODOJO_LAYOUT_PACK"] = str(int(layout_pack))
        extra_env["ROBODOJO_SAVE_VIDEO"] = "1" if self._save_video_enabled() else "0"
        extra_env["ROBODOJO_VIDEO_STRIDE"] = str(self._video_stride())
        self.venv = SubprocVectorEnv(
            task_config=task_config,
            n_envs=self.num_envs,
            env_seeds=env_seeds,
            isaac_python=self._resolve_isaac_python(),
            robodojo_path=self._resolve_robodojo_path(),
            env_cfg_type=str(self.cfg.get("env_cfg_type", "arx_x5")),
            step_timeout_sec=float(self.cfg.get("subproc_step_timeout_sec", 180.0)),
            ready_timeout_sec=float(self.cfg.get("subproc_ready_timeout_sec", 900.0)),
            max_respawns=int(self.cfg.get("subproc_max_respawns", 10)),
            on_timeout=str(self.cfg.get("on_subenv_timeout", "truncate")),
            per_env_task_names=self.per_env_task_names,
            extra_env=extra_env,
        )

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
                self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
                self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            if isinstance(infos["success"], list):
                infos["success"] = torch.as_tensor(
                    np.array(infos["success"]).reshape(-1), device=self.device
                )
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
            for task_idx, task_name in enumerate(self.task_names):
                mask = self.task_ids_tensor == task_idx
                episode_info[f"success_once/{task_name}"] = (
                    self.success_once & mask
                ).clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["task_ids"] = self.task_ids_tensor.clone()
        infos["episode"] = episode_info
        return infos

    def center_and_crop(self, image, center_crop=False):
        image = Image.fromarray(np.array(image)).convert("RGB")
        if center_crop:
            image = center_crop_image(image)
        return np.array(image)

    def _extract_obs_image(self, raw_obs):
        batch_images = []
        per_env_wrists = []
        batch_states = []
        batch_instructions = []
        for obs in raw_obs:
            batch_images.append(
                self.center_and_crop(obs["full_image"], center_crop=self.center_crop)
            )
            wrists = [
                self.center_and_crop(obs["left_wrist_image"], center_crop=self.center_crop),
                self.center_and_crop(
                    obs["right_wrist_image"], center_crop=self.center_crop
                ),
            ]
            per_env_wrists.append(wrists)
            batch_states.append(np.asarray(obs["state"], dtype=np.float32))
            batch_instructions.append(obs.get("instruction", ""))
        batch_images = torch.stack([torch.from_numpy(img) for img in batch_images])
        batch_wrist_images = torch.stack(
            [
                torch.stack([torch.from_numpy(np.asarray(img)) for img in wrists])
                for wrists in per_env_wrists
            ]
        )
        batch_states = torch.stack(
            [torch.from_numpy(state) for state in batch_states]
        )
        return {
            "main_images": batch_images,
            "wrist_images": batch_wrist_images,
            "states": batch_states,
            "task_descriptions": batch_instructions,
            "task_ids": self.task_ids_tensor.clone(),
        }

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward
        if self.use_rel_reward:
            return reward_diff
        return reward

    def _to_tensor_reward(self, step_reward):
        if isinstance(step_reward, torch.Tensor):
            return step_reward.to(dtype=torch.float32, device=self.device)
        return torch.as_tensor(
            np.array(step_reward, dtype=np.float32).reshape(-1),
            device=self.device,
        )

    def _prepare_step_reward(self, step_reward, terminations: torch.Tensor):
        if self.use_custom_reward:
            return self._calc_step_reward(terminations)
        step_reward = self._to_tensor_reward(step_reward)
        if self.use_rel_reward:
            reward_diff = step_reward - self.prev_step_reward
            self.prev_step_reward = step_reward
            return reward_diff
        return step_reward

    def _cal_chunk_rewards(
        self, step_reward: torch.Tensor, chunk_step: int, terminations: torch.Tensor
    ) -> torch.Tensor:
        chunk_rewards = torch.zeros(
            self.num_envs, chunk_step, dtype=torch.float32, device=self.device
        )
        if chunk_step <= 0:
            return chunk_rewards
        chunk_rewards[:, chunk_step - 1] = step_reward
        return chunk_rewards

    def reset(self, env_idx=None, env_seeds=None):
        if self._is_start:
            self._is_start = False
        if env_seeds is None and not self.use_fixed_reset_state_ids:
            self.update_reset_state_ids(env_idx=env_idx)
        env_seeds = self.reset_state_ids.tolist() if env_seeds is None else env_seeds
        self.venv.reset(env_idx=env_idx, env_seeds=env_seeds)
        raw_obs = self.venv.get_obs()
        self._reset_metrics(env_idx)
        return self._extract_obs_image(raw_obs), {}

    def step(self, actions=None, auto_reset=True):
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        elif isinstance(actions, dict):
            actions = actions.get("actions", actions)
        if len(actions.shape) == 2:
            actions = actions[:, None, :]
        raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(
            actions
        )[:5]
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)
        terminations = torch.as_tensor(
            np.array(terminations).reshape(-1), device=self.device
        )
        truncations = torch.as_tensor(
            np.array(truncations).reshape(-1), device=self.device
        )
        step_reward = self._prepare_step_reward(step_reward, terminations)
        self._elapsed_steps += actions.shape[1]
        truncated = torch.logical_or(
            self._elapsed_steps >= self.cfg.max_episode_steps,
            self._elapsed_steps >= self.per_env_horizons,
        )
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)
        infos = self._record_metrics(step_reward, infos)
        if self.ignore_terminations:
            terminations[:] = False
        dones = torch.logical_or(terminations, truncations)
        if dones.any() and auto_reset and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)
        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.cpu().numpy()
        num_envs = chunk_actions.shape[0]
        chunk_step = chunk_actions.shape[1]
        (
            raw_obs,
            step_reward,
            terminations,
            truncations,
            info_list,
            video_frames,
        ) = self.venv.step(chunk_actions)
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)
        obs_list = self._head_video_obs_list(video_frames, extracted_obs)
        infos_list = [infos]
        terminations = torch.as_tensor(
            np.array(terminations).reshape(-1), device=self.device
        )
        truncations = torch.as_tensor(
            np.array(truncations).reshape(-1), device=self.device
        )
        step_reward = self._prepare_step_reward(step_reward, terminations)
        chunk_rewards = self._cal_chunk_rewards(step_reward, chunk_step, terminations)
        self._elapsed_steps += chunk_actions.shape[1]
        truncated = torch.logical_or(
            self._elapsed_steps >= self.cfg.max_episode_steps,
            self._elapsed_steps >= self.per_env_horizons,
        )
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)
        infos = self._record_metrics(step_reward, infos)
        if self.ignore_terminations:
            terminations[:] = False
        past_dones = torch.logical_or(terminations, truncations)
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )
        chunk_terminations = torch.zeros((num_envs, chunk_step), dtype=bool)
        chunk_terminations[:, -1] = terminations
        chunk_truncations = torch.zeros((num_envs, chunk_step), dtype=bool)
        chunk_truncations[:, -1] = truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        final_info = infos.copy()
        extracted_obs, infos = self.reset(env_idx=env_idx.tolist())
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def offload(self, clear_cache=True):
        if hasattr(self, "venv"):
            self.venv.close(clear_cache)

    def sample_action_space(self):
        return np.random.randn(self.num_envs, 1, ACTION_DIM)

    def _save_video_enabled(self) -> bool:
        video_cfg = self.cfg.get("video_cfg")
        if video_cfg is None:
            return False
        return bool(video_cfg.get("save_video", False))

    def _video_stride(self) -> int:
        """YAML ``video_stride``. ``-1`` keeps every control step."""
        video_cfg = self.cfg.get("video_cfg")
        if video_cfg is None:
            return -1
        raw = video_cfg.get("video_stride", -1)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return -1

    def _head_video_obs_list(self, video_frames, extracted_obs) -> list:
        """Head-camera frames for RecordVideo. The last item stays the policy obs.

        Wrist images are not included. ``video_frames`` is one list of HWC uint8
        arrays per env, already subsampled in the Isaac worker.
        """
        if not self._save_video_enabled() or not video_frames:
            return [extracted_obs]
        lengths = [len(frames or []) for frames in video_frames]
        n_frames = max(lengths) if lengths else 0
        if n_frames <= 1:
            return [extracted_obs]
        sequence = []
        for t in range(n_frames - 1):
            images = []
            for frames in video_frames:
                frames = frames or []
                if not frames:
                    images.append(np.zeros((1, 1, 3), dtype=np.uint8))
                    continue
                src = frames[t] if t < len(frames) else frames[-1]
                images.append(
                    self.center_and_crop(src, center_crop=self.center_crop)
                )
            sequence.append(
                {
                    "main_images": np.stack(images),
                    "task_descriptions": list(self.per_env_task_names),
                }
            )
        sequence.append(extracted_obs)
        return sequence

    def _load_success_seeds(self) -> Optional[torch.Tensor]:
        seeds_path = self.cfg.get("seeds_path", None)
        if not seeds_path or not os.path.exists(seeds_path):
            return None
        with open(seeds_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return torch.as_tensor(data, dtype=torch.long)
        pooled = []
        for task_name in self.task_names:
            if task_name not in data:
                return None
            item = data[task_name]
            if isinstance(item, list):
                seeds = item
            else:
                seeds = item.get("success_seeds") or item.get("seeds")
            if not seeds:
                return None
            pooled.extend(seeds)
        return torch.as_tensor(pooled, dtype=torch.long) if pooled else None

    def _init_reset_state_ids(self):
        self.success_seeds = self._load_success_seeds()
        self._current_seed_index = 0
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self, env_idx=None):
        if self.use_fixed_reset_state_ids and hasattr(self, "reset_state_ids"):
            return
        if self.success_seeds is not None and self.success_seeds.numel() > 0:
            total = self.success_seeds.numel()
            indices = (
                torch.arange(self.num_group, device=self.success_seeds.device)
                + self._current_seed_index
            ) % total
            reset_state_ids = self.success_seeds[indices].repeat_interleave(
                repeats=self.group_size
            )
            self._current_seed_index = (self._current_seed_index + self.num_group) % total
        else:
            reset_state_ids = torch.randint(
                low=0,
                high=100000,
                size=(self.num_group,),
                generator=self._generator,
            ).repeat_interleave(repeats=self.group_size)
        if env_idx is not None and hasattr(self, "reset_state_ids"):
            for idx in env_idx:
                self.reset_state_ids[idx] = reset_state_ids[idx]
        else:
            self.reset_state_ids = reset_state_ids
