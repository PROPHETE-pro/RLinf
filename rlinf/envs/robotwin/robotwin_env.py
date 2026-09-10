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

import json
import os
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from PIL import Image

from rlinf.envs.robotwin.seed_utils import partition_success_seeds
from rlinf.envs.utils import center_crop_image, list_of_dict_to_dict_of_list

__all__ = ["RoboTwinEnv"]


class RoboTwinEnv(gym.Env):
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

        self._init_task_names()

        self.center_crop = cfg.get("center_crop", False)
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
        """Resolve task_names and assign env_id % N (Robocasa-style even split)."""
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
        total_num_envs = self.cfg.get("total_num_envs", None)
        if total_num_envs is not None:
            assert int(total_num_envs) % self.num_tasks == 0, (
                f"total_num_envs ({total_num_envs}) must be divisible by "
                f"num_tasks ({self.num_tasks})"
            )

        self.task_ids = [env_id % self.num_tasks for env_id in range(self.num_envs)]
        self.task_ids_tensor = torch.as_tensor(
            self.task_ids, dtype=torch.long, device=self.device
        )
        self.per_env_task_names = [self.task_names[i] for i in self.task_ids]
        # Backward-compatible single-task field (first / only task).
        self.task_name = self.task_names[0]
        if not OmegaConf.select(self.cfg, "task_config.task_name", default=None):
            OmegaConf.update(self.cfg, "task_config.task_name", self.task_name, merge=False)

    def _init_env(self):
        mp.set_start_method("spawn", force=True)
        os.environ["ASSETS_PATH"] = self.cfg.assets_path
        # RoboTwin's cluttered-object loader uses cwd-relative ./assets/...;
        # Ray EnvWorkers start in the RLinf repo, so chdir to the RoboTwin root.
        if self.cfg.assets_path and os.path.isdir(self.cfg.assets_path):
            os.chdir(self.cfg.assets_path)

        env_seeds = self.reset_state_ids.tolist()
        task_config = OmegaConf.to_container(self.cfg.task_config, resolve=True)
        for key in ("use_dense_reward", "dense_shaping_coef", "dense_success_reward"):
            if OmegaConf.select(self.cfg, key, default=None) is not None:
                task_config[key] = self.cfg.get(key)
        use_subproc = self.cfg.get("use_subproc_env", True)

        if use_subproc:
            from rlinf.envs.robotwin.subproc_vector_env import SubprocVectorEnv

            self.venv = SubprocVectorEnv(
                task_config=task_config,
                n_envs=self.num_envs,
                env_seeds=env_seeds,
                instruction_type=self.cfg.get("instruction_type", "seen"),
                step_timeout_sec=self.cfg.get("subproc_step_timeout_sec", 60.0),
                max_respawns=self.cfg.get("subproc_max_respawns", 10),
                on_timeout=self.cfg.get("on_subenv_timeout", "truncate"),
                per_env_task_names=self.per_env_task_names,
            )
        else:
            if self.num_tasks > 1:
                raise ValueError(
                    "RoboTwin multitask (N>1) requires use_subproc_env=true "
                    "so each SubEnv can receive its own task_name."
                )
            from robotwin.envs.vector_env import VectorEnv

            self.venv = VectorEnv(
                task_config=task_config,
                n_envs=self.num_envs,
                env_seeds=env_seeds,
                instruction_type=self.cfg.get("instruction_type", "seen"),
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
            # Per-task success so N>1 metrics are not collapsed by averaging.
            for task_idx, task_name in enumerate(self.task_names):
                mask = self.task_ids_tensor == task_idx
                # Non-matching envs stay False so aggregation over done envs of
                # other tasks does not invent successes for this task; callers
                # that need rates should group by task_ids / task-specific keys.
                episode_info[f"success_once/{task_name}"] = (
                    self.success_once & mask
                ).clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["task_ids"] = self.task_ids_tensor.clone()
        for task_idx, task_name in enumerate(self.task_names):
            mask = self.task_ids_tensor == task_idx
            episode_info[f"return/{task_name}"] = torch.where(
                mask, self.returns, torch.zeros_like(self.returns)
            )
        infos["episode"] = episode_info
        return infos

    def center_and_crop(self, image, center_crop=False):
        image = np.array(image)

        image = Image.fromarray(image).convert("RGB")
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
            wrist_images = []
            if "left_wrist_image" in obs and obs["left_wrist_image"] is not None:
                wrist_images.append(
                    self.center_and_crop(
                        obs["left_wrist_image"], center_crop=self.center_crop
                    )
                )
            if "right_wrist_image" in obs and obs["right_wrist_image"] is not None:
                wrist_images.append(
                    self.center_and_crop(
                        obs["right_wrist_image"], center_crop=self.center_crop
                    )
                )
            per_env_wrists.append(wrist_images)
            batch_states.append(obs["state"])
            batch_instructions.append(obs["instruction"])

        batch_images = torch.stack([torch.from_numpy(img) for img in batch_images])
        max_wrists = max((len(wrists) for wrists in per_env_wrists), default=0)
        if max_wrists > 0:
            ref_wrist = next(
                (wrists[0] for wrists in per_env_wrists if wrists),
                batch_images[0].numpy(),
            )
            padded_wrists = []
            for wrists in per_env_wrists:
                imgs = list(wrists)
                while len(imgs) < max_wrists:
                    imgs.append(np.zeros_like(ref_wrist))
                padded_wrists.append(
                    torch.stack([torch.from_numpy(np.asarray(img)) for img in imgs])
                )
            batch_wrist_images = torch.stack(padded_wrists)
        else:
            batch_wrist_images = None
        batch_states = torch.stack([torch.from_numpy(state) for state in batch_states])

        extracted_obs = {
            "main_images": batch_images,
            "wrist_images": batch_wrist_images,
            "states": batch_states,
            "task_descriptions": batch_instructions,
            "task_ids": self.task_ids_tensor.clone(),
        }

        return extracted_obs

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations

        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _to_tensor_reward(self, step_reward):
        if isinstance(step_reward, torch.Tensor):
            return step_reward.to(dtype=torch.float32, device=self.device)
        return torch.as_tensor(
            np.array(step_reward, dtype=np.float32).reshape(-1),
            device=self.device,
        )

    def _apply_reward_delta(self, step_reward: torch.Tensor) -> torch.Tensor:
        """Convert cumulative env rewards into per-chunk increments for PPO."""
        reward_diff = step_reward - self.prev_step_reward
        self.prev_step_reward = step_reward
        return reward_diff

    def _prepare_step_reward(
        self,
        step_reward,
        terminations: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize env reward into the incremental signal consumed by PPO."""
        if self.use_custom_reward:
            return self._calc_step_reward(terminations)

        step_reward = self._to_tensor_reward(step_reward)
        # gen_dense_reward_once returns cumulative absolute progress; always delta it.
        if self.use_dense_reward or self.use_rel_reward:
            return self._apply_reward_delta(step_reward)
        return step_reward

    def _cal_chunk_rewards(
        self,
        step_reward: torch.Tensor,
        chunk_step: int,
        terminations: torch.Tensor,
    ) -> torch.Tensor:
        """Map one env scalar reward onto the last action-chunk slot for PPO."""
        chunk_rewards = torch.zeros(
            self.num_envs, chunk_step, dtype=torch.float32, device=self.device
        )
        if chunk_step <= 0:
            return chunk_rewards

        reward_idx = chunk_step - 1
        if self.use_custom_reward or self.use_dense_reward or self.use_rel_reward:
            chunk_rewards[:, reward_idx] = step_reward
            return chunk_rewards

        # Legacy absolute-reward path: spread terminal reward across the chunk tail.
        for env_id in range(self.num_envs):
            reward = step_reward[env_id]
            if terminations[env_id]:
                chunk_rewards[env_id, reward_idx:] = reward
            elif reward != 0:
                chunk_rewards[env_id, reward_idx] = reward
        return chunk_rewards

    def reset(
        self,
        env_idx: Optional[Union[int, list[int]]] = None,
        env_seeds=None,
    ):
        if self._is_start:
            self._is_start = False

        env_seeds = self.reset_state_ids.tolist() if env_seeds is None else env_seeds

        self.venv.reset(env_idx=env_idx, env_seeds=env_seeds)
        raw_obs = self.venv.get_obs()
        infos = {}

        self._reset_metrics(env_idx)

        extracted_obs = self._extract_obs_image(raw_obs)

        return extracted_obs, infos

    def step(
        self, actions: Union[torch.Tensor, np.ndarray, dict] = None, auto_reset=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        elif isinstance(actions, dict):
            actions = actions.get("actions", actions)

        # [n_envs, horizon, action_dim]
        if len(actions.shape) == 2:
            # [n_envs, action_dim] -> [n_envs, 1, action_dim]
            actions = actions[:, None, :]

        raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(
            actions
        )
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)

        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )

        if self.use_custom_reward:
            step_reward = self._calc_step_reward(terminations)
        else:
            step_reward = self._prepare_step_reward(step_reward, terminations)

        self._elapsed_steps += actions.shape[1]
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)

        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.cpu().numpy()

        # chunk_actions: [num_envs, chunk_step, action_dim]
        num_envs = chunk_actions.shape[0]
        chunk_step = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(
            chunk_actions
        )
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)
        obs_list.append(extracted_obs)
        infos_list.append(infos)
        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )

        if self.use_custom_reward:
            step_reward = self._calc_step_reward(terminations)
        else:
            step_reward = self._prepare_step_reward(step_reward, terminations)

        chunk_rewards = self._cal_chunk_rewards(
            step_reward, chunk_step, terminations
        )

        self._elapsed_steps += chunk_actions.shape[1]
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()

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
        if self.cfg.is_eval:
            self.update_reset_state_ids(env_idx=env_idx)

        extracted_obs, infos = self.reset(env_idx=env_idx.tolist())
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
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
        return np.random.randn(self.num_envs, self.horizon, 14)

    def _init_reset_state_ids(self):
        if self.cfg.get("seeds_path", None) is not None and os.path.exists(
            self.cfg.seeds_path
        ):
            with open(self.cfg.seeds_path, "r") as f:
                data = json.load(f)
            # Multitask: concatenate partitioned success seeds from each task.
            per_task_seeds = []
            for task_name in self.task_names:
                if task_name not in data:
                    per_task_seeds = None
                    break
                success_seeds = data[task_name].get("success_seeds", None)
                if success_seeds is None:
                    per_task_seeds = None
                    break
                success_seeds = torch.as_tensor(success_seeds, dtype=torch.long)
                partitioned = partition_success_seeds(
                    success_seeds,
                    base_seed=self.base_seed,
                    seed_offset=self.seed_offset,
                    total_num_processes=self.total_num_processes,
                    num_group=max(1, self.num_group // self.num_tasks)
                    if self.num_tasks > 1
                    else self.num_group,
                )
                per_task_seeds.append(partitioned)

            if per_task_seeds is not None and all(s.numel() > 0 for s in per_task_seeds):
                # Round-robin merge so env_id % N maps to task N's seed pool.
                min_len = min(s.numel() for s in per_task_seeds)
                merged = []
                for i in range(min_len):
                    for task_idx in range(self.num_tasks):
                        merged.append(per_task_seeds[task_idx][i])
                self.success_seeds = torch.stack(merged)
                self._current_seed_index = 0
            else:
                self.success_seeds = None
                self._current_seed_index = 0
        else:
            self.success_seeds = None
            self._current_seed_index = 0

        if not hasattr(self, "_generator"):
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self, env_idx=None):
        if self.use_fixed_reset_state_ids and hasattr(self, "reset_state_ids"):
            return

        if env_idx is not None and hasattr(self, "reset_state_ids"):
            if self.success_seeds is not None:
                total_seeds = self.success_seeds.numel()
                indices = (
                    torch.arange(self.num_group, device=self.success_seeds.device)
                    + self._current_seed_index
                ) % total_seeds
                reset_state_ids = self.success_seeds[indices]
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
                self._current_seed_index = (
                    self._current_seed_index + self.num_group
                ) % total_seeds
            else:
                reset_state_ids = torch.randint(
                    low=10000,
                    high=200000,
                    size=(self.num_group,),
                    generator=self._generator,
                )
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
            for idx in env_idx:
                self.reset_state_ids[idx] = reset_state_ids[idx]
        else:
            if self.success_seeds is not None:
                total_seeds = self.success_seeds.numel()
                indices = (
                    torch.arange(self.num_group, device=self.success_seeds.device)
                    + self._current_seed_index
                ) % total_seeds
                reset_state_ids = self.success_seeds[indices]
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
                self._current_seed_index = (
                    self._current_seed_index + self.num_group
                ) % total_seeds
            else:
                reset_state_ids = torch.randint(
                    low=10000,
                    high=200000,
                    size=(self.num_group,),
                    generator=self._generator,
                )
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
            self.reset_state_ids = reset_state_ids

    def check_seeds(self, seeds):
        resutls = self.venv.check_seeds(seeds)

        return resutls
