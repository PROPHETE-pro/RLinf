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

import math
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers.cache_utils import DynamicCache

from opendm.constants.robot import HISTORY_PAD_TOKEN_ID
from opendm.model.dm05.dm05_arch import DM05ForConditionalGeneration
from opendm.model.dm05.dm05_utils import make_suffix_attn_mask

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.utils.logging import get_logger


def _to_uint8_hwc(image_np: np.ndarray) -> np.ndarray:
    image_np = np.asarray(image_np)
    if image_np.dtype != np.uint8:
        image_np = (
            (image_np * 255).astype(np.uint8)
            if image_np.max() <= 1.0
            else image_np.astype(np.uint8)
        )
    is_chw = (
        image_np.ndim == 3
        and image_np.shape[0] in (1, 3)
        and image_np.shape[-1] not in (1, 3)
    )
    if is_chw:
        image_np = np.transpose(image_np, (1, 2, 0))
    return image_np


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class DM05ForRLActionPrediction(BasePolicy, DM05ForConditionalGeneration):
    _no_split_names = [
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
    ]

    def __init__(self, config):
        DM05ForConditionalGeneration.__init__(self, config)
        self._no_split_modules = ["Gemma3DecoderLayer"]
        self.logger = get_logger()

        self.config = config
        self.num_steps = getattr(config, "num_steps", 5)
        self.action_horizon = config.chunk_size
        self.num_action_chunks = getattr(
            config, "output_action_chunks", config.chunk_size
        )
        self.action_dim = config.action_dim
        self.global_step = 0
        self.use_vlm_value = False
        self.value_head = nn.Linear(config.action_config.hidden_size, 1)
        self.value_head = self.value_head.to(
            dtype=self.model.action_out_proj.weight.dtype
        )

        self.processor = None
        self.pixel_transform = None
        self.chat_tokenization = None
        self.state_normalize = None
        self.action_denormalize = None
        self.action_absolute = None
        self.image_prompts = list(
            getattr(config, "image_prompts", ["Head", "Left wrist", "Right wrist"])
        )
        self.robot_type = getattr(config, "robot_type", "Aloha RoboTwin2")
        self.state_desc = getattr(config, "state_desc", None)
        self.add_state = getattr(config, "add_state", True)
        self.default_speed = str(getattr(config, "default_speed", "0.5"))
        self.default_control_mode = getattr(config, "default_control_mode", None)

    def freeze_vlm(self):
        if not getattr(self.config, "train_expert_only", False):
            self.logger.warning("freeze_vlm() called but train_expert_only is False")
            return
        vlm = getattr(self.model, "vlm", None)
        if vlm is None:
            self.logger.warning("freeze_vlm() found no VLM module")
            return
        vlm.eval()
        for param in vlm.parameters():
            param.requires_grad = False

    def _meta_data(self) -> dict:
        meta = {
            "robot_type": self.robot_type,
            "speed": self.default_speed,
        }
        if self.default_control_mode is not None:
            meta["control_mode"] = self.default_control_mode
        if self.state_desc is not None:
            meta["state_desc"] = self.state_desc
        return meta

    def _action_mask(self, batch_size: int, device, dtype) -> torch.Tensor:
        action_mask = torch.zeros(
            batch_size,
            1,
            self.config.action_dim,
            device=device,
            dtype=dtype,
        )
        action_mask[..., : self.config.action_env_dim] = 1.0
        return action_mask

    def obs_processor(self, env_obs):
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        state = env_obs["states"]
        if torch.is_tensor(state):
            state = state.to(dtype=torch.float32)
        processed_obs["observation/state"] = state
        if env_obs.get("wrist_images") is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        return processed_obs

    def input_transform(self, obs: dict, transpose=True):
        if self.state_normalize is not None and "observation/state" in obs:
            state_np = _as_numpy(obs["observation/state"]).astype(np.float32)
            if state_np.ndim == 1:
                state_np = state_np[None, ...]
            normalized = []
            for i in range(state_np.shape[0]):
                sample = self.state_normalize(
                    {"state": state_np[i], "meta_data": self._meta_data()}
                )
                normalized.append(np.asarray(sample["state"], dtype=np.float32))
            state_out = np.stack(normalized, axis=0)
            if torch.is_tensor(obs["observation/state"]):
                obs["observation/state"] = torch.from_numpy(state_out).to(
                    device=obs["observation/state"].device,
                    dtype=torch.float32,
                )
            else:
                obs["observation/state"] = state_out
            obs["states"] = obs["observation/state"]
        return obs

    def output_transform(self, outputs):
        if self.action_denormalize is None:
            self.logger.warning(
                "[output_transform] WARNING: action_denormalize is None! "
                "Actions will NOT be denormalized!"
            )
            return outputs

        state_batch = outputs.get("state", None)
        actions = outputs["actions"]
        batch_size = actions.shape[0]
        transformed_actions = []
        meta_data = self._meta_data()

        for i in range(batch_size):
            action_np = _as_numpy(actions[i])[..., : self.config.action_env_dim]
            sample = {"action": action_np, "meta_data": meta_data}
            if state_batch is not None:
                state_i = state_batch[i]
                sample["state"] = _as_numpy(state_i)
            sample = self.action_denormalize(sample)
            if self.action_absolute is not None:
                sample = self.action_absolute(sample)
            transformed_actions.append(torch.from_numpy(np.asarray(sample["action"])))

        outputs["actions"] = torch.stack(transformed_actions, dim=0).to(actions.device)
        outputs["actions"] = outputs["actions"][:, : self.num_action_chunks]
        return outputs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if torch.is_tensor(sub_value):
                        processed_obs[key][sub_key] = sub_value.to(
                            device=device
                        ).contiguous()
        return processed_obs

    def _tensor_to_pil(self, image) -> Image.Image:
        image_np = _to_uint8_hwc(_as_numpy(image))
        return Image.fromarray(image_np).convert("RGB")

    def _wrist_views(self, wrist_image) -> list[Image.Image]:
        wrist_np = _as_numpy(wrist_image)
        if wrist_np.ndim == 4:
            return [self._tensor_to_pil(view) for view in wrist_np]
        if wrist_np.ndim == 3:
            return [self._tensor_to_pil(wrist_np)]
        raise ValueError(
            f"Unexpected wrist image ndim={wrist_np.ndim}, shape={wrist_np.shape}"
        )

    def _collect_pil_images(self, main_image, wrist_image) -> list[Image.Image]:
        images = [self._tensor_to_pil(main_image)]
        if wrist_image is not None:
            images.extend(self._wrist_views(wrist_image))
        required = len(self.image_prompts)
        if len(images) < required:
            h, w = images[0].size[1], images[0].size[0]
            pad = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
            images.extend([pad] * (required - len(images)))
        return images[:required]

    def _prompts_as_list(self, prompts, batch_size: int) -> list[str]:
        if isinstance(prompts, str):
            return [prompts] * batch_size
        if isinstance(prompts, torch.Tensor):
            return [str(p) for p in prompts]
        return [str(p) for p in prompts]

    def _pad_and_stack_tokens(self, tokenized_list: list[dict]) -> dict:
        pad_token_id = self.processor.tokenizer.pad_token_id
        max_len = getattr(self.config, "max_length", 1024)
        seq_lens = [item["input_ids"].shape[1] for item in tokenized_list]
        pad_to = min(max(seq_lens), max_len)

        input_ids = []
        attention_mask = []
        token_type_ids = []
        pixel_values = []
        for item in tokenized_list:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            tti = item["token_type_ids"]
            seq_len = ids.shape[1]
            if seq_len > pad_to:
                ids = ids[:, :pad_to]
                mask = mask[:, :pad_to]
                tti = tti[:, :pad_to]
                seq_len = pad_to
            pad_len = pad_to - seq_len
            if pad_len > 0:
                ids = torch.cat(
                    [
                        ids,
                        torch.full((1, pad_len), pad_token_id, dtype=ids.dtype),
                    ],
                    dim=1,
                )
                mask = torch.cat(
                    [mask, torch.zeros((1, pad_len), dtype=mask.dtype)],
                    dim=1,
                )
                tti = torch.cat(
                    [tti, torch.zeros((1, pad_len), dtype=tti.dtype)],
                    dim=1,
                )
            input_ids.append(ids)
            attention_mask.append(mask)
            token_type_ids.append(tti)
            pixels = item["pixel_values"]
            if pixels.dim() == 3:
                pixels = pixels.unsqueeze(0)
            pixel_values.append(pixels)

        return {
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_mask": torch.cat(attention_mask, dim=0),
            "token_type_ids": torch.cat(token_type_ids, dim=0),
            "pixel_values": torch.cat(pixel_values, dim=0),
        }

    def _tokenize_observations(self, processed_obs: dict) -> dict:
        raw_main = processed_obs["observation/image"]
        raw_wrist = processed_obs.get("observation/wrist_image")
        states = _as_numpy(processed_obs["observation/state"])
        if states.ndim == 1:
            states = states[None, ...]
        batch_size = states.shape[0]
        prompts = self._prompts_as_list(processed_obs.get("prompt", ""), batch_size)
        meta_data = self._meta_data()

        tokenized_list = []
        for i in range(batch_size):
            wrist_i = None if raw_wrist is None else raw_wrist[i]
            images = self._collect_pil_images(raw_main[i], wrist_i)
            sample = {
                "images": images,
                "prompt": prompts[i],
                "state": states[i],
                "meta_data": meta_data,
            }
            if self.pixel_transform is not None:
                sample = self.pixel_transform(sample)
            tokenized_list.append(self.chat_tokenization(sample))
        return self._pad_and_stack_tokens(tokenized_list)

    def forward(self, forward_type="default_forward", **kwargs):
        if "forward_inputs" in kwargs and "data" not in kwargs:
            kwargs["data"] = kwargs.pop("forward_inputs")
        if forward_type == "default_forward":
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Forward type {forward_type} not implemented")

    def default_forward(self, data, **kwargs):
        compute_values = kwargs.get("compute_values", False)
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        if "input_ids" in data:
            tokenized = data
        else:
            observation = self.input_transform(data, transpose=False)
            tokenized = self._tokenize_observations(observation)

        device = chains.device
        target_dtype = next(self.parameters()).dtype
        log_probs, value_t, entropy = self.get_log_prob_value(
            tokenized["input_ids"].to(device),
            tokenized["attention_mask"].to(device),
            tokenized["pixel_values"].to(device=device, dtype=target_dtype),
            tokenized["token_type_ids"].to(device),
            data.get("observation/state"),
            chains.to(device=device, dtype=target_dtype),
            denoise_inds.to(device),
            compute_values,
        )
        log_probs = log_probs[
            :, :, : self.num_action_chunks, : self.config.action_env_dim
        ]
        entropy = entropy[:, :, : self.num_action_chunks, : self.config.action_env_dim]
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]
        value_t = value_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    def get_suffix_out(self, input_ids, kv_cache, prefix_len, x_t, timestep):
        batch_size = x_t.shape[0]
        device = x_t.device
        model_dtype = self.model.action_in_proj.weight.dtype
        x_t = x_t.to(dtype=model_dtype)
        x_t = x_t * self._action_mask(batch_size, device, model_dtype)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=device)
        if timestep.dim() == 0:
            timestep = timestep.broadcast_to(batch_size)
        timestep = timestep.to(dtype=model_dtype)

        suffix_embeds = self.model.action_in_proj(x_t)
        adarms_cond = self._build_adarms_cond(timestep, suffix_embeds.dtype)
        suffix_len = int(suffix_embeds.shape[1])
        invisible_prefix_token_ids = (HISTORY_PAD_TOKEN_ID,)
        suffix_attn_mask = make_suffix_attn_mask(
            input_ids=input_ids,
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            batch_size=batch_size,
            device=suffix_embeds.device,
            dtype=suffix_embeds.dtype,
            pad_token_id=self.model.vlm.model.language_model.padding_idx,
            invisible_prefix_token_ids=invisible_prefix_token_ids,
        )
        suffix_position_ids = self._build_suffix_position_ids(
            prefix_len,
            suffix_len,
            device,
            input_ids=input_ids,
            pad_token_id=self.model.vlm.model.language_model.padding_idx,
            invisible_prefix_token_ids=invisible_prefix_token_ids,
        )
        return self._suffix_forward(
            suffix_embeds=suffix_embeds,
            attention_mask=suffix_attn_mask,
            position_ids=suffix_position_ids,
            past_key_values=kv_cache,
            adarms_cond=adarms_cond,
        )

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        input_ids,
        kv_cache,
        prefix_len,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        bsize = x_t.shape[0]
        device = x_t.device
        if isinstance(idx, int):
            idx = torch.tensor(idx, device=device).expand(bsize)

        if getattr(self.config, "noise_anneal", False):
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = torch.tensor(
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps,
                device=device,
            )
        else:
            noise_level = torch.tensor(self.config.noise_level, device=device)

        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        suffix_out = self.get_suffix_out(
            input_ids, kv_cache, prefix_len, x_t, t_input
        )
        v_t = self.model.action_out_proj(
            suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)
        )

        if (
            self.config.add_value_head
            and compute_values
            and not getattr(self.config, "value_after_vlm", False)
        ):
            suffix_out_value = torch.mean(
                suffix_out[:, : self.config.chunk_size]
                if getattr(self.config, "chunk_critic_input", True)
                else suffix_out,
                dim=1,
                keepdim=False,
            )
            if getattr(self.config, "detach_critic_input", True):
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(
                suffix_out_value.to(self.value_head.weight.dtype)
            )[:, 0]
        else:
            value_t = torch.zeros(bsize, device=device)

        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x_t_mean = (1 - (t_input - delta)) * x0_pred + (t_input - delta) * x1_pred
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x_t_mean = (1 - (t_input - delta)) * x0_pred + (
                    t_input - delta - sigma_i**2 * delta / (2 * t_input)
                ) * x1_pred
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                cos_term = torch.cos(torch.pi * noise_level / 2).to(device)
                sin_term = torch.sin(torch.pi * noise_level / 2).to(device)
                x_t_mean = (1 - (t_input - delta)) * x0_pred + (
                    t_input - delta
                ) * cos_term * x1_pred
                x_t_std = (t_input - delta) * sin_term
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return x_t_mean, x_t_std, value_t

    def get_logprob_norm(self, sample, mu, sigma):
        if getattr(self.config, "safe_get_logprob", False):
            return -torch.pow((sample - mu), 2)
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        return torch.where(mask, torch.zeros_like(log_prob), log_prob)

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    @torch.no_grad()
    def sample_actions(
        self, processed_obs, noise=None, mode="train", compute_values=True
    ):
        tokenized = processed_obs
        if "input_ids" not in tokenized:
            tokenized = self._tokenize_observations(processed_obs)

        device = next(self.parameters()).device
        target_dtype = next(self.parameters()).dtype
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        pixel_values = tokenized["pixel_values"].to(device=device, dtype=target_dtype)
        token_type_ids = tokenized["token_type_ids"].to(device)
        batch_size = input_ids.shape[0]
        num_steps = self.num_steps

        no_grad_ctx = (
            torch.no_grad()
            if getattr(self.config, "train_expert_only", False)
            else torch.enable_grad()
        )
        with no_grad_ctx:
            kv_cache, prefix_len = self._compute_prefix_cache(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
                cache_cls=DynamicCache,
            )

        x_t = torch.randn(
            batch_size,
            self.config.chunk_size,
            self.config.action_dim,
            device=device,
            dtype=target_dtype,
        )
        x_t = x_t * self._action_mask(batch_size, device, target_dtype)

        chains = [x_t]
        log_probs = []
        values = []

        if getattr(self.config, "joint_logprob", False):
            log_probs.append(
                self.get_logprob_norm(x_t, torch.zeros_like(x_t), torch.ones_like(x_t))
            )

        if mode == "train":
            if getattr(self.config, "joint_logprob", False):
                denoise_inds = torch.arange(num_steps)
            elif getattr(self.config, "ignore_last", False):
                denoise_inds = torch.tensor(
                    [random.randint(0, num_steps - 2)] * num_steps
                )
            else:
                denoise_inds = torch.tensor(
                    [random.randint(0, num_steps - 1)] * num_steps
                )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(batch_size, 1).to(device)

        for idx in range(num_steps):
            sample_mode = "train" if idx == denoise_inds[0][idx] else "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                input_ids,
                kv_cache,
                prefix_len,
                sample_mode,
                num_steps,
                compute_values,
            )
            x_t = x_t_mean + torch.randn_like(x_t) * x_t_std
            x_t = x_t * self._action_mask(batch_size, device, x_t.dtype)
            log_probs.append(self.get_logprob_norm(x_t, x_t_mean, x_t_std))
            values.append(value_t)
            chains.append(x_t)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.num_action_chunks, : self.config.action_env_dim
        ]
        if getattr(self.config, "joint_logprob", False):
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0], device=device),
                denoise_inds[:, 0],
            ]

        if self.use_vlm_value:
            raise NotImplementedError("use_vlm_value is not supported for DM05")
        values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)

        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids,
        }

    def get_log_prob_value(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        token_type_ids,
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        bsize = chains.shape[0]
        no_grad_ctx = (
            torch.no_grad()
            if getattr(self.config, "train_expert_only", False)
            else torch.enable_grad()
        )
        with no_grad_ctx:
            kv_cache, prefix_len = self._compute_prefix_cache(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
                cache_cls=DynamicCache,
            )

        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        if getattr(self.config, "joint_logprob", False):
            num_steps = self.config.num_steps
            chains_log_probs.append(
                self.get_logprob_norm(
                    chains[:, 0],
                    torch.zeros_like(chains[:, 0]),
                    torch.ones_like(chains[:, 0]),
                )
            )
            chains_entropy.append(self.gaussian_entropy(torch.ones_like(chains[:, 0])))
        else:
            num_steps = 1

        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind].clone()
            chains_next = chains[torch.arange(bsize), denoise_ind + 1].clone()
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                input_ids,
                kv_cache,
                prefix_len,
                "train",
                self.config.num_steps,
                compute_values,
            )
            chains_log_probs.append(
                self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            )
            chains_entropy.append(self.gaussian_entropy(x_t_std))
            chains_values.append(value_t)

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)
        chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_entropy

    def predict_action_batch(self, env_obs, **kwargs):
        mode = kwargs.get("mode", "train")
        compute_values = kwargs.get("compute_values", True)
        to_process_obs = self.obs_processor(env_obs)
        raw_state = to_process_obs["observation/state"]
        if torch.is_tensor(raw_state):
            raw_state_np = raw_state.detach().cpu().float().numpy().copy()
        else:
            raw_state_np = np.array(raw_state, dtype=np.float32, copy=True)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        tokenized = self._tokenize_observations(processed_obs)
        processed_obs.update(tokenized)

        outputs = self.sample_actions(
            processed_obs=processed_obs, mode=mode, compute_values=compute_values
        )
        if self.action_denormalize is not None:
            outputs["state"] = raw_state_np
            outputs["meta_data"] = self._meta_data()
            outputs = self.output_transform(outputs)

        actions = outputs["actions"][:, :, : self.config.action_env_dim]
        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
            "pixel_values": outputs["pixel_values"],
            "token_type_ids": outputs["token_type_ids"],
        }
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)
        return actions, {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }
