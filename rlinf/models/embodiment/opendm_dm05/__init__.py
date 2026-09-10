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

import os

from omegaconf import DictConfig, OmegaConf


def _opendm_cfg_as_dict(cfg: DictConfig) -> dict:
    opendm_cfg = cfg.get("opendm")
    if opendm_cfg is None:
        return {}
    if isinstance(opendm_cfg, dict):
        return opendm_cfg
    return OmegaConf.to_container(opendm_cfg, resolve=True)


def get_model(cfg: DictConfig, torch_dtype=None):
    import torch
    from transformers import AutoProcessor

    from opendm.constants.robot import ROBOT_STATE_DESCS, ActionMode, RobotType
    from opendm.data.augmentations import NoAugmentationPipeline
    from opendm.data.transforms import (
        ActionAbsolute,
        ChatTokenization,
        Denormalize,
        Normalize,
        PixelTransform,
    )
    from opendm.model.dm05.dm05_arch import DM05Config

    from rlinf.models.embodiment.opendm_dm05.dm05_policy import (
        DM05ForRLActionPrediction,
    )
    from rlinf.utils.logging import get_logger

    logger = get_logger()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    if not cfg.model_path or not os.path.exists(cfg.model_path):
        raise ValueError(f"Model path does not exist: {cfg.model_path}")

    opendm_cfg = _opendm_cfg_as_dict(cfg)
    target_dtype = torch_dtype
    if target_dtype is None:
        target_dtype = (
            torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float32
        )

    try:
        config = DM05Config.from_pretrained(cfg.model_path)
        config.num_steps = cfg.get("num_steps", 5)
        config.action_env_dim = cfg.action_dim
        config.add_value_head = cfg.get("add_value_head", True)
        config.noise_level = opendm_cfg.get("noise_level", 0.5)
        config.noise_method = opendm_cfg.get("noise_method", "flow_sde")
        config.detach_critic_input = opendm_cfg.get("detach_critic_input", True)
        config.train_expert_only = opendm_cfg.get("train_expert_only", True)
        config.output_action_chunks = cfg.num_action_chunks
        config.safe_get_logprob = cfg.get("safe_get_logprob", False)
        config.chunk_critic_input = cfg.get("chunk_critic_input", True)
        config.noise_anneal = cfg.get("noise_anneal", False)
        config.joint_logprob = cfg.get("joint_logprob", False)
        config.value_after_vlm = cfg.get("value_after_vlm", False)
        config.num_images_in_input = opendm_cfg.get("num_images_in_input", 3)
        config.image_prompts = list(
            opendm_cfg.get(
                "image_prompts",
                ["Head", "Left wrist", "Right wrist"],
            )
        )
        config.robot_type = opendm_cfg.get(
            "robot_type", RobotType.ALOHA_ROBOTWIN2.value
        )
        config.add_state = opendm_cfg.get("add_state", True)
        config.n_bins = opendm_cfg.get("n_bins", 256)
        config.max_length = opendm_cfg.get("max_length", 1024)
        config.action_mode = opendm_cfg.get("action_mode", ActionMode.ABSOLUTE.value)
        config.default_speed = str(opendm_cfg.get("default_speed", "0.5"))
        config.default_control_mode = opendm_cfg.get("default_control_mode")

        def _state_desc_values(descs) -> list[str]:
            # HF PretrainedConfig JSON-dumps the config during init; enums are not serializable.
            return [desc.value if hasattr(desc, "value") else str(desc) for desc in descs]

        try:
            robot_enum = RobotType(config.robot_type)
            config.state_desc = _state_desc_values(ROBOT_STATE_DESCS[robot_enum])
        except (ValueError, KeyError):
            config.state_desc = _state_desc_values(
                ROBOT_STATE_DESCS[RobotType.ALOHA_ROBOTWIN2]
            )

        if hasattr(config, "vlm_config") and config.vlm_config is not None:
            if hasattr(config.vlm_config, "text_config"):
                config.vlm_config.text_config._attn_implementation = "sdpa"
            if hasattr(config.vlm_config, "vision_config"):
                config.vlm_config.vision_config._attn_implementation = "sdpa"
        if hasattr(config, "action_config") and config.action_config is not None:
            config.action_config._attn_implementation = "sdpa"

        original_offline = os.environ.get("HF_HUB_OFFLINE", None)
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            model = DM05ForRLActionPrediction.from_pretrained(
                cfg.model_path,
                config=config,
                torch_dtype=target_dtype,
                local_files_only=True,
            )
        finally:
            if original_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_offline

        model = model.to(dtype=target_dtype)
        model.set_attention_implementation(
            llm_attn_implementation="sdpa",
            vision_attn_implementation="sdpa",
            action_attn_implementation="sdpa",
            bf16=target_dtype == torch.bfloat16,
        )

        processor = AutoProcessor.from_pretrained(
            cfg.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model.processor = processor
        model.chat_tokenization = ChatTokenization(
            processor=processor,
            n_bins=config.n_bins,
            max_length=config.max_length,
            image_prompts=config.image_prompts,
            add_state=config.add_state,
            is_history=False,
            enable_logging=False,
        )
        model.pixel_transform = PixelTransform(
            transform_pipeline=NoAugmentationPipeline(),
        )
        model.image_prompts = config.image_prompts
        model.robot_type = config.robot_type
        model.state_desc = config.state_desc
        model.add_state = config.add_state
        model.default_speed = config.default_speed
        model.default_control_mode = config.default_control_mode

        norm_stats_file = os.path.join(cfg.model_path, "norm_stats.json")
        if os.path.exists(norm_stats_file):
            model.state_normalize = Normalize(
                norm_stats_path=norm_stats_file,
                norm_keys=["state"],
                use_quantiles=True,
            )
            model.action_denormalize = Denormalize(
                norm_stats_path=norm_stats_file,
                norm_keys=["action"],
                use_quantiles=True,
            )
        else:
            logger.warning(
                "norm_stats.json not found at %s; actions will not be normalized.",
                norm_stats_file,
            )
            model.state_normalize = None
            model.action_denormalize = None

        action_mode = str(config.action_mode).lower()
        if action_mode in {ActionMode.RELATIVE.value, "relative"}:
            model.action_absolute = ActionAbsolute()
        else:
            model.action_absolute = None

        model._train_expert_only = bool(config.train_expert_only)
        if model._train_expert_only:
            model.freeze_vlm()

    except Exception as e:
        logger.error(f"Failed to load pretrained DM05 model: {e}")
        raise

    return model
