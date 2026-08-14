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

"""LWD (DIVL + QAM) policy worker on the SAC / async SAC stack."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.algorithms.divl import (
    DIVLConfig,
    aggregate_q,
    categorical_support,
    divl_critic_loss,
    divl_critic_targets,
    divl_value_loss,
)
from rlinf.algorithms.qam import compute_qam_loss, sample_flow_time
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class EmbodiedLWDFSDPPolicy(EmbodiedSACFSDPPolicy):
    """DIVL (V→Q) + QAM flow policy updates without SAC α / intervention."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.vf_optimizer = None
        self.vf_lr_scheduler = None
        self.lwd_optimizer = None
        self._lwd_cfg = None
        self._support = None

    def init_worker(self):
        self.setup_model_and_optimizer(initialize_target=True)
        self.setup_lwd_components()
        self.setup_sac_components()
        self.soft_update_target_model(tau=1.0)
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        """Wrap model and build V / Q / policy optimizers (no entropy α)."""
        module = self.model_provider_func()
        if initialize_target:
            target_module = self.model_provider_func()

        if self.cfg.actor.model.get("gradient_checkpointing", False):
            self.logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
            if initialize_target:
                target_module.gradient_checkpointing_enable()
        else:
            self.logger.info("[FSDP] Gradient checkpointing is disabled")

        from rlinf.utils.utils import collect_param_names_need_sync

        self.param_names_need_sync = collect_param_names_need_sync(module)

        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                model=target_module, device_mesh=self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True

        self.use_dsrl = False
        # Policy (action expert) | Critic (Q) | Value (V)
        param_filters = {
            "critic": [
                "lwd_critic_image_encoder",
                "lwd_critic_state_encoder",
                "lwd_action_pool",
                "lwd_q_head",
            ],
            "value": [
                "lwd_value_image_encoder",
                "lwd_value_state_encoder",
                "lwd_v_head",
            ],
        }
        filtered_optim_config = {
            "critic": self.cfg.actor.critic_optim,
            "value": self.cfg.actor.get("value_optim", self.cfg.actor.critic_optim),
        }
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=param_filters,
            filtered_optim_config=filtered_optim_config,
        )
        # build_optimizers returns [main, critic, value] in filter order
        self.optimizer = optimizers[0]
        critic_params = optimizers[1].param_groups[0]["params"]
        value_params = optimizers[2].param_groups[0]["params"]

        value_optim_cfg = self.cfg.actor.get(
            "value_optim", self.cfg.actor.critic_optim
        )
        critic_optim_cfg = self.cfg.actor.critic_optim
        from rlinf.hybrid_engines.fsdp.fsdp_model_manager import warmup_optimizer_state

        self.lwd_optimizer = torch.optim.Adam(
            [
                {
                    "params": value_params,
                    "lr": value_optim_cfg.lr,
                    "betas": (
                        value_optim_cfg.get("adam_beta1", 0.9),
                        value_optim_cfg.get("adam_beta2", 0.999),
                    ),
                    "eps": value_optim_cfg.get("adam_eps", 1e-8),
                },
                {
                    "params": critic_params,
                    "lr": critic_optim_cfg.lr,
                    "betas": (
                        critic_optim_cfg.get("adam_beta1", 0.9),
                        critic_optim_cfg.get("adam_beta2", 0.999),
                    ),
                    "eps": critic_optim_cfg.get("adam_eps", 1e-8),
                },
            ]
        )
        warmup_optimizer_state(self.lwd_optimizer)
        # Single Adam step for V+Q avoids FSDP orig-param writeback corruption
        # from back-to-back subset optimizer steps (use_orig_params=True).
        self.vf_optimizer = self.lwd_optimizer
        self.qf_optimizer = self.lwd_optimizer

        self.entropy_temp = None
        self.alpha_optimizer = None

        self.build_lr_schedulers()
        gs_cfg = self.cfg.actor.fsdp_config.get("grad_scaler", {})
        enabled = bool(gs_cfg.get("enabled", False)) if gs_cfg is not None else False
        self.grad_scaler = self.build_grad_scaler(enabled)

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        value_optim_cfg = self.cfg.actor.get("value_optim", self.cfg.actor.critic_optim)
        self.vf_lr_scheduler = self.build_lr_scheduler(
            self.lwd_optimizer, value_optim_cfg
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.lwd_optimizer, self.cfg.actor.critic_optim
        )

    def setup_lwd_components(self):
        lwd = self.cfg.algorithm.get("lwd", {})
        openpi_cfg = self.cfg.actor.model.get("openpi", {})
        num_chunks = int(self.cfg.actor.model.get("num_action_chunks", 50))
        self._lwd_cfg = DIVLConfig(
            num_atoms=int(lwd.get("num_atoms", openpi_cfg.get("lwd_num_atoms", 51))),
            v_min=float(lwd.get("v_min", openpi_cfg.get("lwd_v_min", 0.0))),
            v_max=float(lwd.get("v_max", openpi_cfg.get("lwd_v_max", 1.0))),
            tau_base=float(lwd.get("tau_base", 0.5)),
            tau_min=float(lwd.get("tau_min", 0.1)),
            tau_max=float(lwd.get("tau_max", 0.9)),
            entropy_alpha=float(lwd.get("alpha", lwd.get("entropy_alpha", 1.0))),
            gamma=float(self.cfg.algorithm.get("gamma", 0.999)),
            num_action_chunks=num_chunks,
            agg_q=str(lwd.get("agg_q", openpi_cfg.get("lwd_agg_q", "min"))),
        )
        self.qam_lambda = float(lwd.get("qam_lambda", 1.0))
        self.lwd_stage = str(self.cfg.runner.get("lwd_stage", "online"))
        self.finetune_backbone = bool(
            lwd.get(
                "finetune_backbone_offline",
                self.lwd_stage == "offline",
            )
        )
        self._support = categorical_support(
            self._lwd_cfg.num_atoms,
            self._lwd_cfg.v_min,
            self._lwd_cfg.v_max,
            device=self.device,
            dtype=self.torch_dtype or torch.float32,
        )

    def _maybe_freeze_backbone(self):
        """Online LWD: freeze VLM backbone; offline may finetune."""
        if self.finetune_backbone and self.lwd_stage == "offline":
            return
        for name, param in self.model.named_parameters():
            if any(
                key in name
                for key in (
                    "lwd_",
                    "action_in_proj",
                    "action_out_proj",
                    "action_time_mlp",
                    "time_mlp",
                    "state_proj",
                    "gemma_expert",
                )
            ):
                continue
            if "paligemma" in name or "vision" in name or "language_model" in name:
                param.requires_grad_(False)

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """Receive online rollouts into B_on only (no intervention → demo)."""
        from rlinf.utils.metric_utils import compute_split_num
        from rlinf.utils.utils import clear_memory
        from rlinf.data.embodied_io_struct import Trajectory

        clear_memory(sync=False)
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.replay_buffer.add_trajectories(recv_list)
        # Intentionally do NOT append intervene trajs to demo_buffer (LWD).

    def forward_divl_value(self, batch):
        curr_obs = batch["curr_obs"]
        actions = batch["actions"]
        kwargs = {"train": True}
        with torch.no_grad():
            q_tgt = self.target_model(
                forward_type=ForwardType.DIVL_CRITIC,
                obs=curr_obs,
                actions=actions,
                **kwargs,
            )
            target_q = aggregate_q(q_tgt, self._lwd_cfg.agg_q)
        v_logits = self.model(
            forward_type=ForwardType.DIVL_VALUE, obs=curr_obs, **kwargs
        )
        support = self._support.to(device=v_logits.device, dtype=v_logits.dtype)
        loss, metrics = divl_value_loss(
            v_logits,
            target_q.to(dtype=v_logits.dtype),
            support=support,
            v_min=self._lwd_cfg.v_min,
            v_max=self._lwd_cfg.v_max,
        )
        return loss, metrics

    def forward_divl_critic(self, batch):
        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminations = batch["terminations"]
        kwargs = {"train": True}

        with torch.no_grad():
            next_v = self.model(
                forward_type=ForwardType.DIVL_VALUE, obs=next_obs, **kwargs
            )
            support = self._support.to(device=next_v.device, dtype=next_v.dtype)
            y_q, tgt_metrics = divl_critic_targets(
                next_v,
                rewards.to(dtype=next_v.dtype),
                terminations,
                support=support,
                cfg=self._lwd_cfg,
            )

        q_values = self.model(
            forward_type=ForwardType.DIVL_CRITIC,
            obs=curr_obs,
            actions=actions,
            **kwargs,
        )
        loss, metrics = divl_critic_loss(
            q_values, y_q.to(dtype=q_values.dtype)
        )
        metrics.update(tgt_metrics)
        return loss, metrics

    def _inner_policy_module(self):
        """One level below root FSDP; keep nested LWD FSDP wraps intact."""
        module = self.model
        if hasattr(module, "_fsdp_wrapped_module"):
            module = module._fsdp_wrapped_module
        return module

    def _sync_fsdp_param_views(self) -> None:
        """Rebind FSDP flat-param views after subset optimizer steps.

        With ``use_orig_params=True``, sequential V/Q optimizers can leave
        orig-param writeback metadata stale and break the next forward.
        """
        if not self.cfg.actor.fsdp_config.get("use_orig_params", False):
            return
        rebind = getattr(self._strategy, "_rebind_handle_views", None)
        iter_handles = getattr(self._strategy, "_iter_fsdp_handles", None)
        if rebind is None or iter_handles is None:
            return
        for handle in iter_handles(self.model):
            rebind(handle)

    def _inner_target_module(self):
        module = self.target_model
        if hasattr(module, "_fsdp_wrapped_module"):
            module = module._fsdp_wrapped_module
        return module

    def forward_qam(self, batch):
        curr_obs = batch["curr_obs"]
        policy = self._inner_policy_module()
        actions = batch["actions"].to(dtype=torch.float32).detach()
        actions = policy._reshape_actions_for_flow(actions)
        actions = actions.requires_grad_(True)

        freeze_backbone = not (
            self.finetune_backbone and self.lwd_stage == "offline"
        )
        # Q(s,a) for terminal adjoint: call LWD critic on the unwrapped module to
        # avoid FSDP root writeback when ``actions`` requires grad (QAM adjoint).
        q_values = policy.divl_critic_forward(
            obs=curr_obs,
            actions=actions,
            train=True,
            detach_encoder=True,
        )
        q_agg = aggregate_q(q_values, self._lwd_cfg.agg_q)

        time = sample_flow_time(
            actions.shape[0], device=actions.device, dtype=actions.dtype
        )
        online_out = policy(
            forward_type=ForwardType.QAM_POLICY,
            obs=curr_obs,
            actions=actions.detach(),
            time=time,
            freeze_backbone=freeze_backbone,
        )
        with torch.no_grad():
            ref_out = self._inner_target_module()(
                forward_type=ForwardType.QAM_POLICY,
                obs=curr_obs,
                actions=actions.detach(),
                noise=online_out["noise"],
                time=time,
                freeze_backbone=True,
            )
        loss, metrics = compute_qam_loss(
            actions=actions,
            noise=online_out["noise"],
            time=time,
            u_online=online_out["u_pred"],
            u_reference=ref_out["u_pred"],
            q_values=q_agg,
            qam_lambda=self.qam_lambda,
        )
        return loss, metrics

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, train_actor: bool = True):
        """Ordered updates: V → Q → QAM (no α)."""
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )

        for i, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[i] = batch

        metrics_data = {}

        # V → Q → QAM: accumulate all backwards first, then optimizer steps.
        # Avoids FSDP orig-param writeback errors from forward after subset step().
        self.lwd_optimizer.zero_grad()
        if train_actor:
            self.optimizer.zero_grad()

        gbs_v_loss = []
        gbs_q_loss = []
        gbs_qam_loss = []
        all_v_metrics = {}
        all_q_metrics = {}
        all_qam_metrics = {}

        for batch in train_micro_batch_list:
            v_loss, v_metrics = self.forward_divl_value(batch)
            v_loss = v_loss / self.gradient_accumulation
            v_loss.backward()
            gbs_v_loss.append(v_loss.item() * self.gradient_accumulation)
            append_to_dict(all_v_metrics, v_metrics)

            q_loss, q_metrics = self.forward_divl_critic(batch)
            q_loss = q_loss / self.gradient_accumulation
            q_loss.backward()
            gbs_q_loss.append(q_loss.item() * self.gradient_accumulation)
            append_to_dict(all_q_metrics, q_metrics)

            if train_actor:
                qam_loss, qam_metrics = self.forward_qam(batch)
                qam_loss = qam_loss / self.gradient_accumulation
                qam_loss.backward()
                gbs_qam_loss.append(qam_loss.item() * self.gradient_accumulation)
                append_to_dict(all_qam_metrics, qam_metrics)

        value_clip = self.cfg.actor.get(
            "value_optim", self.cfg.actor.critic_optim
        ).get("clip_grad", 10.0)
        lwd_grad_norm = self.model.clip_grad_norm_(
            max_norm=max(value_clip, self.cfg.actor.critic_optim.clip_grad)
        )
        self.lwd_optimizer.step()
        self.vf_lr_scheduler.step()
        self._sync_fsdp_param_views()
        metrics_data.update(
            {
                "lwd/value_loss": np.mean(gbs_v_loss),
                "lwd/critic_loss": np.mean(gbs_q_loss),
                "value/grad_norm": float(lwd_grad_norm),
                "critic/grad_norm": float(lwd_grad_norm),
                "critic/lr": self.lwd_optimizer.param_groups[1]["lr"],
                **{f"value/{k}": np.mean(v) for k, v in all_v_metrics.items()},
                **{f"critic/{k}": np.mean(v) for k, v in all_q_metrics.items()},
            }
        )

        if train_actor:
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()
            self._sync_fsdp_param_views()
            metrics_data.update(
                {
                    "lwd/qam_loss": np.mean(gbs_qam_loss),
                    "actor/grad_norm": float(actor_grad_norm),
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    **{f"qam/{k}": np.mean(v) for k, v in all_qam_metrics.items()},
                }
            )

        if (
            self.target_model_initialized
            and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0
        ):
            self.soft_update_target_model()

        return metrics_data
