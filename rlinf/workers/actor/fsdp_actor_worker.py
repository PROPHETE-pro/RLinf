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

import time
from functools import partial
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    prepare_pack_fsdp,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    all_reduce_dict,
    all_reduce_int,
    masked_normalization,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    CRITIC_EXPLAINED_VARIANCE_KEY,
    append_to_dict,
    compute_critic_explained_variance_from_stats,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
    pop_critic_explained_variance_stats,
)
from rlinf.utils.logging import log_progress, set_progress_state, start_progress_heartbeat
from rlinf.utils.robotwin_hang_diagnostics import configure_from_cfg
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    load_runner_ckpt_state_dict,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


def compute_rollout_train_kl(
    m_batch: dict, loss_mask: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Compute the masked mean of absolute difference between rollout and training logprobs.

    Args:
        m_batch: Dictionary containing 'rollout_logprobs' and 'recomputed_logprobs'.
        loss_mask: Mask tensor for computing weighted mean.

    Returns:
        Masked mean of abs(recomputed_logprobs - rollout_logprobs), or None if keys are missing.
    """
    if "rollout_logprobs" not in m_batch or "recomputed_logprobs" not in m_batch:
        return None
    rollout_logprobs = m_batch["rollout_logprobs"]
    recomputed_logprobs = m_batch["recomputed_logprobs"]
    kl = torch.abs(recomputed_logprobs - rollout_logprobs)
    return masked_mean(kl, loss_mask)


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        if cfg_fsdp is None:
            cfg_fsdp = cfg.actor
        Worker.__init__(self)
        super().__init__(cfg_fsdp, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            cfg.actor.model.encoder_seq_length - cfg.data.max_prompt_length
        )
        self.calculate_entropy = cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = cfg.algorithm.kl_beta
        self.kl_penalty_type = cfg.algorithm.kl_penalty_type
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0.0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

        self.total_batch_size_per_dp = (
            cfg.data.rollout_batch_size * cfg.algorithm.group_size // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(cfg.algorithm.loss_agg_func)
        self.enable_offload = not self.is_pipeline and cfg.actor.get(
            "enable_offload", False
        )
        self.micro_batch_size = cfg.actor.micro_batch_size
        self.n_mini_batches = cfg.algorithm.n_minibatches
        self.task_type = cfg.runner.task_type
        self.entropy_op_type = cfg.algorithm.get("entropy_op_type", "flash_attn")
        self.enable_dp_load_balance = cfg.actor.get("enable_dp_load_balance", False)
        self.lr_sched_sync_with_optim = cfg.actor.get("lr_sched_sync_with_optim", True)
        self.enable_dynamic_batch_size = cfg.runner.get(
            "enable_dynamic_batch_size", False
        )
        if self.is_pipeline:
            assert not self.enable_dp_load_balance, (
                "DP load balance is not supported in pipeline mode."
            )
            assert not self.enable_dynamic_batch_size, (
                "Dynamic batch size is not supported in pipeline mode."
            )
        self.max_tokens_per_mbs = cfg.runner.get("max_tokens_per_mbs", 2048)
        self.variable_seq_lengths = self.cfg.actor.model.get(
            "variable_seq_lengths", False
        )

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if (
            self.kl_beta > 0 or self.reinpp_kl_beta > 0
        ) and self.combine_reference_model:
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)
            self.offload_model_buffer = {}

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        pass

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    @Worker.timer("actor/sync_model_to_rollout")
    def sync_model_to_rollout(self):
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        rollout_dtype = None
        if self._cfg.get("sync_precision", None) is not None:
            rollout_dtype = torch_dtype_from_precision(self._cfg.sync_precision)

        rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        has_visual = any("visual." in k for k in rollout_state_dict.keys())
        model_bucket_list = self.divide_model_to_bucket(rollout_state_dict, has_visual)
        del rollout_state_dict
        send_handles = []
        buffer = {}
        for bucket_idx, model_bucket in enumerate(model_bucket_list):
            for k, v in model_bucket.items():
                if isinstance(v, DTensor):
                    v = v.full_tensor()
                if rollout_dtype is not None:
                    v = v.to(rollout_dtype)
                if not self.is_pipeline:
                    v = reduce_tensor(v)
                buffer[k] = v
            if bucket_idx == 0:
                buffer["bucket_length"] = len(model_bucket_list)

            for send_handle in send_handles:
                send_handle.wait()
            send_handles = []

            if not self.is_pipeline:
                send_handle = self.send(
                    buffer,
                    self._rollout_group_name,
                    self._weight_dst_rank_in_rollout,
                    async_op=True,
                )
                send_handles.append(send_handle)
            else:
                for rank in self._weight_dst_rank_in_rollout:
                    send_handle = self.send(
                        buffer,
                        self._rollout_group_name,
                        rank,
                        async_op=True,
                    )
                    send_handles.append(send_handle)
            buffer = {}

        for send_handle in send_handles:
            send_handle.wait()

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad()

        clear_memory(sync=False)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def get_dynamic_batch_as_much(
        self,
        input_channel: Channel,
        min_result_len: int,
        max_result_len: int,
        cliped_results=[],
        unfinished_result=None,
    ):
        assert not input_channel.is_local
        rollout_results = cliped_results
        # get min_result_len
        while len(rollout_results) < min_result_len:
            if unfinished_result is not None:
                rollout_result: RolloutResult = unfinished_result.wait()
                unfinished_result = None
            else:
                rollout_result: RolloutResult = input_channel.get()
            rollout_results.append(rollout_result)

        # try to get result as much
        # get result in every 0.1s and do all reduce to get the min result between dp (result_len)
        # stop at: the min result between dp (result_len) is same as the last min result
        last_result_len = 0
        result_len = len(rollout_results)
        time_until = time.time() + 0.1
        while last_result_len < result_len:
            if len(rollout_results) < max_result_len:
                if unfinished_result is None:
                    unfinished_result = input_channel.get(async_op=True)
                else:
                    time.sleep(0.001)
                if unfinished_result.done():
                    rollout_results.append(unfinished_result.wait())
                    unfinished_result = None
                if time.time() >= time_until:
                    last_result_len = result_len
                    result_len = all_reduce_int(len(rollout_results))
                    if last_result_len < result_len:
                        time_until = time.time() + 0.1
            else:
                last_result_len = result_len
                result_len = all_reduce_int(len(rollout_results))

        cliped_results = list(rollout_results[result_len:])
        rollout_results = rollout_results[:result_len]

        batches = []
        for rollout_result in rollout_results:
            batch = rollout_result.to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.cfg.actor.model.encoder_seq_length,
                self.tokenizer.eos_token_id,
            )
            batches.append(batch)

        batch = RolloutResult.merge_batches(batches)
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        return batch, rollout_result, result_len, cliped_results, unfinished_result

    @staticmethod
    def _split_to_micro_batch(
        batch,
        enable_dynamic_batch_size: bool,
        *,
        max_tokens_per_mbs: Optional[int] = None,
        split_num,
    ):
        if enable_dynamic_batch_size:
            (
                micro_batches_iter,
                _,
                micro_batch_cnt,
                dbs_indices,
            ) = split_dynamic_batch_size(
                batch=batch,
                cp_world_size=1,
                vpp_world_size=1,
                max_tokens_per_mbs=max_tokens_per_mbs,
                microbatch_group_size_per_vp_stage=1,
            )
        else:
            micro_batch_cnt = split_num
            micro_batches_iter = get_iterator_k_split(batch, micro_batch_cnt)
            dbs_indices = None
        return micro_batches_iter, micro_batch_cnt, dbs_indices

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    def compute_logprobs(self, logits, target):
        return compute_logprobs_from_logits(
            logits,
            target,
            op_type=self.entropy_op_type,
        )

    def forward_batch(
        self, m_batch: dict[str, torch.Tensor], calculate_entropy: bool = False
    ) -> torch.Tensor:
        input_ids = m_batch["input_ids"]
        attention_mask = m_batch["attention_mask"]
        position_ids = m_batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch.keys():
            for key in m_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                    dim=0,
                ).to(Worker.torch_device_type)

        if self.enable_dynamic_batch_size or self.variable_seq_lengths:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = self.cfg.actor.model.encoder_seq_length
            max_prompt_len = self.cfg.data.max_prompt_length
            max_response_len = max_seq_len_unpack - max_prompt_len
            idx_starts, idx_ends = prepare_pack_fsdp(m_batch, max_prompt_len)

            input_ids, position_ids, attention_mask = pack_fsdp_input(
                input_ids,
                position_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_to_fixed_len=not self.variable_seq_lengths,
            )

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )

        logits: torch.Tensor = outputs.logits

        logits.div_(self.cfg.algorithm.sampling_params.temperature)

        if self.enable_dynamic_batch_size or self.variable_seq_lengths:
            logprobs = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=max_seq_len_unpack,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
            logprobs = logprobs[:, -max_response_len:]
        else:
            # (bsz, response_length, vocab_size)
            logits = logits[:, -self.response_len - 1 : -1, :]
            responses = input_ids[:, -self.response_len :]
            logprobs = self.compute_logprobs(logits, responses)

        if calculate_entropy:
            entropy = compute_entropy_from_logits(logits)

            if self.enable_dynamic_batch_size or self.variable_seq_lengths:
                entropy = unpack_sequences(
                    entropy, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0
                )[:, -self.response_len :]

            return logprobs, entropy

        return logprobs

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        num_sequences: int,
        compute_ref_logprobs: bool,
    ):
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=num_sequences
            // self.cfg.algorithm.logprob_forward_micro_batch_size,
        )
        if self.enable_dynamic_batch_size:
            indices = sum(dbs_indices, [])
            revert_indices = torch.tensor(
                get_reverse_idx(indices),
                dtype=torch.long,
            )
        micro_batches = list(micro_batches_iter)

        recomputed_logprobs, ref_logprobs = None, None

        # Recompute logprobs
        recomputed_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()

        if self.enable_dynamic_batch_size:
            assert len(indices) == recomputed_logprobs.size(0), (
                f"Dynamic batch size indices length {len(indices)} does not equal "
                f"output length {recomputed_logprobs.size(0)}"
            )
            recomputed_logprobs = recomputed_logprobs[revert_indices]

        # Ref logprobs
        if compute_ref_logprobs:
            assert self.ref_policy_state_dict is not None, (
                "Reference policy state dict is None but compute_ref_logprobs is True"
            )
            with cpu_weight_swap(
                self.model,
                self.ref_policy_state_dict,
                self.offload_model_buffer,
            ):
                ref_logprobs = torch.cat(
                    [self.forward_batch(batch) for batch in micro_batches]
                ).cpu()

                if self.enable_dynamic_batch_size:
                    assert len(indices) == ref_logprobs.size(0), (
                        f"Dynamic batch size indices length {len(indices)} does not equal "
                        f"output length {ref_logprobs.size(0)}"
                    )
                    ref_logprobs = ref_logprobs[revert_indices]

        return recomputed_logprobs, ref_logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
        do_offload=False,
    ):
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
            do_offload: Whether offload weights after inference is done
        """
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        inference_split = self.cfg.actor.get("inference_split", None)
        if inference_split is None:
            if not self.is_pipeline:
                inference_split = 1
            else:
                inference_split = self.cfg.algorithm.n_minibatches
        assert self.total_batch_size_per_dp % inference_split == 0, (
            f"FSDPActor: total_batch_size_per_dp[{self.total_batch_size_per_dp}] should be divisible by inference_split[{inference_split}]"
        )

        min_result_len = 1
        max_result_len = (
            self.cfg.data.rollout_batch_size // self._world_size // inference_split
        )
        if not self.is_pipeline:
            min_result_len = max_result_len
            coll_rollout_results = []
        total_result_len = 0
        total_result_len_per_dp = self.cfg.data.rollout_batch_size // self._world_size
        cliped_results, unfinished_result = [], None
        while total_result_len < total_result_len_per_dp:
            batch, rollout_result, result_len, cliped_results, unfinished_result = (
                self.get_dynamic_batch_as_much(
                    input_channel,
                    min(min_result_len, total_result_len_per_dp - total_result_len),
                    min(max_result_len, total_result_len_per_dp - total_result_len),
                    cliped_results,
                    unfinished_result,
                )
            )
            total_result_len += result_len
            self.log_debug(
                f"[dynamic inference rank-{self._rank}] inference result_len={result_len}, total_result_len={total_result_len}/{total_result_len_per_dp}"
            )
            self._load_weight_and_optimizer()
            self.model.eval()

            with self.worker_timer():
                with torch.no_grad():
                    recomputed_logprobs, ref_logprobs = self.inference_step(
                        batch, rollout_result.num_sequence, compute_ref_logprobs
                    )

                rollout_result.recomputed_logprobs = recomputed_logprobs

                # Ref logprobs
                if compute_ref_logprobs:
                    rollout_result.ref_logprobs = ref_logprobs

            if self.is_pipeline:
                # for pipeline mode, send after inference to reduce latency.
                # should do split to ensure actor won't get too much batches.
                split_results = RolloutResult.split_results(rollout_result, result_len)
                for split_result in split_results:
                    output_channel.put(split_result, async_op=True)
            else:
                coll_rollout_results.append(rollout_result)

        if not self.is_pipeline:
            # for coll mode, merge results to reduce send time.
            rollout_result = RolloutResult.merge_result_list(coll_rollout_results)
            split_results = RolloutResult.split_results(
                rollout_result,
                min(total_result_len, self.cfg.algorithm.n_minibatches),
            )
            for split_result in split_results:
                output_channel.put(split_result)
        assert total_result_len == total_result_len_per_dp, (
            f"Expected {total_result_len_per_dp} sequences from channel, but got {total_result_len}"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batches_iter, micro_batch_cnt, _ = self._split_to_micro_batch(
                batch,
                self.enable_dynamic_batch_size,
                max_tokens_per_mbs=self.max_tokens_per_mbs,
                split_num=global_batch_size // self.micro_batch_size,
            )
            self.gradient_accumulation = micro_batch_cnt
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == micro_batch_cnt,
            )
            for k, v in m_batch.items():
                m_batch[k] = (
                    v.to(Worker.torch_device_type) if isinstance(v, torch.Tensor) else v
                )

            # batch for forward
            logprobs, entropy = self.forward_batch(m_batch, True)

            # batch for backward
            # Prefer recomputed_logprobs (from actor inference), fallback to rollout_logprobs
            old_logprobs = m_batch.get("recomputed_logprobs")
            if old_logprobs is None:
                old_logprobs = m_batch["rollout_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            if self.cfg.algorithm.get("importance_sampling_fix", False):
                if (
                    "rollout_logprobs" not in m_batch
                    or "recomputed_logprobs" not in m_batch
                ):
                    raise ValueError(
                        "importance_sampling_fix requires both rollout_logprobs and recomputed_logprobs"
                    )
                rollout_logprobs = m_batch["rollout_logprobs"]
                recomputed_logprobs = m_batch["recomputed_logprobs"]
                advantages = advantages * torch.clamp(
                    (recomputed_logprobs - rollout_logprobs).exp(),
                    max=self.cfg.algorithm.importance_sampling_clip,
                )

            loss, mbs_metrics_data = policy_loss(
                task_type=self.task_type,
                loss_type=self.cfg.algorithm.loss_type,
                loss_agg_func=self.loss_agg_func,
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                clip_ratio_c=clip_ratio_c,
                clip_ratio_low=clip_ratio_low,
                clip_ratio_high=clip_ratio_high,
                loss_mask=loss_mask,
                clip_log_ratio_min=self.cfg.algorithm.get("clip_log_ratio_min", None),
                clip_log_ratio_max=self.cfg.algorithm.get("clip_log_ratio_max", None),
                fast_path_zero_loss_mask=True,
            )

            entropy_loss = torch.tensor(
                0.0, device=Worker.torch_platform.current_device()
            )
            if self.calculate_entropy:
                entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                if self.calculate_entropy_loss:
                    loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

            kl_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
            if self.kl_beta > 0 and ref_logprobs is not None:
                kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                kl_loss = self.loss_agg_func(kld, loss_mask)
                loss = loss + kl_loss * self.kl_beta

            # add to log
            # scale loss for gradient accumulation and backprop
            final_loss_metric = loss.detach()
            loss = loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": final_loss_metric,
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()

        if self.lr_sched_sync_with_optim:
            self.lr_scheduler.step()

        # display the degree of mismatch between training and rollout
        rollout_train_kl = compute_rollout_train_kl(m_batch, loss_mask)

        # aggregate metrics across micro-batches
        explained_variance_stats = pop_critic_explained_variance_stats(mbs_metrics_list)
        mean_metric_dict = {
            key: torch.mean(torch.stack(value))
            for key, value in mbs_metrics_list.items()
        }
        if rollout_train_kl is not None:
            mean_metric_dict["actor/rollout_train_kl"] = rollout_train_kl

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        if explained_variance_stats:
            reduced_stats = all_reduce_dict(
                explained_variance_stats, op=torch.distributed.ReduceOp.SUM
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )

        mean_metric_dict["actor/grad_norm"] = float(grad_norm)
        mean_metric_dict["actor/lr"] = lr_list[0]
        return mean_metric_dict

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer("run_training"):
            for _ in range(self.n_mini_batches):
                mean_metric_dict = self.training_step(batch=train_batch_iterator)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def _dp_load_balance(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        assert batch_size == self.total_batch_size_per_dp, (
            f"DP Load balance is only available when a single batch contains all data, e.g., in collocated mode. But got {batch_size=} and {self.total_batch_size_per_dp=}."
        )
        batch = RolloutDataBalance.from_rollout_batches(
            rollout_batches=batch,
            dp_world_size=torch.distributed.get_world_size(),
            dp_rank=torch.distributed.get_rank(),
            dp_group=torch.distributed.group.WORLD,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        return batch

    def run_training(
        self, input_channel: Channel, do_offload=False
    ) -> tuple[dict, list]:
        # Get all batches for this DP
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        if self.is_pipeline:
            return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        assert (
            "recomputed_logprobs" in global_batch or "rollout_logprobs" in global_batch
        )

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.enable_dp_load_balance:
            global_batch = self._dp_load_balance(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                logprob = batch.get("recomputed_logprobs")
                if logprob is None:
                    logprob = batch.get("rollout_logprobs")
                logprob = logprob.to(Worker.torch_device_type)

                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].to(Worker.torch_device_type),
                    loss_mask=mask.to(Worker.torch_device_type),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=logprob,
                    ref_logprob=batch["ref_logprobs"].to(Worker.torch_device_type)
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages
        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        configure_from_cfg(self.cfg)
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0
        self._init_sft_co_train_options()
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

        # create weight syncer
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer")
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        self.update_epoch = self.cfg.algorithm.get("update_epoch", 1)

        self._sync_weight_comm_options = self.weight_syncer.comm_options

        self._is_weight_sender = self._rank == 0
        self._actor_world_size = self._world_size
        self._rollout_all_ranks = list(
            range(self._component_placement.get_world_size("rollout"))
        )

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            keep_critic_weight = bool(
                self.cfg.runner.get("keep_critic_weight", False)
            )
            model_dict = load_runner_ckpt_state_dict(
                self.cfg.runner.ckpt_path,
                keep_critic_weight=keep_critic_weight,
            )
            # strict=False when critic keys are dropped so random value_head remains.
            model.load_state_dict(model_dict, strict=keep_critic_weight)

        return model

    def get_rollout_state_dict(self) -> dict:
        return self.get_model_state_dict(cpu_offload=False, full_state_dict=False)

    @Worker.timer("actor/sync_model_to_rollout")
    async def sync_model_to_rollout(self) -> None:
        log_progress(
            "Actor",
            f"rank={self._rank} sync_model_to_rollout: start",
            rank=self._rank,
        )
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            if not self._is_weight_sender:
                return
            await self.broadcast(
                data,
                groups=[
                    (self._group_name, 0),
                    (self._rollout_group_name, self._rollout_all_ranks),
                ],
                src=(self._group_name, 0),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def recv_func():
            return await self.recv(
                src_group_name=self._rollout_group_name,
                src_rank=0,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        if not self.weight_syncer.sender_initialized():
            log_progress(
                "Actor",
                f"rank={self._rank} sync_model_to_rollout: init_sender",
                rank=self._rank,
            )
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
                param_names_need_sync=self.param_names_need_sync,
                is_sender=self._is_weight_sender,
            )

        version = (
            self.get_rollout_sync_version()
            if hasattr(self, "get_rollout_sync_version")
            else self.version
        )
        log_progress(
            "Actor",
            f"rank={self._rank} sync_model_to_rollout: sync(version={version})",
            rank=self._rank,
        )
        await self.weight_syncer.sync(state_dict, send_func, version=version)

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad(True)
        log_progress(
            "Actor",
            f"rank={self._rank} sync_model_to_rollout: done",
            rank=self._rank,
        )

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        start_progress_heartbeat(component="Actor", rank=self._rank)
        set_progress_state(
            "Actor",
            f"recv_rollout_trajectories: start "
            f"(expect {split_num} chunks, send_num={send_num}, recv_num={recv_num})",
            rank=self._rank,
            all_ranks=True,
        )
        recv_list = []
        for i in range(split_num):
            set_progress_state(
                "Actor",
                f"recv_rollout_trajectories: waiting chunk {i + 1}/{split_num}",
                rank=self._rank,
                all_ranks=True,
            )
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)
            log_progress(
                "Actor",
                f"rank={self._rank} recv_rollout_trajectories: "
                f"got chunk {i + 1}/{split_num}",
                rank=self._rank,
            )

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)
        log_progress(
            "Actor",
            f"rank={self._rank} recv_rollout_trajectories: done",
            rank=self._rank,
            all_ranks=True,
        )

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.env.train.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        log_progress(
            "Actor",
            f"rank={self._rank} compute_advantages_and_returns: start",
            rank=self._rank,
        )
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        log_progress(
            "Actor",
            f"rank={self._rank} compute_advantages_and_returns: done",
            rank=self._rank,
        )
        return rollout_metrics

    def _init_sft_co_train_options(self) -> None:
        """Optional co-train enhancements (see actor.sft_co_train in yaml)."""
        sft_cfg = self.cfg.actor.get("sft_co_train", {})
        if sft_cfg is None:
            sft_cfg = {}
        self.sft_dynamic_loss_norm = bool(sft_cfg.get("dynamic_loss_norm", False))
        self.sft_target_loss_ratio = float(sft_cfg.get("target_loss_ratio", 0.5))
        self.sft_only_global_steps = int(sft_cfg.get("sft_only_global_steps", 0))
        self.sft_rollout_success = bool(sft_cfg.get("rollout_success_sft", False))
        self.sft_rollout_success_ratio = float(
            sft_cfg.get("rollout_success_sft_ratio", 1.0)
        )
        self.sft_offline_ratio = float(sft_cfg.get("offline_sft_ratio", 1.0))
        if self.sft_only_global_steps > 0 and self.critic_warmup_steps > 0:
            self.log_on_first_rank(
                "Disabling critic_warmup_steps because sft_only_global_steps "
                f"({self.sft_only_global_steps}) requires policy updates via SFT."
            )
            self.critic_warmup_steps = 0

    def _in_sft_only_phase(self) -> bool:
        return (
            self.enable_sft_co_train
            and self.sft_only_global_steps > 0
            and self.version < self.sft_only_global_steps
        )

    def _in_critic_warmup_phase(self) -> bool:
        return (
            self.critic_warmup_steps > 0
            and self.optimizer_steps < self.critic_warmup_steps
        )

    def _micro_batch_has_rollout_success(
        self, micro_batch: dict[str, torch.Tensor]
    ) -> bool:
        rewards = micro_batch.get("rewards")
        if rewards is None:
            return False
        if rewards.dim() >= 2:
            return bool(rewards.reshape(rewards.shape[0], -1).amax(dim=-1).any())
        return bool((rewards > 0).any())

    def _global_rollout_success_active(
        self, micro_batch: dict[str, torch.Tensor]
    ) -> bool:
        """True if any actor rank has rollout success in this micro-batch."""
        local_flag = int(self._micro_batch_has_rollout_success(micro_batch))
        if self._world_size <= 1:
            return bool(local_flag)
        global_flag = all_reduce_int(
            local_flag, op=torch.distributed.ReduceOp.MAX
        )
        return global_flag > 0

    def _compute_rollout_success_sft_loss(
        self, micro_batch: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        forward_inputs = micro_batch.get("forward_inputs")
        rewards = micro_batch.get("rewards")
        if forward_inputs is None or rewards is None:
            return None

        if rewards.dim() >= 2:
            success_mask = rewards.reshape(rewards.shape[0], -1).amax(dim=-1) > 0
        else:
            success_mask = rewards > 0

        if not success_mask.any():
            return None

        max_sft_bs = int(self.cfg.actor.get("sft_co_train", {}).get("max_success_batch", 0) or 0)
        if max_sft_bs > 0:
            success_idx = success_mask.nonzero(as_tuple=False).flatten()
            success_mask = torch.zeros_like(success_mask)
            success_mask[success_idx[:max_sft_bs]] = True

        filtered: dict[str, torch.Tensor] = {}
        batch_size = success_mask.shape[0]
        for key, value in forward_inputs.items():
            if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                filtered[key] = value[success_mask]

        if not filtered:
            return None

        if not hasattr(self.model, "prepare_dagger_sft_batch"):
            return None

        data = self.model.prepare_dagger_sft_batch(filtered)
        return self.model(
            data=data,
            forward_type=ForwardType.SFT,
            use_action_chunk_loss=True,
        )

    def _compute_zero_rollout_success_sft_loss(
        self, micro_batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Run a full SFT forward with zero grad for FSDP rank sync."""
        forward_inputs = micro_batch.get("forward_inputs")
        if forward_inputs is not None and hasattr(
            self.model, "prepare_dagger_sft_batch"
        ):
            filtered: dict[str, torch.Tensor] = {}
            for key, value in forward_inputs.items():
                if isinstance(value, torch.Tensor) and value.shape[0] > 0:
                    filtered[key] = value[:1]
            if filtered:
                data = self.model.prepare_dagger_sft_batch(filtered)
                loss = self.model(
                    data=data,
                    forward_type=ForwardType.SFT,
                    use_action_chunk_loss=True,
                )
                return loss * 0.0

        for param in self.model.parameters():
            if param.requires_grad:
                return param.reshape(-1)[0] * 0.0
        device = Worker.torch_platform.current_device()
        return torch.zeros((), device=device, requires_grad=True)

    def _compute_rollout_success_sft_loss_for_backward(
        self, micro_batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        loss = self._compute_rollout_success_sft_loss(micro_batch)
        if loss is not None:
            return loss
        return self._compute_zero_rollout_success_sft_loss(micro_batch)

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            repo_id = resolve_lerobot_repo_id(self.cfg.actor.get("sft_data_path"))
            if repo_id is None:
                raise ValueError(
                    "actor.sft_data_path must be set to a local dataset path or "
                    "LeRobot repo id when enable_sft_co_train=True."
                )

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            if "config_name" not in self.cfg.actor:
                raise ValueError(
                    "config_name is required when enable_sft_co_train=True"
                )
            training_config_name = self.cfg.actor.config_name
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _enable_policy_grad_for_sft(self) -> list[str]:
        """Re-enable policy grads when value-head warmup froze them but SFT must train."""
        enabled: list[str] = []
        for name, param in self.model.named_parameters():
            if "value_head" in name or "model.value_head" in name:
                continue
            if not param.requires_grad:
                param.requires_grad = True
                enabled.append(name)
        return enabled

    def _restore_policy_grad_after_sft(self, enabled_names: list[str]) -> None:
        if not self._in_critic_warmup_phase():
            return
        for name, param in self.model.named_parameters():
            if name in enabled_names:
                param.requires_grad = False

    def _skip_sft_during_critic_warmup(self, metrics_data: dict) -> bool:
        """Skip SFT co-train updates while only the critic is training."""
        if not self._in_critic_warmup_phase():
            return False
        metrics_data["sft_loss"] = 0.0
        metrics_data["sft_co_train/scaled_sft"] = 0.0
        metrics_data["sft_co_train/skipped_critic_warmup"] = 1.0
        if self.sft_rollout_success:
            metrics_data["sft_loss/rollout_success"] = 0.0
        return True

    def _scale_single_sft_term(
        self,
        ppo_loss_value: float,
        sft_loss: torch.Tensor,
        weight: float,
    ) -> torch.Tensor:
        if self._in_sft_only_phase() and self.sft_dynamic_loss_norm:
            sft_mag = torch.abs(sft_loss.detach()).clamp(min=1e-8)
            scale = self.sft_target_loss_ratio / sft_mag
            return self.sft_loss_weight * weight * sft_loss * scale

        if self.sft_dynamic_loss_norm:
            ppo_mag = max(abs(ppo_loss_value), 1e-8)
            sft_mag = torch.abs(sft_loss.detach()).clamp(min=1e-8)
            scale = (self.sft_target_loss_ratio * ppo_mag) / sft_mag
            return self.sft_loss_weight * weight * sft_loss * scale

        return self.sft_loss_weight * weight * sft_loss

    def _backward_sft_co_train_terms(
        self,
        metrics_data: dict,
        ppo_loss_value: float,
        micro_batch: dict[str, torch.Tensor] | None,
        grad_scale_divisor: float,
    ) -> None:
        """Run each SFT term as an immediate forward/backward pair.

        Batching multiple checkpointed forwards before backward triggers
        torch.utils.checkpoint.CheckpointError when co-training with PPO.
        """
        if self._skip_sft_during_critic_warmup(metrics_data):
            return

        need_policy_grad = self._in_sft_only_phase()
        enabled_grad_names: list[str] = []
        if need_policy_grad:
            enabled_grad_names = self._enable_policy_grad_for_sft()

        planned_terms: list[tuple[str, float]] = []
        if self.sft_offline_ratio > 0:
            planned_terms.append(("offline", self.sft_offline_ratio))
        global_rollout_success = (
            self.sft_rollout_success
            and micro_batch is not None
            and self._global_rollout_success_active(micro_batch)
        )
        if global_rollout_success:
            planned_terms.append(
                ("rollout_success", self.sft_rollout_success_ratio)
            )

        if not planned_terms:
            self._restore_policy_grad_after_sft(enabled_grad_names)
            # Sparse-reward success SFT often has no positive chunk in a
            # micro-batch. Skip instead of crashing the SFT-only phase.
            if self._in_sft_only_phase() and self.sft_offline_ratio > 0:
                raise RuntimeError(
                    "SFT-only co-train phase produced no SFT loss terms. "
                    "Check sft_data_path and sft_co_train.offline_sft_ratio."
                )
            metrics_data["sft_loss"] = 0.0
            metrics_data["sft_co_train/scaled_sft"] = 0.0
            if self.sft_rollout_success:
                metrics_data["sft_loss/rollout_success"] = 0.0
            return

        total_weight = sum(weight for _, weight in planned_terms)
        raw_loss_sum = 0.0
        scaled_loss_sum = 0.0

        try:
            for name, weight in planned_terms:
                normalized_weight = weight / total_weight
                if name == "offline":
                    try:
                        observation, actions = next(self.sft_iterator)
                    except StopIteration:
                        self.train_epoch += 1
                        self.data_loader.set_epoch(self.train_epoch)
                        self.sft_iterator = iter(self.data_loader)
                        observation, actions = next(self.sft_iterator)

                    sft_loss = self.model(
                        data=(observation, actions),
                        forward_type=ForwardType.SFT,
                    )
                else:
                    sft_loss = self._compute_rollout_success_sft_loss_for_backward(
                        micro_batch
                    )
                    if not self._micro_batch_has_rollout_success(micro_batch):
                        metrics_data["sft_loss/rollout_success"] = 0.0

                scaled = self._scale_single_sft_term(
                    ppo_loss_value, sft_loss, normalized_weight
                )
                metrics_data[f"sft_loss/{name}"] = sft_loss.detach().item()
                self.grad_scaler.scale(scaled / grad_scale_divisor).backward()
                if name != "rollout_success" or self._micro_batch_has_rollout_success(
                    micro_batch
                ):
                    raw_loss_sum += sft_loss.detach().item() * normalized_weight
                scaled_loss_sum += scaled.detach().item()
        finally:
            self._restore_policy_grad_after_sft(enabled_grad_names)

        metrics_data["sft_loss"] = float(raw_loss_sum)
        metrics_data["sft_co_train/scaled_sft"] = float(scaled_loss_sum)
        metrics_data["loss_ratio"] = (
            np.abs(metrics_data["sft_loss"]) / np.abs(ppo_loss_value)
            if np.abs(ppo_loss_value) > 0
            else float("inf")
        )
        if metrics_data["loss_ratio"] > 1e5 and not self.sft_dynamic_loss_norm:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={metrics_data['loss_ratio']:.3e}, "
                f"sft_loss={metrics_data['sft_loss']:.6f}, "
                f"ppo_loss={ppo_loss_value:.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )

    def _train_sft_epoch(
        self,
        metrics_data: dict[str, torch.Tensor],
        loss: torch.Tensor,
        micro_batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Legacy combined SFT loss path (used by NFT co-train)."""
        ppo_loss_value = (
            loss.detach().item() if isinstance(loss, torch.Tensor) else float(loss)
        )
        metrics_data["ppo_loss"] = ppo_loss_value

        if self._in_critic_warmup_phase():
            return loss

        need_policy_grad = self._in_sft_only_phase()
        enabled_grad_names: list[str] = []
        if need_policy_grad:
            enabled_grad_names = self._enable_policy_grad_for_sft()

        weighted_sft_terms: list[tuple[torch.Tensor, float]] = []
        try:
            if self.sft_offline_ratio > 0:
                try:
                    observation, actions = next(self.sft_iterator)
                except StopIteration:
                    self.train_epoch += 1
                    self.data_loader.set_epoch(self.train_epoch)
                    self.sft_iterator = iter(self.data_loader)
                    observation, actions = next(self.sft_iterator)

                offline_sft = self.model(
                    data=(observation, actions),
                    forward_type=ForwardType.SFT,
                )
                weighted_sft_terms.append((offline_sft, self.sft_offline_ratio))
                metrics_data["sft_loss/offline"] = offline_sft.detach().item()
        finally:
            self._restore_policy_grad_after_sft(enabled_grad_names)

        if not weighted_sft_terms:
            return loss

        total_weight = sum(weight for _, weight in weighted_sft_terms)
        combined_sft = sum(
            term * weight for term, weight in weighted_sft_terms
        ) / total_weight
        metrics_data["sft_loss"] = combined_sft.detach().item()
        scaled_sft = self._scale_single_sft_term(
            ppo_loss_value, combined_sft, 1.0
        )
        metrics_data["sft_co_train/scaled_sft"] = scaled_sft.detach().item()
        return loss + scaled_sft if loss.requires_grad else scaled_sft

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        log_progress(
            "Actor", f"rank={self._rank} run_training: start", rank=self._rank
        )
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for epoch_idx in range(update_epoch):
            log_progress(
                "Actor",
                f"rank={self._rank} run_training: update_epoch "
                f"{epoch_idx + 1}/{update_epoch}",
                rank=self._rank,
            )
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    self.train_micro_batch(
                        micro_batch=batch,
                        metrics=metrics,
                        is_last=(idx + 1) == self.gradient_accumulation,
                    )
                    # avoid gpu memory leak
                    train_micro_batch[idx] = None
                    del batch

                self.torch_platform.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        explained_variance_stats = pop_critic_explained_variance_stats(metrics)
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        if explained_variance_stats:
            reduced_stats = all_reduce_dict(
                explained_variance_stats, op=torch.distributed.ReduceOp.SUM
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )

        log_progress(
            "Actor", f"rank={self._rank} run_training: done", rank=self._rank
        )
        return mean_metric_dict

    def train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
    ) -> None:
        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        advantages = micro_batch["advantages"]
        prev_logprobs = micro_batch["prev_logprobs"]
        returns = micro_batch.get("returns", None)
        prev_values = micro_batch.get("prev_values", None)
        loss_mask = micro_batch.get("loss_mask", None)
        loss_mask_sum = micro_batch.get("loss_mask_sum", None)
        forward_inputs = micro_batch.get("forward_inputs", None)

        kwargs = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs["temperature"] = self.cfg.rollout.sampling_params.temperature_train
            kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            kwargs["prev_logprobs"] = prev_logprobs

        compute_values = self.cfg.algorithm.adv_type == "gae"
        ppo_loss = None
        metrics_data: dict = {}
        metrics_data["sft_co_train/sft_only_phase"] = float(self._in_sft_only_phase())

        if not self._in_sft_only_phase():
            with self.amp_context:
                output_dict = self.model(
                    forward_inputs=forward_inputs,
                    compute_logprobs=True,
                    compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                    compute_values=compute_values,
                    use_cache=False,
                    **kwargs,
                )

            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.GR00T,
                SupportedModel.GR00T_N1D6,
                SupportedModel.GR00T_N1D7,
                SupportedModel.ABOT_M0,
            ]:
                prev_logprobs = output_dict["prev_logprobs"]

            loss_kwargs = {
                "loss_type": self.cfg.algorithm.loss_type,
                "logprob_type": self.cfg.algorithm.logprob_type,
                "reward_type": self.cfg.algorithm.reward_type,
                "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                "logprobs": output_dict["logprobs"],
                "values": output_dict.get("values", None),
                "old_logprobs": prev_logprobs,
                "advantages": advantages,
                "returns": returns,
                "prev_values": prev_values,
                "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                "value_clip": self.cfg.algorithm.get("value_clip", None),
                "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                "loss_mask": loss_mask,
                "loss_mask_sum": loss_mask_sum,
                "max_episode_steps": self.cfg.env.train.max_episode_steps,
                "task_type": self.cfg.runner.task_type,
                "critic_warmup": self._in_critic_warmup_phase(),
            }

            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.GR00T_N1D6,
                SupportedModel.GR00T_N1D7,
            ]:
                loss_kwargs["clip_ratio_c"] = self.cfg.algorithm.get("clip_ratio_c", 3.0)
                if self.cfg.algorithm.get("clip_log_ratio_min") is not None:
                    loss_kwargs["clip_log_ratio_min"] = (
                        self.cfg.algorithm.clip_log_ratio_min
                    )
                if self.cfg.algorithm.get("clip_log_ratio_max") is not None:
                    loss_kwargs["clip_log_ratio_max"] = (
                        self.cfg.algorithm.clip_log_ratio_max
                    )

            ppo_loss, metrics_data = policy_loss(**loss_kwargs)
            entropy_loss = torch.tensor(
                0.0, device=Worker.torch_platform.current_device()
            )
            if (
                self.cfg.algorithm.entropy_bonus > 0
                and not loss_kwargs["critic_warmup"]
            ):
                entropy = output_dict["entropy"]
                entropy = reshape_entropy(
                    entropy,
                    entropy_type=self.cfg.algorithm.entropy_type,
                    action_dim=self.cfg.actor.model.get("action_dim", 7),
                    batch_size=output_dict["logprobs"].shape[0],
                )
                entropy_loss = masked_mean(entropy, mask=loss_mask)
                ppo_loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
            metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()
        else:
            metrics_data["ppo_loss"] = 0.0
            metrics_data["actor/entropy_loss"] = 0.0

        ppo_loss_value = (
            ppo_loss.detach().item()
            if isinstance(ppo_loss, torch.Tensor)
            else float(metrics_data.get("ppo_loss", 0.0))
        )
        metrics_data["ppo_loss"] = ppo_loss_value

        grad_scale_divisor = float(self.gradient_accumulation)
        with backward_ctx:
            if ppo_loss is not None and ppo_loss.requires_grad:
                self.grad_scaler.scale(ppo_loss / grad_scale_divisor).backward()

            if self.enable_sft_co_train:
                self._backward_sft_co_train_terms(
                    metrics_data,
                    ppo_loss_value,
                    micro_batch,
                    grad_scale_divisor,
                )
            elif self._in_sft_only_phase():
                raise RuntimeError(
                    "SFT-only co-train phase requires enable_sft_co_train=True."
                )

        total_loss_value = ppo_loss_value + metrics_data.get("sft_co_train/scaled_sft", 0.0)
        metrics_data["actor/total_loss"] = total_loss_value
        append_to_dict(metrics, metrics_data)

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def finish_global_batch(self, metrics: dict[str, list[float]]) -> None:
        self.torch_platform.empty_cache()
        grad_norm, lr_list = self.optimizer_step()
        self.optimizer.zero_grad()
        metric_data = {
            "actor/grad_norm": grad_norm,
            "actor/lr": lr_list[0],
        }
        if len(lr_list) > 1:
            metric_data["critic/lr"] = lr_list[1]
        append_to_dict(metrics, metric_data)
