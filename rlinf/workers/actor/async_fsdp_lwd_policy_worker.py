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

"""Async LWD worker: background traj recv without intervention → demo."""

from __future__ import annotations

import asyncio
import queue
import threading
import time

import torch

from rlinf.scheduler import Worker
from rlinf.utils.logging import log_progress
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.workers.actor.fsdp_lwd_policy_worker import EmbodiedLWDFSDPPolicy


class AsyncEmbodiedLWDFSDPPolicy(EmbodiedLWDFSDPPolicy):
    should_stop = False

    async def recv_rollout_trajectories(self, input_channel):
        if getattr(self, "_recv_queue", None) is None:
            self._recv_queue = queue.Queue()
        if (
            getattr(self, "_recv_rollout_thread", None) is None
            or not self._recv_rollout_thread.is_alive()
        ):
            self._recv_rollout_thread = threading.Thread(
                target=self._recv_rollout_thread_main,
                args=(input_channel,),
                daemon=True,
            )
            self._recv_rollout_thread.start()

    def _recv_rollout_thread_main(self, input_channel):
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        while not self.should_stop:
            for _ in range(split_num):
                trajectory = input_channel.get()
                self._recv_queue.put(trajectory)

    def _drain_received_trajectories(self, max_trajectories: int | None = None):
        if getattr(self, "_recv_queue", None) is None:
            return
        recv_list = []
        processed = 0
        while True:
            try:
                recv_list.append(self._recv_queue.get_nowait())
                processed += 1
                if max_trajectories is not None and processed >= max_trajectories:
                    break
            except queue.Empty:
                break
        if not recv_list:
            return
        # B_on only — never route intervention trajectories into demo_buffer.
        self.replay_buffer.add_trajectories(recv_list)

    async def _wait_for_buffer_ready(self):
        """Wait until demo (offline) or replay (online) is ready."""
        min_replay = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        allow_demo_only = bool(
            self.cfg.algorithm.get("allow_demo_only", False)
            or self.cfg.runner.get("lwd_stage", "online") == "offline"
        )
        min_demo = 0
        if self.demo_buffer is not None:
            min_demo = self.cfg.algorithm.demo_buffer.get("min_buffer_size", 0)

        while True:
            self._drain_received_trajectories(
                max_trajectories=self.cfg.actor.get("recv_drain_max_trajectories", 1024)
            )
            if allow_demo_only and self.demo_buffer is not None:
                if await self.demo_buffer.is_ready_async(min_demo):
                    return
            if await self.replay_buffer.is_ready_async(min_replay):
                return
            await asyncio.sleep(1)

    @Worker.timer("run_training")
    async def run_training(self):
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        await self._wait_for_buffer_ready()
        torch.distributed.barrier()

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self._maybe_freeze_backbone()
        self.model.train()
        metrics = {}

        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
        log_interval = max(1, int(self.cfg.algorithm.get("update_log_interval", 1)))
        start_msg = (
            f"LWD run_training: start update_epoch={update_epoch} "
            f"(update_step={self.update_step}, log_interval={log_interval})"
        )
        log_progress("Actor", start_msg, rank=self._rank)
        self.log_on_first_rank(start_msg)

        train_start = time.time()
        for epoch_idx in range(update_epoch):
            await asyncio.sleep(0)
            iter_start = time.time()
            metrics_data = self.update_one_epoch()
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

            done = epoch_idx + 1
            if done % log_interval != 0 and done != update_epoch:
                continue

            iter_sec = time.time() - iter_start
            total_sec = time.time() - train_start
            loss_parts = []
            for key in ("lwd/value_loss", "lwd/critic_loss", "lwd/qam_loss"):
                if key in metrics_data:
                    loss_parts.append(f"{key.split('/')[-1]}={metrics_data[key]:.4f}")
            loss_summary = ", ".join(loss_parts) if loss_parts else "loss=n/a"
            progress_msg = (
                f"LWD update_epoch {done}/{update_epoch} "
                f"(update_step={self.update_step}, "
                f"iter={iter_sec:.1f}s, elapsed={total_sec:.1f}s, {loss_summary})"
            )
            log_progress("Actor", progress_msg, rank=self._rank)
            self.log_on_first_rank(progress_msg)

        done_msg = (
            f"LWD run_training: finished {update_epoch} update(s) in "
            f"{time.time() - train_start:.1f}s (update_step={self.update_step})"
        )
        log_progress("Actor", done_msg, rank=self._rank)
        self.log_on_first_rank(done_msg)

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict

    async def stop(self):
        self.should_stop = True
        self.buffer_dataset.close()
        recv_thread = getattr(self, "_recv_rollout_thread", None)
        if recv_thread is not None and recv_thread.is_alive():
            await asyncio.to_thread(recv_thread.join, 5)
