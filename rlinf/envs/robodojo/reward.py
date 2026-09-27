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

"""Chunk reward for RoboDojo training.

The cumulative score follows RoboTwin: success jumps to ``dense_success_reward``
and optional progress is ``dense_shaping_coef * dense_progress``. The trainer
uses the difference from the previous chunk, then subtracts a per-step time cost.
"""

from __future__ import annotations

import math

def success_deadline_scale(
    step_count: int,
    official_step_lim: int,
    mode: str,
    decay_steps: float,
) -> float:
    """Scale of the terminal success score relative to the official step limit."""
    if step_count <= official_step_lim:
        return 1.0
    if str(mode).lower() == "decay":
        width = float(decay_steps)
        if width <= 0:
            return 0.0
        return math.exp(-(step_count - official_step_lim) / width)
    return 0.0


def cumulative_score(
    *,
    success: bool,
    step_count: int,
    official_step_lim: int,
    deadline_mode: str = "hard",
    decay_steps: float = 525.0,
    dense_progress: float | None = None,
    use_dense_reward: bool = False,
    shaping_coef: float = 0.5,
    success_reward: float = 1.0,
) -> float:
    """Cumulative score after one chunk. Missing progress leaves a sparse score."""
    progress = None
    if use_dense_reward and dense_progress is not None:
        progress = float(shaping_coef) * float(dense_progress)
    if success:
        scale = success_deadline_scale(
            int(step_count), int(official_step_lim), deadline_mode, decay_steps
        )
        if scale > 0.0:
            return float(success_reward) * scale
        if progress is not None:
            return progress
        return 0.0
    if progress is not None:
        return progress
    return 0.0


def reallocate_chunk_advantages(
    advantages: "torch.Tensor",
    progress_deltas: "torch.Tensor",
) -> "torch.Tensor":
    """Split a chunk advantage across control steps without changing its sum.

    ``advantages`` is ``[n_chunk, batch, 1]`` from chunk-level GAE.
    ``progress_deltas`` is ``[n_chunk, batch, T]``, one progress change per step.
    Step ``t`` receives ``d_t + (A - sum(d)) / T``, so the 50 values still add up to ``A``.
    """
    import torch

    if advantages.ndim == 2:
        advantages = advantages.unsqueeze(-1)
    chunk_adv = advantages[..., 0]
    deltas = progress_deltas.to(dtype=chunk_adv.dtype, device=chunk_adv.device)
    if deltas.shape[:2] != chunk_adv.shape:
        raise ValueError(
            f"progress deltas {tuple(deltas.shape)} do not match advantages {tuple(chunk_adv.shape)}"
        )
    steps = deltas.shape[-1]
    net = deltas.sum(dim=-1)
    shared = (chunk_adv - net).unsqueeze(-1) / max(steps, 1)
    return deltas + shared


def chunk_reward_from_cumulative(
    cumulative: float,
    prev_cumulative: float,
    time_cost_per_step: float,
    executed_steps: int,
) -> float:
    """Per-chunk PPO reward: cumulative delta minus the time cost of this chunk."""
    steps = max(int(executed_steps), 0)
    return float(cumulative) - float(prev_cumulative) - float(time_cost_per_step) * steps
