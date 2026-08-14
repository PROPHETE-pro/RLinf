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

"""DIVL: Distributional Implicit V-Learning with adaptive quantile."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DIVLConfig:
    num_atoms: int = 51
    v_min: float = 0.0
    v_max: float = 1.0
    tau_base: float = 0.5
    tau_min: float = 0.1
    tau_max: float = 0.9
    entropy_alpha: float = 1.0
    gamma: float = 0.999
    num_action_chunks: int = 10
    agg_q: str = "min"  # min | mean


def categorical_support(
    num_atoms: int,
    v_min: float,
    v_max: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return atom support z of shape [num_atoms]."""
    return torch.linspace(v_min, v_max, num_atoms, device=device, dtype=dtype)


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Softmax over the atom dimension (last dim)."""
    return F.softmax(logits, dim=-1)


def categorical_expectation(
    probs: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    """E[V] from categorical probs. probs [B, A], support [A] -> [B, 1]."""
    return (probs * support.unsqueeze(0)).sum(dim=-1, keepdim=True)


def categorical_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Shannon entropy of categorical distribution. probs [B, A] -> [B, 1]."""
    log_p = torch.log(probs.clamp_min(eps))
    return -(probs * log_p).sum(dim=-1, keepdim=True)


def normalized_entropy(
    probs: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Entropy / log(A) in [0, 1]."""
    num_atoms = probs.shape[-1]
    max_ent = torch.log(
        torch.tensor(float(num_atoms), device=probs.device, dtype=probs.dtype)
    )
    return categorical_entropy(probs, eps=eps) / max_ent.clamp_min(eps)


def adaptive_tau(
    probs: torch.Tensor,
    *,
    tau_base: float,
    tau_min: float,
    tau_max: float,
    alpha: float,
) -> torch.Tensor:
    """Schedule τ from normalized entropy; stop-gradient for bootstrap.

    Higher entropy (uncertain V) -> higher τ (more optimistic quantile).
    """
    h = normalized_entropy(probs).detach()
    tau = tau_base + alpha * (h - 0.5)
    return tau.clamp(tau_min, tau_max)


def categorical_quantile(
    probs: torch.Tensor, support: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """Inverse-CDF quantile of categorical V.

    Args:
        probs: [B, A]
        support: [A] sorted ascending
        tau: [B, 1] in (0, 1)

    Returns:
        [B, 1] quantile values
    """
    cdf = torch.cumsum(probs, dim=-1)
    # First atom where CDF >= τ
    ge = cdf >= tau
    # Fallback to last atom if numerical issues
    idx = ge.float().argmax(dim=-1)
    none_ge = ~ge.any(dim=-1)
    idx = torch.where(
        none_ge,
        torch.full_like(idx, probs.shape[-1] - 1),
        idx,
    )
    return support[idx].unsqueeze(-1)


def project_distribution(
    target_values: torch.Tensor,
    support: torch.Tensor,
    v_min: float,
    v_max: float,
) -> torch.Tensor:
    """Project scalar targets onto categorical support (C51-style).

    Args:
        target_values: [B, 1]
        support: [A]

    Returns:
        target_probs [B, A]
    """
    num_atoms = support.shape[0]
    delta_z = (v_max - v_min) / max(num_atoms - 1, 1)
    # Keep projection in float32 for numerical stability, then cast masses
    # back to target dtype (often bfloat16) for scatter_add_.
    tz = target_values.detach().float().clamp(v_min, v_max)  # [B, 1]
    b = (tz - v_min) / max(delta_z, 1e-8)  # [B, 1] float32
    lower = b.floor().long().clamp(0, num_atoms - 1)
    upper = b.ceil().long().clamp(0, num_atoms - 1)
    # When lower==upper (exact atom), put full mass there.
    out_dtype = target_values.dtype
    target_probs = torch.zeros(
        target_values.shape[0],
        num_atoms,
        device=target_values.device,
        dtype=out_dtype,
    )
    upper_mass = (b - lower.float()).to(dtype=out_dtype)
    lower_mass = (1.0 - (b - lower.float())).to(dtype=out_dtype)
    target_probs.scatter_add_(1, lower, lower_mass)
    target_probs.scatter_add_(1, upper, upper_mass)
    return target_probs


def aggregate_q(q_values: torch.Tensor, agg: str = "min") -> torch.Tensor:
    """Aggregate multi-head Q. q_values [B, num_q] -> [B, 1]."""
    if q_values.ndim == 1:
        return q_values.unsqueeze(-1)
    if agg == "min":
        return q_values.min(dim=-1, keepdim=True).values
    if agg == "mean":
        return q_values.mean(dim=-1, keepdim=True)
    raise ValueError(f"Unsupported agg_q={agg}")


def chunk_sparse_reward(rewards: torch.Tensor) -> torch.Tensor:
    """Aggregate chunk rewards for LWD sparse success.

    rewards: [B, chunk] or [B, 1] -> [B, 1] (sum over chunk).
    """
    if rewards.ndim == 1:
        return rewards.unsqueeze(-1)
    return rewards.sum(dim=-1, keepdim=True)


def divl_value_loss(
    v_logits: torch.Tensor,
    target_q: torch.Tensor,
    *,
    support: torch.Tensor,
    v_min: float,
    v_max: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Categorical V NLL against EMA critic target Q(s, a).

    Args:
        v_logits: [B, num_atoms]
        target_q: [B, 1] detached Q from target critic
    """
    target_probs = project_distribution(
        target_q.detach(), support, v_min, v_max
    )
    log_probs = F.log_softmax(v_logits, dim=-1)
    loss = -(target_probs * log_probs).sum(dim=-1).mean()
    probs = logits_to_probs(v_logits.detach())
    v_mean = categorical_expectation(probs, support).mean().item()
    return loss, {"v_mean": v_mean, "v_nll": loss.item()}


def divl_critic_targets(
    next_v_logits: torch.Tensor,
    rewards: torch.Tensor,
    terminations: torch.Tensor,
    *,
    support: torch.Tensor,
    cfg: DIVLConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Bootstrap targets y_Q = r_chunk + γ^H Quant_τ(V(s')).

    Args:
        next_v_logits: [B, num_atoms] from online or target V (no grad)
        rewards: [B, chunk] or [B, 1]
        terminations: [B, chunk] or [B, 1] bool/float

    Returns:
        y_q [B, 1], metrics
    """
    probs = logits_to_probs(next_v_logits)
    tau = adaptive_tau(
        probs,
        tau_base=cfg.tau_base,
        tau_min=cfg.tau_min,
        tau_max=cfg.tau_max,
        alpha=cfg.entropy_alpha,
    )
    quant = categorical_quantile(probs, support, tau)
    r = chunk_sparse_reward(rewards).to(dtype=quant.dtype)
    if terminations.ndim > 1:
        done = terminations.any(dim=-1, keepdim=True)
    else:
        done = terminations.unsqueeze(-1)
    done = done.to(dtype=torch.bool)
    discount = cfg.gamma ** cfg.num_action_chunks
    y_q = r + (~done).to(dtype=quant.dtype) * discount * quant
    return y_q, {
        "tau_mean": tau.mean().item(),
        "quant_mean": quant.mean().item(),
        "reward_mean": r.mean().item(),
    }


def divl_critic_loss(
    q_values: torch.Tensor,
    target_q: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """TD MSE for (multi-head) chunk-level Q."""
    target = target_q.detach().expand_as(q_values)
    loss = F.mse_loss(q_values, target)
    return loss, {
        "q_mean": q_values.mean().item(),
        "q_target_mean": target_q.mean().item(),
        "critic_td": loss.item(),
    }
