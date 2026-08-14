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

"""QAM: Q-weighted Adjoint Matching for flow action experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class QAMConfig:
    qam_lambda: float = 1.0
    num_flow_steps: int = 1  # number of t samples per action (expectation)
    detach_reference: bool = True


def sample_flow_time(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample t ~ U(0, 1) with shape [B]."""
    return torch.rand(batch_size, device=device, dtype=dtype)


def flow_interpolate(
    actions: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """OpenPI-style flow path: x_t = t*noise + (1-t)*a, u = noise - a.

    Args:
        actions: [B, H, D] clean action chunk
        noise: [B, H, D]
        time: [B]

    Returns:
        x_t, u_t with same shape as actions
    """
    t = time[:, None, None]
    x_t = t * noise + (1.0 - t) * actions
    u_t = noise - actions
    return x_t, u_t


def terminal_adjoint(
    actions: torch.Tensor,
    q_values: torch.Tensor,
    *,
    qam_lambda: float,
) -> torch.Tensor:
    """Terminal adjoint g̃_1 = -∇_a Q / λ.

    Args:
        actions: [B, H, D] requiring grad
        q_values: [B, 1] scalar Q (already aggregated)
        qam_lambda: temperature / scale

    Returns:
        g1 [B, H, D]
    """
    if qam_lambda <= 0:
        raise ValueError(f"qam_lambda must be > 0, got {qam_lambda}")
    # Sum Q so autograd yields ∇_a Q per sample
    grads = torch.autograd.grad(
        outputs=q_values.sum(),
        inputs=actions,
        create_graph=False,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return -grads / qam_lambda


def adjoint_at_time(g1: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Propagate terminal adjoint along linear flow: g̃_t ≈ (1-t) g̃_1.

    For the OT path x_t = t ε + (1-t) a, the adjoint of the endpoint w.r.t.
    intermediate state scales by (1-t) under the linear interpolant.
    """
    t = time[:, None, None]
    return (1.0 - t) * g1


def qam_target_velocity(
    u_ref: torch.Tensor,
    g_t: torch.Tensor,
) -> torch.Tensor:
    """Reference velocity plus adjoint guidance: u* = u_β + g̃_t."""
    return u_ref + g_t


def qam_loss(
    u_pred: torch.Tensor,
    u_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L_QAM = ||u_θ(x_t, t) - (u_β + g̃_t)||^2."""
    loss = F.mse_loss(u_pred, u_target.detach())
    return loss, {
        "qam_loss": loss.item(),
        "u_pred_norm": u_pred.detach().norm().item(),
        "u_target_norm": u_target.detach().norm().item(),
    }


def compute_qam_loss(
    *,
    actions: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
    u_online: torch.Tensor,
    u_reference: torch.Tensor,
    q_values: torch.Tensor,
    qam_lambda: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full QAM objective given velocities and Q(s, a).

    actions must require grad for the adjoint; q_values should depend on
    actions through the critic graph (or be recomputed with create_graph).
    """
    if not actions.requires_grad:
        actions = actions.detach().requires_grad_(True)
    g1 = terminal_adjoint(actions, q_values, qam_lambda=qam_lambda)
    g_t = adjoint_at_time(g1, time)
    u_star = qam_target_velocity(u_reference, g_t)
    loss, metrics = qam_loss(u_online, u_star)
    metrics["adjoint_norm"] = g1.detach().norm().item()
    metrics["q_for_qam"] = q_values.detach().mean().item()
    return loss, metrics
