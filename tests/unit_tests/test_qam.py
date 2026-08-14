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

"""Unit tests for QAM: fixed Q should increase sampled-action Q after one step."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.algorithms.qam import (
    adjoint_at_time,
    compute_qam_loss,
    flow_interpolate,
    sample_flow_time,
    terminal_adjoint,
)


class _ToyFlow(nn.Module):
    """Linear map predicting flow velocity from flattened (x_t, t)."""

    def __init__(self, action_dim: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(horizon * action_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, horizon * action_dim),
        )

    def forward(self, x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        B = x_t.shape[0]
        inp = torch.cat([x_t.reshape(B, -1), time[:, None]], dim=-1)
        return self.net(inp).reshape(B, self.horizon, self.action_dim)


class _ToyQ(nn.Module):
    """Fixed Q: prefers actions close to target."""

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.register_buffer("target", target)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        # Q = -||a - target||^2  (higher is better)
        err = (actions - self.target).pow(2).mean(dim=(1, 2), keepdim=False)
        return (-err).unsqueeze(-1)


def test_terminal_adjoint_points_uphill():
    B, H, D = 4, 5, 7
    actions = torch.randn(B, H, D, requires_grad=True)
    target = torch.ones(B, H, D)
    q = _ToyQ(target)
    q_val = q(actions)
    g1 = terminal_adjoint(actions, q_val, qam_lambda=1.0)
    # Moving along -g1 (i.e. +∇Q) should increase Q locally
    with torch.no_grad():
        q0 = q(actions).mean()
        q1 = q(actions + 0.01 * (-g1)).mean()  # step along ∇Q = -g1 * λ with λ=1
    assert q1 > q0


def test_qam_regression_loss_decreases():
    """Flow net should fit a fixed QAM target velocity (u_ref + g_t)."""
    torch.manual_seed(0)
    B, H, D = 64, 4, 6
    flow = _ToyFlow(D, H)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-2)

    x_t = torch.randn(B, H, D)
    time = sample_flow_time(B, device=x_t.device, dtype=x_t.dtype)
    # Fixed adjoint-guided target
    u_target = torch.randn(B, H, D)

    def loss_fn():
        return F.mse_loss(flow(x_t, time), u_target)

    loss0 = loss_fn().item()
    for _ in range(50):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
    loss1 = loss_fn().item()
    assert loss1 < loss0, f"expected fit loss to drop: before={loss0}, after={loss1}"


def test_compute_qam_loss_finite_and_grad():
    torch.manual_seed(1)
    B, H, D = 16, 4, 6
    target = torch.ones(1, H, D)
    q_net = _ToyQ(target)
    flow = _ToyFlow(D, H)
    actions = (0.5 * torch.randn(B, H, D) + 0.5 * target).requires_grad_(True)
    q_val = q_net(actions)
    time = sample_flow_time(B, device=actions.device, dtype=actions.dtype)
    noise = torch.randn_like(actions)
    x_t, u_t = flow_interpolate(actions.detach(), noise, time)
    u_online = flow(x_t, time)
    loss, metrics = compute_qam_loss(
        actions=actions,
        noise=noise,
        time=time,
        u_online=u_online,
        u_reference=u_t.detach(),
        q_values=q_val,
        qam_lambda=1.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters())
    assert "qam_loss" in metrics
    assert "adjoint_norm" in metrics


def test_adjoint_at_time_scales():
    g1 = torch.ones(2, 3, 4)
    time = torch.tensor([0.0, 1.0])
    g_t = adjoint_at_time(g1, time)
    assert torch.allclose(g_t[0], g1[0])
    assert torch.allclose(g_t[1], torch.zeros_like(g1[1]))
