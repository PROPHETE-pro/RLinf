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

"""Unit tests for DIVL (categorical V, adaptive tau, chunk TD Q)."""

import torch

from rlinf.algorithms.divl import (
    DIVLConfig,
    adaptive_tau,
    categorical_expectation,
    categorical_quantile,
    categorical_support,
    divl_critic_loss,
    divl_critic_targets,
    divl_value_loss,
    logits_to_probs,
    normalized_entropy,
)


def test_divl_value_nll_finite():
    B, A = 16, 51
    support = categorical_support(A, 0.0, 1.0, device=torch.device("cpu"), dtype=torch.float32)
    logits = torch.randn(B, A, requires_grad=True)
    target_q = torch.rand(B, 1)
    loss, metrics = divl_value_loss(
        logits, target_q, support=support, v_min=0.0, v_max=1.0
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert "v_mean" in metrics


def test_divl_adaptive_tau_and_quantile():
    B, A = 8, 51
    support = categorical_support(A, 0.0, 1.0, device=torch.device("cpu"), dtype=torch.float32)
    logits = torch.randn(B, A)
    probs = logits_to_probs(logits)
    h = normalized_entropy(probs)
    assert (h >= 0).all() and (h <= 1.0 + 1e-5).all()
    tau = adaptive_tau(
        probs, tau_base=0.5, tau_min=0.1, tau_max=0.9, alpha=1.0
    )
    assert (tau >= 0.1).all() and (tau <= 0.9).all()
    # stop-gradient: changing probs after detach schedule shouldn't require grad on tau
    assert not tau.requires_grad
    quant = categorical_quantile(probs, support, tau)
    assert quant.shape == (B, 1)
    assert (quant >= 0.0 - 1e-5).all() and (quant <= 1.0 + 1e-5).all()


def test_divl_critic_td_finite():
    B, A, Q = 16, 51, 2
    support = categorical_support(A, 0.0, 1.0, device=torch.device("cpu"), dtype=torch.float32)
    next_v = torch.randn(B, A)
    rewards = torch.zeros(B, 10)
    rewards[:, -1] = torch.rand(B)  # sparse chunk-end reward
    terminations = torch.zeros(B, 10, dtype=torch.bool)
    terminations[:, -1] = torch.rand(B) > 0.5
    cfg = DIVLConfig(num_atoms=A, num_action_chunks=10, gamma=0.99)
    y_q, tgt_metrics = divl_critic_targets(
        next_v, rewards, terminations, support=support, cfg=cfg
    )
    assert torch.isfinite(y_q).all()
    assert "tau_mean" in tgt_metrics

    q_values = torch.randn(B, Q, requires_grad=True)
    loss, metrics = divl_critic_loss(q_values, y_q)
    assert torch.isfinite(loss)
    loss.backward()
    assert q_values.grad is not None


def test_expectation_matches_uniform():
    A = 51
    support = categorical_support(A, 0.0, 1.0, device=torch.device("cpu"), dtype=torch.float32)
    probs = torch.ones(1, A) / A
    mean = categorical_expectation(probs, support)
    assert torch.isclose(mean, torch.tensor([[0.5]]), atol=1e-3)
