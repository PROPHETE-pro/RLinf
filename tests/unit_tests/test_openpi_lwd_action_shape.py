# Copyright 2026 The RLinf Authors.

from types import SimpleNamespace

import torch

from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0ForRLActionPrediction


def test_reshape_actions_for_flow_restores_flat_chunk():
    model = SimpleNamespace(
        config=SimpleNamespace(
            action_dim=32,
            action_env_dim=14,
            action_horizon=50,
            action_chunk=50,
        ),
    )
    model._flow_env_action_shape = (
        lambda: OpenPi0ForRLActionPrediction._flow_env_action_shape(model)
    )
    flat = torch.arange(4 * 700, dtype=torch.float32).reshape(4, 700)
    out = OpenPi0ForRLActionPrediction._reshape_actions_for_flow(model, flat)
    assert out.shape == (4, 50, 14)
