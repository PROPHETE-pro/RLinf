# Copyright 2025 The RLinf Authors.

from omegaconf import OmegaConf

from rlinf.utils.robotwin_hang_diagnostics import (
    configure_from_cfg,
    enabled,
    is_robotwin_train_env,
    should_enable_robotwin_hang_diagnostics,
)


def test_robotwin_train_enables_diagnostics():
    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": False},
            "env": {"train": {"env_type": "robotwin"}},
        }
    )
    assert is_robotwin_train_env(cfg)
    assert should_enable_robotwin_hang_diagnostics(cfg)
    configure_from_cfg(cfg)
    assert enabled()


def test_libero_train_disables_diagnostics():
    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": False},
            "env": {"train": {"env_type": "libero"}},
        }
    )
    assert not is_robotwin_train_env(cfg)
    configure_from_cfg(cfg)
    assert not enabled()


def test_robotwin_only_eval_disables_diagnostics():
    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": True},
            "env": {"train": {"env_type": "robotwin"}},
        }
    )
    configure_from_cfg(cfg)
    assert not enabled()


def test_robotwin_explicit_disable():
    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": False},
            "env": {
                "train": {
                    "env_type": "robotwin",
                    "enable_hang_diagnostics": False,
                }
            },
        }
    )
    configure_from_cfg(cfg)
    assert not enabled()
