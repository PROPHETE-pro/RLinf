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

"""Gate RoboTwin hang diagnostics (PROGRESS logs + chunk_step watchdog).

Enabled only when training with ``env.train.env_type: robotwin``.
Libero / ManiSkill / other env types are unaffected.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

_ENABLED: bool = False
_HEARTBEAT_INTERVAL_SEC: float = 60.0
_HEARTBEAT_STUCK_THRESHOLD_SEC: float = 45.0
_CHUNK_LOG_INTERVAL: int = 8
_MILESTONE_ALL_RANKS: bool = False


def is_robotwin_train_env(cfg: DictConfig) -> bool:
    """True when this job trains (not only_eval) with env_type robotwin."""
    if bool(OmegaConf.select(cfg, "runner.only_eval", default=False)):
        return False
    train_env_cfg = OmegaConf.select(cfg, "env.train", default=None)
    if train_env_cfg is None:
        return False
    env_type = OmegaConf.select(cfg, "env.train.env_type", default=None)
    if env_type is None:
        return False
    return str(env_type).lower() == "robotwin"


def hang_diagnostics_explicitly_enabled(cfg: DictConfig) -> bool:
    """Respect ``env.train.enable_hang_diagnostics`` when set; default True for robotwin."""
    val = OmegaConf.select(cfg, "env.train.enable_hang_diagnostics", default=None)
    if val is None:
        return True
    return bool(val)


def should_enable_robotwin_hang_diagnostics(cfg: DictConfig) -> bool:
    return is_robotwin_train_env(cfg) and hang_diagnostics_explicitly_enabled(cfg)


def configure_from_cfg(cfg: DictConfig) -> bool:
    """Set process-local flag from Hydra cfg. Call once per process at worker/runner init."""
    global _ENABLED
    global _HEARTBEAT_INTERVAL_SEC
    global _HEARTBEAT_STUCK_THRESHOLD_SEC
    global _CHUNK_LOG_INTERVAL
    global _MILESTONE_ALL_RANKS

    _ENABLED = should_enable_robotwin_hang_diagnostics(cfg)
    if _ENABLED:
        _HEARTBEAT_INTERVAL_SEC = float(
            OmegaConf.select(
                cfg, "env.train.hang_diagnostics_heartbeat_interval_sec", default=60.0
            )
        )
        _HEARTBEAT_STUCK_THRESHOLD_SEC = float(
            OmegaConf.select(
                cfg,
                "env.train.hang_diagnostics_heartbeat_stuck_threshold_sec",
                default=45.0,
            )
        )
        _CHUNK_LOG_INTERVAL = max(
            1,
            int(
                OmegaConf.select(
                    cfg, "env.train.hang_diagnostics_chunk_log_interval", default=8
                )
            ),
        )
        _MILESTONE_ALL_RANKS = bool(
            OmegaConf.select(
                cfg, "env.train.hang_diagnostics_milestone_all_ranks", default=False
            )
        )
    return _ENABLED


def enabled() -> bool:
    return _ENABLED


def heartbeat_interval_sec() -> float:
    return _HEARTBEAT_INTERVAL_SEC


def heartbeat_stuck_threshold_sec() -> float:
    return _HEARTBEAT_STUCK_THRESHOLD_SEC


def chunk_log_interval() -> int:
    return _CHUNK_LOG_INTERVAL


def milestone_all_ranks() -> bool:
    return _MILESTONE_ALL_RANKS
