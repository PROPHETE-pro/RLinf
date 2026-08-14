# RoboTwin Hang 诊断改动备案

## 2026-07-24：Hang 诊断日志节流

### 目的

- `set_progress_state(log=True)` 在每个 chunk 的 recv/predict/send 都打印，8 rank × 32 chunk × 4 epoch 导致单 job log 数万行 PROGRESS。
- 改为默认只更新 HEARTBEAT 状态；按 chunk 间隔 / 卡住阈值打印，保留 step 级与 bootstrap/send_rollout 等关键里程碑。

### 修改文件

| 文件 | 改动 |
|------|------|
| `rlinf/utils/logging.py` | `set_progress_state` 默认 `log=False`；里程碑节流；HEARTBEAT 60s、卡住才全 rank + cuda mem |
| `rlinf/utils/robotwin_hang_diagnostics.py` | 新增节流配置项读取 |
| `examples/embodiment/config/env/robotwin_open_microwave.yaml` | 文档化 `hang_diagnostics_*` 配置 |
| `tests/unit_tests/test_hang_diagnostics_logging.py` | 节流逻辑回归 |

### 配置（`env.train`）

```yaml
hang_diagnostics_heartbeat_interval_sec: 60      # HEARTBEAT 间隔
hang_diagnostics_heartbeat_stuck_threshold_sec: 45  # 超过此秒数视为卡住，全 rank 打印
hang_diagnostics_chunk_log_interval: 8           # 每 N 个 chunk 打印一次进度
hang_diagnostics_milestone_all_ranks: false      # true=所有 rank 打印 chunk 里程碑
```

---

## 2026-07-24：Subproc recover info 对齐 + 容错

### 目的

- `SUBENV_RECOVER` 后 `_truncate_step_result` 返回的 info 仅有 `subenv_timeout` 等键，与正常 SubEnv step 的 `success` 键不一致，导致 `list_of_dict_to_dict_of_list` 在 mixed batch 中 `AssertionError` 并级联 kill 整个 job（pi05_5 log 19:31:53）。

### 修改文件

| 文件 | 改动 |
|------|------|
| `rlinf/envs/robotwin/subproc_vector_env.py` | recover info 对齐 RoboTwin（含 `success: False`）；缓存 `_last_obs`；respawn/get_obs 失败时返回 synthetic truncate 而非 crash |
| `rlinf/envs/utils.py` | `list_of_dict_to_dict_of_list` 对 mixed keys 做 union + fill_missing |
| `tests/unit_tests/test_subproc_vector_env.py` | 新增 mixed info / recovery failed 回归 |

---

## 2026-07-24：Stage-C Runner Env/Rollout 同步 + bootstrap 诊断

### 目的

- 多 rank 训练时 Runner 在 `generate_rollouts` 中过早调用 `actor.recv_rollout_trajectories().wait()`，仅 Env rank 0 先完成并 send，导致 rank 1–3 永久失步（bootstrap `recv_from` 阻塞 + 下一步 channel 混叠）。
- 调整等待顺序为 `rollout.wait → env.wait → actor.recv`，并补全 epoch 末 bootstrap 的 `set_progress_state`，避免 HEARTBEAT 误判。

### 根因摘要

1. Env/Rollout 每 rank 成对握手：chunk 循环结束后 Rollout `final bootstrap recv obs` → send bootstrap_values → Env `final bootstrap recv`。
2. 旧 Runner 顺序：launch → **actor.recv** → rollout.wait；rank 0 单通路跑通，rank 1–3 的 `interact()` 未返回即启动下一步。
3. HEARTBEAT 显示 `chunk 32/32 send_to done` 并非未进 bootstrap，而是 bootstrap 段缺少 `set_progress_state` 更新。

### 修改文件

| 文件 | 改动 |
|------|------|
| `rlinf/runners/embodied_runner.py` | `run()` / `run_pipeline()` 的 `generate_rollouts`：`rollout.wait → env.wait → (reward.wait) → actor.recv` |
| `rlinf/workers/env/env_worker.py` | bootstrap 段 `set_progress_state`；`send_rollout` / `interact returning` 等 `all_ranks=True` |
| `rlinf/workers/actor/fsdp_actor_worker.py` | `recv_rollout_trajectories: done` 加 `all_ranks=True` |
| `tests/unit_tests/test_embodied_runner_wait_order.py` | 回归：`rollout.wait → env.wait → actor.recv` 顺序 |

### 验证方法

使用 `robotwin_open_microwave_ppo_openpi_pi05_3.yaml` 跑 2–3 step，确认每 step：

- Env rank 0–3：`send_rollout_trajectories: done`
- Actor rank 0–3：`recv_rollout_trajectories: done`
- HEARTBEAT 在 epoch 末可见 `waiting final bootstrap recv from Rollout`

### 回滚 Stage-C

```bash
cd /mnt/pfs/7wsqem/grt/RLinf
git checkout -- \
  rlinf/runners/embodied_runner.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/actor/fsdp_actor_worker.py
# 并删除 CHANGELOG 中 Stage-C 章节（或 git checkout CHANGELOG_ROBOTWIN_HANG.md）
```

---

## 2026-07-24：Stage-B SubprocVectorEnv（子进程隔离 + 单 SubEnv 超时恢复）

### 目的

- RoboTwin 原 `VectorEnv` 用线程并行 SubEnv，native hang 无法打断，进程级 watchdog 只能 SIGKILL 整个 EnvWorker。
- 改为每 SubEnv 一个子进程：超时 kill 该子进程 → respawn + reset → 返回 `truncated=True, reward=0`，当前 training step 继续。

### 新增文件

| 文件 | 说明 |
|------|------|
| `rlinf/envs/robotwin/subproc_vector_env.py` | `SubprocVectorEnv` / `SubprocSubEnvWorker` / `build_robotwin_task_args` |
| `tests/unit_tests/test_subproc_vector_env.py` | 超时恢复、混合 step 等单元测试 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `rlinf/envs/robotwin/robotwin_env.py` | `_init_env()` 默认 `use_subproc_env: true` 时使用 `SubprocVectorEnv` |
| `rlinf/workers/env/env_worker.py` | robotwin + `use_subproc_env=true` 时禁用 `ChunkStepWatchdog` |
| `examples/embodiment/config/env/robotwin_open_microwave.yaml` | 新增 subproc 配置，`chunk_step_timeout_sec: 0` |

### 配置项（RoboTwin env yaml）

```yaml
use_subproc_env: true              # 默认 true；false 回退线程版 VectorEnv
subproc_step_timeout_sec: 60.0     # 单 SubEnv step/reset 超时（秒）
subproc_max_respawns: 10           # 单 SubEnv 连续 hang 上限
on_subenv_timeout: truncate        # truncate | fail
chunk_step_timeout_sec: 0          # subproc 模式下禁用进程级 watchdog
```

### 超时恢复语义（`on_subenv_timeout: truncate`）

1. 父进程 `poll(timeout)` 超时 → `terminate/kill` 该 SubEnv 子进程
2. spawn 新子进程，`reset(env_seed=old_seed+1)`
3. 该 env 返回 `truncated=True, reward=0` + 有效 obs
4. 其他 SubEnv / Rollout / Actor 不受影响
5. 日志：`[SUBENV_RECOVER] sub_env=N elapsed=... action=kill+respawn seed=X->Y`

### 不影响的路径

- Libero / LiberoPlus（`env_type: libero`）
- RoboTwin 设 `use_subproc_env: false` 时回退 Stage-A 线程版 + watchdog

### 回滚 Stage-B

```bash
cd /mnt/pfs/7wsqem/grt/RLinf

# 方式 1：仅配置回退（保留代码）
# 在 robotwin yaml 中设 use_subproc_env: false, chunk_step_timeout_sec: 60.0

# 方式 2：完全回滚 Stage-B 代码
git checkout -- \
  rlinf/envs/robotwin/robotwin_env.py \
  rlinf/workers/env/env_worker.py \
  examples/embodiment/config/env/robotwin_open_microwave.yaml
rm -f rlinf/envs/robotwin/subproc_vector_env.py
rm -f tests/unit_tests/test_subproc_vector_env.py
```

---

## 2026-07-24：Stage-A 仅 RoboTwin 训练启用 log 监控 + watchdog 门控

### 目的

- PROGRESS / HEARTBEAT 日志与 `chunk_step` watchdog 原先对所有 `env_type` 生效，会误伤 LiberoPlus 等训练路径。
- 现改为：**仅 `env.train.env_type: robotwin` 且非 `only_eval` 时启用**。

### 新增文件

| 文件 | 说明 |
|------|------|
| `rlinf/utils/robotwin_hang_diagnostics.py` | 门控逻辑：`configure_from_cfg()` / `enabled()` |

### 修改文件

| 文件 | 改动 |
|------|------|
| `rlinf/utils/logging.py` | `log_progress` / `set_progress_state` / `start_progress_heartbeat` 在 `enabled()==False` 时为 no-op |
| `rlinf/workers/env/env_worker.py` | 启动时 `configure_from_cfg`；watchdog 仅 robotwin train 创建 |
| `rlinf/runners/embodied_runner.py` | 启动时 `configure_from_cfg`（Runner PROGRESS 日志） |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | 启动时 `configure_from_cfg` |
| `rlinf/workers/actor/fsdp_actor_worker.py` | `EmbodiedFSDPActor.__init__` 中 `configure_from_cfg` |
| `examples/embodiment/config/env/robotwin_open_microwave.yaml` | 文档化 `enable_hang_diagnostics` |

### 启用条件

同时满足：

1. `runner.only_eval != true`
2. `env.train.env_type == "robotwin"`
3. `env.train.enable_hang_diagnostics != false`（默认 true）

### 不影响的路径

- Libero / LiberoPlus（`env_type: libero`）
- ManiSkill、RoboCasa、RealWorld 等其它 env_type
- RoboTwin 纯 eval（`only_eval: true`）

### 配置项（RoboTwin env yaml）

```yaml
env:
  train:
    enable_hang_diagnostics: true   # false 则关闭 PROGRESS 日志与 watchdog
    chunk_step_timeout_sec: 60.0    # 0 则仅关闭 watchdog，日志仍可由上一项控制
```

### 回滚

```bash
cd /mnt/pfs/7wsqem/grt/RLinf
git checkout -- \
  rlinf/utils/logging.py \
  rlinf/workers/env/env_worker.py \
  rlinf/runners/embodied_runner.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  rlinf/workers/actor/fsdp_actor_worker.py \
  examples/embodiment/config/env/robotwin_open_microwave.yaml
rm -f rlinf/utils/robotwin_hang_diagnostics.py CHANGELOG_ROBOTWIN_HANG.md
```
