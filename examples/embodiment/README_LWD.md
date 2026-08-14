# RoboTwin LWD（DIVL + QAM）使用说明

本文档按**推荐执行顺序**说明：单卡冒烟 → 多卡正式训练，并解释相关 Hydra YAML 参数含义。

算法形态：共享 OpenPI π0.5（`pi05_robotwin_4task`）在多 task env 上 rollout；价值用 **DIVL**，flow 策略用 **QAM**；数据为 **B_off（demo）∪ B_on（online replay）** 混合采样。无 human intervention。

---

## 0. 文件索引

| 路径 | 作用 |
|------|------|
| `config/robotwin_lwd_openpi_pi05_1task.yaml` | N=1 冒烟 / 短 horizon（`place_mouse_pad`） |
| `config/robotwin_lwd_openpi_pi05_4task.yaml` | N=4 正式联合训练 |
| `config/env/robotwin_lwd_multitask.yaml` | 多任务 env 默认（稀疏奖励、统一 horizon） |
| `scripts/lwd_smoke_verify.py` | 均分检查 / per-task success 聚合 / 命令清单 |
| `scripts/convert_robotwin_lerobot_to_demo_buffer.py` | Clean/Randomized LeRobot → `demo_buffer`（DEMO_PATH） |
| `run_async.sh` | Async 训练入口（调用 `train_async.py`；日志固定写入 `RLinf/logs/`） |
| `rlinf/algorithms/divl.py` | DIVL 损失 |
| `rlinf/algorithms/qam.py` | QAM 损失 |
| `rlinf/workers/actor/async_fsdp_lwd_policy_worker.py` | Async LWD worker |

入口选择：`algorithm.loss_type: embodied_lwd` → `AsyncEmbodiedLWDFSDPPolicy`。

---

## 1. 你手里已有的东西 vs 还要准备什么（必读）

你现在通常有两类资产，**不要混用路径**：

| 你已有的 | 典型路径（本机示例） | 填到哪里 |
|----------|----------------------|----------|
| **SFT 训好的策略权重** | `/mnt/pfs/7wsqem/grt/openpi/checkpoints/pi05_robotwin_4task/pi05_robotwin_4task_torch` | `actor.model.model_path` / `rollout.model.model_path` |
| **SFT 用的原始演示数据（LeRobot）** | `/mnt/pfs/7wsqem/grt/starVLA/playground/Datasets/RoboTwin/Clean/<task>`（或 `Randomized/<task>`） | **不能**直接当 `DEMO_PATH` |

### `DEMO_PATH` / `algorithm.demo_buffer.load_path` 是什么？

它是 **B_off：已经转成 RLinf `TrajectoryReplayBuffer` checkpoint 的轨迹目录**，里面应有：

- `metadata.json`
- `trajectory_index.json`
- 若干 `*.pt` 轨迹文件  

用于 LWD offline / 混合采样的 **演示轨迹数据**，**不是** SFT 模型，也 **不是** 原始 LeRobot `Clean/<task>` 目录。

```text
Clean/place_mouse_pad/          ← LeRobot SFT 数据（parquet + mp4）
        │
        │  convert_robotwin_lerobot_to_demo_buffer.py
        ▼
data/lwd_demo_buffers/.../      ← DEMO_PATH（TrajectoryReplayBuffer）
        │
        │  algorithm.demo_buffer.load_path
        ▼
LWD 训练采样 B_off

pi05_robotwin_4task_torch/      ← SFT 模型
        │
        │  actor.model.model_path
        ▼
策略初始化 / rollout
```

### 如何从你的 Clean 数据生成 `DEMO_PATH`

在 RLinf 根目录、激活训练环境后：

```bash
conda activate rlinf-openpi-robotwin
cd /mnt/pfs/7wsqem/grt/RLinf
export PYTHONPATH=.

# —— 1-task 冒烟（place_mouse_pad）——
python examples/embodiment/scripts/convert_robotwin_lerobot_to_demo_buffer.py \
  --lerobot-root /mnt/pfs/7wsqem/grt/starVLA/playground/Datasets/RoboTwin/Clean \
  --tasks place_mouse_pad \
  --output /mnt/pfs/7wsqem/grt/RLinf/data/lwd_demo_buffers/place_mouse_pad \
  --action-chunk 50 \
  --image-size 224 \
  --max-episodes 50

# 之后：
export DEMO_PATH=/mnt/pfs/7wsqem/grt/RLinf/data/lwd_demo_buffers/place_mouse_pad

# —— 4-task 正式 ——
python examples/embodiment/scripts/convert_robotwin_lerobot_to_demo_buffer.py \
  --lerobot-root /mnt/pfs/7wsqem/grt/starVLA/playground/Datasets/RoboTwin/Clean \
  --tasks open_microwave,hanging_mug,place_mouse_pad,blocks_ranking_size \
  --output /mnt/pfs/7wsqem/grt/RLinf/data/lwd_demo_buffers/robotwin_4task \
  --action-chunk 50 \
  --image-size 224

export DEMO_PATH=/mnt/pfs/7wsqem/grt/RLinf/data/lwd_demo_buffers/robotwin_4task
```

说明：

- `--action-chunk` 应与 YAML 里 `actor.model.num_action_chunks`（RoboTwin LWD 默认 **50**，与 `pi05_robotwin_4task` / OpenPI `action_horizon` 一致）对齐。
- Clean 演示默认视为成功，最后一 chunk 写稀疏奖励 `1.0`。  
- 若要用 Randomized，把 `--lerobot-root` 改成 `.../RoboTwin/Randomized` 即可。  
- 转换完成后可用 `ls "$DEMO_PATH"` 确认存在 `metadata.json` 与 `trajectory_index.json`。

YAML 中模型路径示例（你已有）：

```yaml
actor:
  model:
    model_path: "/mnt/pfs/7wsqem/grt/openpi/checkpoints/pi05_robotwin_4task/pi05_robotwin_4task_torch"
```

---

## 2. 环境准备

在 **RLinf 仓库根目录**执行：

```bash
# 建议使用 RoboTwin 训练环境（示例）
conda activate rlinf-openpi-robotwin

cd /mnt/pfs/7wsqem/grt/RLinf

export EMBODIED_PATH="$(pwd)/examples/embodiment"
export REPO_PATH="$(pwd)"
export ROBOTWIN_PATH="/mnt/pfs/7wsqem/grt/RoboTwin"   # 按本机修改
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH}"
export ROBOT_PLATFORM=ALOHA   # RoboTwin / Aloha；run_async.sh 默认 LIBERO，务必覆盖

# 先完成第 1 节转换，再 export：
export DEMO_PATH=/mnt/pfs/7wsqem/grt/RLinf/data/lwd_demo_buffers/place_mouse_pad
```

### 日志目录（训练命令统一走 `run_async.sh`）

`run_async.sh` 会**强制覆盖** YAML 里的 `runner.logger.log_path`，每次运行写入：

```text
RLinf/logs/<YYYYMMDD-HH:MM:SS>-<config_name>/
├── run_embodiment.log    # 完整终端输出（tee）
├── metrics.log           # 训练指标表格（每步追加）
├── tensorboard/          # TensorBoard events + config.yaml
└── <experiment_name>/checkpoints/   # 若触发 save_interval
```

查看最新一次运行：

```bash
ls -lt /mnt/pfs/7wsqem/grt/RLinf/logs/ | head
tail -50 "$(ls -td /mnt/pfs/7wsqem/grt/RLinf/logs/*robotwin_lwd* | head -1)/metrics.log"
```

训练前请确认：

- `env.*.assets_path` → RoboTwin 仿真资源（与 SFT 数据目录不同）  
- `actor.model.model_path` → **SFT 模型**（`pi05_robotwin_4task_torch`）  
- `algorithm.demo_buffer.load_path` / `$DEMO_PATH` → **转换后的 demo_buffer 目录**（不是 Clean/<task>）

---

## 3. 单卡 / 无卡冒烟（不启完整训练或强制 1 GPU）

按顺序执行。第 3.1–3.2 **不需要 GPU**；第 3.3–3.4 为**强制单卡**短跑训练。**请先完成第 1 节的数据转换。**

### 3.1 算法单测（无 GPU）

```bash
cd /mnt/pfs/7wsqem/grt/RLinf
export PYTHONPATH=.

python -m pytest \
  tests/unit_tests/test_divl.py \
  tests/unit_tests/test_qam.py \
  tests/unit_tests/test_robotwin_multitask.py \
  -q
```

**期望：** 全部 PASSED（DIVL NLL/自适应 τ/chunk TD；QAM adjoint 与回归；task 均分逻辑）。

### 3.2 Env 均分检查（无 GPU、不启 SAPIEN）

```bash
python examples/embodiment/scripts/lwd_smoke_verify.py env-split --num-envs 32
python examples/embodiment/scripts/lwd_smoke_verify.py env-split --num-envs 8 --tasks place_mouse_pad
```

**期望：** N=4 时每个 task 恰好 8 个 env；N=1 时 8 个均属于 `place_mouse_pad`。

打印完整命令清单：

```bash
python examples/embodiment/scripts/lwd_smoke_verify.py checklist \
  --demo-path "${DEMO_PATH}"
```

### 3.3 单卡 Offline 冒烟（仅 B_off）

```bash
cd /mnt/pfs/7wsqem/grt/RLinf
export ROBOT_PLATFORM=ALOHA
# DEMO_PATH = 第 1 节转换输出，例如：
# export DEMO_PATH=/mnt/pfs/7wsqem/grt/RLinf/data/lwd_demo_buffers/place_mouse_pad

# 1task yaml 已默认 placement=GPU0；日志写入 RLinf/logs/<时间戳>-robotwin_lwd_openpi_pi05_1task/
CUDA_VISIBLE_DEVICES=0 bash examples/embodiment/run_async.sh \
  robotwin_lwd_openpi_pi05_1task \
  runner.lwd_stage=offline \
  algorithm.allow_demo_only=true \
  algorithm.demo_ratio=1.0 \
  algorithm.replay_buffer.min_buffer_size=0 \
  algorithm.demo_buffer.load_path="${DEMO_PATH}" \
  algorithm.demo_buffer.min_buffer_size=1 \
  runner.max_epochs=2 \
  algorithm.update_epoch=5 \
  env.train.total_num_envs=4 \
  env.eval.total_num_envs=4 \
  actor.global_batch_size=32 \
  actor.micro_batch_size=8
```

说明：

| 覆盖项 | 含义 |
|--------|------|
| （yaml 内 `env/rollout/actor: 0`） | 1task 配置默认单卡；勿用带逗号的合并键做 Hydra 覆盖 |
| `CUDA_VISIBLE_DEVICES=0` | 限制进程可见 GPU |
| `runner.lwd_stage=offline` | Offline 阶段语义（可配合 backbone 微调开关） |
| `allow_demo_only=true` + `demo_ratio=1.0` + `replay min=0` | 只从 `demo_buffer` 采样 |
| `demo_buffer.load_path` | **转换后的** TrajectoryReplayBuffer 目录（B_off） |

**期望：** 日志出现有限的 `lwd/value_loss`、`lwd/critic_loss`、`lwd/qam_loss`；无 intervention / HG-DAgger 依赖。

### 3.4 单卡 Online 短跑（B_off ∪ B_on）

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/embodiment/run_async.sh \
  robotwin_lwd_openpi_pi05_1task \
  runner.lwd_stage=online \
  algorithm.demo_ratio=0.5 \
  algorithm.allow_demo_only=false \
  algorithm.demo_buffer.load_path="${DEMO_PATH}" \
  runner.max_epochs=5 \
  runner.weight_sync_interval=2 \
  env.train.total_num_envs=4 \
  env.eval.total_num_envs=4 \
  actor.global_batch_size=32 \
  actor.micro_batch_size=8
```

**期望：** `replay_buffer/num_trajectories`（或等价 stats）随 rollout 增长；权重按 `weight_sync_interval` 同步；loss 保持有限。

---

## 4. 多卡正式训练

默认 YAML 使用：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    env, rollout, actor: all   # 本机全部可见 GPU，三组件共置
```

机器有 N 张卡且未改 placement 时，以 YAML 为准。LWD 配置已改为**分键**写法（Hydra 可覆盖）：

```yaml
component_placement:
  env: 0          # 或 0-7 / all
  rollout: 0
  actor: 0
```

Hydra 覆盖通过 `run_async.sh` 第三个参数起透传（不要再用 `env, rollout, actor` 合并键）：

```bash
bash examples/embodiment/run_async.sh robotwin_lwd_openpi_pi05_4task \
  cluster.component_placement.env=0-7 \
  cluster.component_placement.rollout=0-7 \
  cluster.component_placement.actor=0-7
```

> Hydra 无法可靠解析带逗号的 key（如 `env, rollout, actor`），会报 `LexerNoViableAltException`。

### 4.1 约束（必须满足）

1. `total_num_envs % len(task_names) == 0`  
2. 每个 actor/env rank 上的 `num_envs` 也需能被 N 整除（代码会 assert）  
3. `actor.global_batch_size % (micro_batch_size * world_size) == 0`  

示例（8 卡、N=4）：`total_num_envs=32` → 每 task 8 个 env；若 `world_size=8`、`stage_num=1`，则每 rank `num_envs=4`，仍被 N=4 整除。

### 4.2 8 卡 × 1-task 预热（可选）

```bash
cd /mnt/pfs/7wsqem/grt/RLinf
export ROBOT_PLATFORM=ALOHA
# DEMO_PATH 指向转换后的 buffer，例如 place_mouse_pad 或 4task 合并目录

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/embodiment/run_async.sh \
  robotwin_lwd_openpi_pi05_1task \
  cluster.component_placement.env=0-7 \
  cluster.component_placement.rollout=0-7 \
  cluster.component_placement.actor=0-7 \
  runner.lwd_stage=online \
  algorithm.demo_buffer.load_path="${DEMO_PATH}" \
  algorithm.demo_ratio=0.5 \
  runner.max_epochs=100 \
  env.train.total_num_envs=8 \
  env.eval.total_num_envs=8 \
  actor.global_batch_size=64 \
  actor.micro_batch_size=8
```

### 4.3 8 卡 × 4-task 正式训练

先用第 1 节生成四任务 `DEMO_PATH`，再：

```bash
export DEMO_PATH=/mnt/pfs/7wsqem/grt/RLinf/data/lwd_demo_buffers/robotwin_4task

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash examples/embodiment/run_async.sh \
  robotwin_lwd_openpi_pi05_4task \
  runner.lwd_stage=online \
  algorithm.demo_buffer.load_path="${DEMO_PATH}" \
  algorithm.demo_ratio=0.5 \
  runner.max_epochs=1000
```

（`robotwin_lwd_openpi_pi05_4task.yaml` 默认已是 `env/rollout/actor: 0-7`。）

默认已设：`total_num_envs=32`，四任务  
`open_microwave` / `hanging_mug` / `place_mouse_pad` / `blocks_ranking_size`，horizon 1600，稀疏成功奖励。

### 4.4 按 task 看 success

训练日志 / 自定义 JSONL 中应出现 `success_once/<task_name>` 与 `task_ids`。聚合：

```bash
python examples/embodiment/scripts/lwd_smoke_verify.py task-success \
  --log-json /path/to/metrics.jsonl
```

**期望：** 四个 task 都有 episode 计数；在 env 均分时各 task 样本量接近。

---

## 5. Offline → Online 推荐流水线

```text
[转换 Clean→demo_buffer] → [单测] → [env-split]
  → [单卡 offline] → [单卡 online] → [8 卡 1-task] → [8 卡 4-task]
```

| 阶段 | `runner.lwd_stage` | `demo_ratio` | `allow_demo_only` | replay `min_buffer_size` | 数据 |
|------|-------------------|--------------|-------------------|--------------------------|------|
| Offline | `offline` | `1.0` | `true` | `0` | 仅 B_off（转换后的 demo_buffer） |
| Online | `online` | `0.5`（默认） | `false` | `≥10` | B_off ∪ B_on |

Online 时：rollout 轨迹只写入 **replay_buffer（B_on）**；**不会**把 intervention 轨迹追加进 demo_buffer。

更新顺序（每个 `update_one_epoch`）：**V（DIVL）→ Q（DIVL）→ QAM（flow）**，无 SAC α。

---

## 6. YAML 参数说明

### 6.1 `cluster`

| 字段 | 含义 |
|------|------|
| `num_nodes` | 节点数；单机为 `1` |
| `component_placement` | 建议分键：`env` / `rollout` / `actor` 各自写 `0`、`0-7` 或 `all`。勿用 `env, rollout, actor` 合并键（Hydra CLI 会 Lexer 报错） |

### 6.2 `runner`

| 字段 | 含义 |
|------|------|
| `lwd_stage` | `offline` / `online`。影响是否允许仅 demo 采样，以及是否按 `finetune_backbone_offline` 微调 VLM |
| `max_epochs` | 外层训练轮数 |
| `max_steps` | `-1` 表示不按 step 截断 |
| `save_interval` | checkpoint 间隔（epoch） |
| `weight_sync_interval` | actor → rollout 权重同步间隔 |
| `val_check_interval` | `-1` 关闭定期 eval |
| `logger.*` | YAML 占位；实际目录由 `run_async.sh` 强制设为 `RLinf/logs/<时间戳>-<config>/` |

### 6.3 `algorithm`（LWD 核心）

| 字段 | 含义 |
|------|------|
| `loss_type` | 必须为 `embodied_lwd`，才会选 LWD async worker |
| `adv_type` | 占位/兼容，LWD 不走 PPO advantage |
| `update_epoch` | 每个 runner epoch 内调用 `update_one_epoch` 的次数 |
| `gamma` | 折扣；chunk bootstrap 用 `gamma ** num_action_chunks` |
| `tau` | target 网络 EMA 系数 |
| `demo_ratio` | 每个 batch 来自 demo 的比例；`0.5` = 一半 B_off 一半 B_on |
| `allow_demo_only` | `true` 时 replay 未就绪也可只采 demo（offline） |
| `target_update_freq` | 每多少次 update soft-update 一次 target |
| `target_update_type` | `all`：整网 EMA；`q_head_only`：仅 Q 相关 |

#### `algorithm.lwd`

| 字段 | 含义 |
|------|------|
| `tau_base` / `tau_min` / `tau_max` | 自适应分位数 τ 的中心与上下界 |
| `alpha` | 由 V 归一化熵调节 τ 的强度（`entropy_alpha`） |
| `num_atoms` | categorical V 的原子数（C51 风格） |
| `v_min` / `v_max` | 价值支撑区间（稀疏成功奖励建议 `[0,1]`） |
| `qam_lambda` | QAM 终端 adjoint：`g̃_1 = -∇_a Q / λ` |
| `agg_q` | 多头 Q 聚合：`min` / `mean` |
| `finetune_backbone_offline` | offline 时是否允许微调 VLM backbone；online 默认冻 backbone，只训 action expert + V/Q |

#### `algorithm.demo_buffer` / `replay_buffer`

| 字段 | 含义 |
|------|------|
| `load_path` | **TrajectoryReplayBuffer 目录**（需含 `metadata.json`）。**不要**填 LeRobot `Clean/<task>`，也不要填 SFT 模型目录。用第 1 节脚本转换后再填 |
| `min_buffer_size` | 开始采样前的最小轨迹/样本门槛 |
| `enable_cache` / `cache_size` | 内存缓存 |
| `sample_window_size` | 采样窗口 |
| `auto_save` | 是否自动落盘 buffer |

### 6.4 `env.train` / `env.eval`

| 字段 | 含义 |
|------|------|
| `task_names` | 任务列表；长度 N。`env_id % N` 均分。N=1 或 N=4 为实践保证范围 |
| `total_num_envs` | 并行 env 总数，**必须能被 N 整除** |
| `use_custom_reward` | `true`：用 `reward_coef * success` |
| `use_dense_reward` | LWD 应 `false`（稀疏成功） |
| `use_rel_reward` | LWD 建议 `false`（非相对差分） |
| `max_episode_steps` | episode 截断步数；4-task 统一 1600 |
| `max_steps_per_rollout_epoch` | 每个 rollout epoch 的步数预算 |
| `assets_path` | RoboTwin **仿真资源**根目录（不是 starVLA Datasets） |
| `use_subproc_env` | 多任务必须为 `true`（每 SubEnv 独立 `task_name`） |
| `task_config.step_lim` | 底层 task 步数上限，宜与 `max_episode_steps` 对齐 |
| `task_config.task_name` | 占位；真实任务名来自 `task_names[env_id % N]` |

指标：`infos["episode"]["success_once/<task>"]`、`task_ids`，避免 N>1 时被全局平均掩盖。

### 6.5 `rollout`

| 字段 | 含义 |
|------|------|
| `collect_transitions` | Async SAC/LWD 需 `true`，写入 transition/replay |
| `pipeline_stage_num` | 流水线 stage 数；影响每 rank env 切分 |
| `generation_backend` | `huggingface` 本地策略推理 |
| `model.model_path` | 一般与 actor 相同（**SFT 模型**） |

### 6.6 `actor`

| 字段 | 含义 |
|------|------|
| `global_batch_size` | 全局 batch；需整除 `micro_batch_size * world_size` |
| `micro_batch_size` | 微批，用于梯度累积 |
| `model.model_path` | **SFT 训好的 OpenPI 权重目录**（如 `pi05_robotwin_4task_torch`） |
| `model.num_action_chunks` | action chunk 长度 H（与 bootstrap `γ^H`、reward 聚合、转换脚本 `--action-chunk` 对齐） |
| `model.action_dim` | 环境动作维（RoboTwin Aloha=14） |
| `model.num_steps` | flow 去噪步数 |
| `optim` / `critic_optim` / `value_optim` | 策略（QAM）/ Q / V 三套优化器 |
| `fsdp_config.sharding_strategy` | 常用 `no_shard`（与现有 OpenPI SAC 一致） |

#### `actor.model.openpi`（LWD）

| 字段 | 含义 |
|------|------|
| `config_name` | `pi05_robotwin_4task` |
| `use_lwd` | 挂载 DIVL V/Q 头与 QAM 路径；**必须 true** |
| `use_dsrl` | 必须 `false`（LWD 不用 DSRL noise-Q） |
| `train_expert_only` | 加载时冻结 VLM、只训 expert（与 online 策略一致） |
| `lwd_state_dim` | proprio 维，RoboTwin=14 |
| `lwd_num_q_heads` | Double Q 头数（默认 2） |
| `lwd_num_atoms` / `lwd_v_min` / `lwd_v_max` | categorical V，应与 `algorithm.lwd` 一致 |
| `lwd_*_latent_dim` / `lwd_hidden_dims` | 轻量 encoder / MLP 宽度 |
| `lwd_agg_q` | 模型侧默认聚合方式 |
| `num_images_in_input` | RoboTwin 头+双腕 → `3` |
| `action_env_dim` | 与 `action_dim` 对齐（14） |

---

## 7. 常见问题

**Q: `DEMO_PATH` 填 Clean 数据行不行？**  
不行。Clean 是 LeRobot（`data/*.parquet` + `videos/*.mp4`）。必须先跑  
`scripts/convert_robotwin_lerobot_to_demo_buffer.py`，把输出目录当作 `DEMO_PATH`。

**Q: `DEMO_PATH` 填 SFT 模型目录行不行？**  
不行。模型填 `actor.model.model_path`（例如 `.../pi05_robotwin_4task_torch`）。

**Q: 默认到底几卡？**  
看 YAML：`robotwin_lwd_openpi_pi05_1task` 默认 `env/rollout/actor: 0`（单卡）；`..._4task` 默认 `0-7`（八卡）。  
CLI 改卡数用分键覆盖，例如 `cluster.component_placement.env=0-7`（三组件都要写）。

**Q: Hydra `LexerNoViableAltException` 且指向 `env, rollout`？**  
不要覆盖合并键 `env, rollout, actor`。改 YAML，或分三次覆盖 `cluster.component_placement.env=...` 等。

**Q: `total_num_envs must be divisible by the number of environment processes`？**  
`total_num_envs` 必须能被 `env_world_size`（placement 占用的 GPU 数）整除。单卡时 `env_world_size=1`，`total_num_envs=4` 即可；八卡时用 `8/16/32` 等。

**Q: 日志写在哪？**  
训练请用 `run_async.sh`：日志固定落在 `RLinf/logs/<时间戳>-<config>/`，含 `run_embodiment.log`、`metrics.log`、`tensorboard/`。YAML 里的 `runner.logger.log_path` 会被脚本覆盖。

**Q: `run_async.sh` 怎么用 Hydra 覆盖？**  
用法：`bash run_async.sh <config> [ROBOT_PLATFORM] key=value ...`。第二个参数若不含 `=` 则当作 `ROBOT_PLATFORM`；其余参数原样透传给 Hydra。

**Q: `total_num_envs` 报 assert？**  
检查 `total_num_envs % N == 0`，以及 `total_num_envs / (world_size * pipeline_stage_num) % N == 0`。

**Q: Offline 一直等 buffer？**  
确认 `demo_buffer.load_path` 指向**转换后的**目录且含 `metadata.json`，并设  
`allow_demo_only=true`、`demo_ratio=1.0`、`replay_buffer.min_buffer_size=0`。

**Q: 指标看不清多任务？**  
看 `success_once/<task_name>`，或用 `lwd_smoke_verify.py task-success`；不要只看全局 `success_once` 平均。

---

## 8. 快速对照

| 场景 | Config | Placement | 关键覆盖 |
|------|--------|-----------|---------|
| 转换 demo | — | 无卡 | `convert_robotwin_lerobot_to_demo_buffer.py` |
| 单测 | — | 无卡 | `pytest ...` |
| Env 均分 | — | 无卡 | `lwd_smoke_verify.py env-split` |
| 单卡 offline | `..._1task` | `0` | `lwd_stage=offline`, `demo_ratio=1.0`, `allow_demo_only=true`, `load_path=$DEMO_PATH` |
| 单卡 online | `..._1task` | `0` | `lwd_stage=online`, `demo_ratio=0.5`, `load_path=$DEMO_PATH` |
| 8 卡 1-task | `..._1task` | `0-7` | `demo_buffer.load_path=$DEMO_PATH` |
| 8 卡 4-task | `..._4task` | `0-7` | 同上；`total_num_envs=32` |
