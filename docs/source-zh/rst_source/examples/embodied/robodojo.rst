基于 RoboDojo 的强化学习训练
========================================

`RoboDojo <https://robodojo-benchmark.com/doc/>`__ 是基于 Isaac Sim / IsaacLab
的双臂基准（dual ARX X5）。RLinf 以 ``env_type: robodojo`` 接入，用于 OpenDM
后训练 RL。Isaac Kit 运行在 **独立的 RoboDojo conda 进程** 中；RLinf 的
actor / rollout 仍使用 OpenDM 解释器。

该路径 **不会** 修改现有 RoboTwin 配置或 ``rlinf/envs/robotwin/``。

环境
----------------------------------------

- **仿真：** NVIDIA Isaac Sim 5.1 + IsaacLab 2.3.2（conda ``RoboDojo``，Python 3.11）
- **机器人：** dual ARX X5，14 维绝对关节动作
- **观测：** 头部 + 左右腕部 RGB、本体觉 ``[B, 14]``、语言指令
- **任务：** 通过 ``env.train.task_config.task_name`` 覆盖 51 个 dual_x5 任务；跳过 ``imitate_sorting_sequence``、``make_kong``、``play_tic_tac_toe``（Franka competition 本体）

快速开始
----------------------------------------

前置：本地 RoboDojo 仓库与资产、conda 环境 ``RoboDojo``，以及
``opendm/checkpoints/dm05_robodojo`` 权重。

仅 Isaac worker 冒烟（不启动 RLinf actor）::

   bash examples/embodiment/run_robodojo.sh --doctor
   bash examples/embodiment/run_robodojo.sh --worker-smoke

在 ``stack_bowls`` 上跑 OpenDM PPO::

   bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05

1 epoch PPO 冒烟::

   bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05 \
     runner.max_epochs=1

切换任务，无需新写 YAML::

   bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05 \
     env.train.task_config.task_name=hang_mugs \
     env.train.max_episode_steps=800 \
     env.eval.task_config.task_name=hang_mugs \
     env.eval.max_episode_steps=800

列出任务步数上限 / OpenDM 支持情况::

   python toolkits/robodojo/sync_task_horizon.py
   python toolkits/robodojo/sync_task_horizon.py --opendm-only

可选 4 任务多任务（``num_envs`` 必须能被任务数整除）::

   bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05 \
     env.train.task_names=[stack_bowls,hang_mugs,push_T,align_blocks] \
     env.train.total_num_envs=4

OpenDM 可训 dual_x5 任务（51）
----------------------------------------

``align_blocks``, ``arrange_largest_number``, ``arrange_largest_number_random``,
``build_tower``, ``classify_objects``, ``classify_objects_by_language``,
``cover_blocks``, ``deposit_coin``, ``fasten_screws``, ``fill_egg_holder``,
``fill_pen_holder``, ``fold_clothes``, ``fold_clothes_random``, ``general_pickup``,
``hang_mugs``, ``hang_mugs_random``, ``insert_key``, ``insert_tubes``,
``make_toast``, ``make_toast_random``, ``match_and_pick_from_conveyor``,
``organize_table``, ``pack_objects_into_box``, ``pack_objects_into_box_random``,
``pick_from_conveyor_by_image``, ``play_Xylophone``, ``play_stacking_toy``,
``plug_in_charger``, ``pour_balls_into_vase``, ``pour_by_language``,
``pour_liquid_into_cup``, ``pour_liquid_into_cup_random``, ``press_by_number``,
``push_T``, ``push_T_random``, ``put_bottles_into_dustbin``, ``solve_equation``,
``sort_nesting_dolls_by_size``, ``sort_nesting_dolls_by_size_random``,
``stack_blocks``, ``stack_blocks_by_language``, ``stack_blocks_random``,
``stack_bowls``, ``stack_bowls_random``, ``store_laptop_and_headphones``,
``store_laptop_and_headphones_random``, ``store_tools_in_toolbox``, ``swap_T``,
``swap_blocks``, ``sweep_blocks``, ``sweep_blocks_random``。

不支持的 Franka competition 任务：``imitate_sorting_sequence``、
``make_kong``、``play_tic_tac_toe``。

配置
----------------------------------------

- 环境默认：``examples/embodiment/config/env/robodojo.yaml``
- PPO 配方：``examples/embodiment/config/robodojo_stack_bowls_ppo_opendm_dm05.yaml``
- OpenDM ``robot_type`` 必须为 ``Dual ARX5``（不是 ``Aloha RoboTwin2``）
- 权重：``/kpfs_ssd/data/ruitong_gan/opendm/checkpoints/dm05_robodojo``
- 启动脚本导出 ``ROBODOJO_PATH`` / ``ROBODOJO_PYTHON``；Isaac 子进程使用该解释器

每个 Isaac 进程只承载 **1** 个仿真 env。``total_num_envs`` 通过更多子进程扩展（通常一卡一个 Isaac 进程）。

评测
----------------------------------------

RLinf 评测::

   bash evaluations/run_eval.sh robodojo robodojo_stack_bowls_opendm_dm05_eval

官方 RoboDojo 评测（保持不变，用于对照物理/观测语义）::

   bash opendm/script/run_robodojo_eval.sh eval --task stack_bowls
