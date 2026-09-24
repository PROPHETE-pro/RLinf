RL with RoboDojo Benchmark
==========================

`RoboDojo <https://robodojo-benchmark.com/doc/>`__ is an Isaac Sim / IsaacLab
bimanual benchmark (dual ARX X5). RLinf wraps it as ``env_type: robodojo`` for
OpenDM post-training RL. Isaac Kit runs in a **separate RoboDojo conda
process**; the RLinf actor / rollout stay on the OpenDM interpreter.

This path does **not** change the existing RoboTwin configs or
``rlinf/envs/robotwin/``.

Environment
-----------

- **Simulator:** NVIDIA Isaac Sim 5.1 + IsaacLab 2.3.2 (conda env ``RoboDojo``, Python 3.11)
- **Robot:** dual ARX X5, 14-D absolute joint action
- **Observation:** head + left/right wrist RGB, proprioception ``[B, 14]``, language instruction
- **Tasks:** 51 dual_x5 tasks via ``env.train.task_config.task_name``; skip ``imitate_sorting_sequence``, ``make_kong``, ``play_tic_tac_toe`` (Franka competition embodiment)

Quick Start
-----------

Prerequisites: a local RoboDojo checkout with assets, conda env ``RoboDojo``,
and an OpenDM checkpoint at ``opendm/checkpoints/dm05_robodojo``.

Isaac worker smoke (no RLinf actor)::

   bash examples/embodiment/run_robodojo.sh --doctor
   bash examples/embodiment/run_robodojo.sh --worker-smoke

OpenDM PPO on ``stack_bowls``::

   bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05

1-epoch PPO smoke::

   bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05 \
     runner.max_epochs=1

Switch task without writing a new YAML::

   bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05 \
     env.train.task_config.task_name=hang_mugs \
     env.train.max_episode_steps=800 \
     env.eval.task_config.task_name=hang_mugs \
     env.eval.max_episode_steps=800

List task horizons / OpenDM support::

   python toolkits/robodojo/sync_task_horizon.py
   python toolkits/robodojo/sync_task_horizon.py --opendm-only

Optional 4-task multitask env (requires ``num_envs`` divisible by 4)::

   bash examples/embodiment/run_robodojo.sh robodojo_stack_bowls_ppo_opendm_dm05 \
     env.train.task_names=[stack_bowls,hang_mugs,push_T,align_blocks] \
     env.train.total_num_envs=4

OpenDM-supported dual_x5 tasks (51)
-----------------------------------

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
``swap_blocks``, ``sweep_blocks``, ``sweep_blocks_random``.

Unsupported Franka competition tasks: ``imitate_sorting_sequence``,
``make_kong``, ``play_tic_tac_toe``.

Configuration
-------------

- Env defaults: ``examples/embodiment/config/env/robodojo.yaml``
- PPO recipe: ``examples/embodiment/config/robodojo_stack_bowls_ppo_opendm_dm05.yaml``
- OpenDM ``robot_type`` must be ``Dual ARX5`` (not ``Aloha RoboTwin2``)
- Checkpoint: ``/kpfs_ssd/data/ruitong_gan/opendm/checkpoints/dm05_robodojo``
- Launch exports ``ROBODOJO_PATH`` / ``ROBODOJO_PYTHON``; Isaac subprocesses use that interpreter

Each Isaac process hosts **one** sim env. ``total_num_envs`` scales by spawning
more subprocesses (typically one GPU per Isaac process).

Evaluation
----------

RLinf eval::

   bash evaluations/run_eval.sh robodojo robodojo_stack_bowls_opendm_dm05_eval

Official RoboDojo eval (unchanged, for physics/obs parity)::

   bash opendm/script/run_robodojo_eval.sh eval --task stack_bowls
