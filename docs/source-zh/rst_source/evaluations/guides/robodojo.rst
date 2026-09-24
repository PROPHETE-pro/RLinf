RoboDojo 评测
=============

RoboDojo 是基于 Isaac Sim 的双臂基准。RLinf 在单任务 env worker（每个 env 一个
Isaac 进程）上评测 OpenDM，并报告 ``eval/success_once``。

相关训练文档：:doc:`../../examples/embodied/robodojo`

官方 RoboDojo 评测（WebSocket 策略服务）保持不变，用作物理/观测对照：

.. code-block:: bash

   bash opendm/script/run_robodojo_eval.sh eval --task stack_bowls

RLinf 评测
----------

.. code-block:: bash

   export ROBODOJO_PATH=/kpfs_ssd/data/ruitong_gan/RoboDojo
   export ROBODOJO_PYTHON=/kpfs_ssd/data/ruitong_gan/miniconda/envs/RoboDojo/bin/python
   bash evaluations/run_eval.sh robodojo robodojo_stack_bowls_opendm_dm05_eval

用 Hydra 覆盖任务：

.. code-block:: bash

   bash evaluations/run_eval.sh robodojo robodojo_stack_bowls_opendm_dm05_eval \
     env.eval.task_config.task_name=hang_mugs \
     env.eval.max_episode_steps=800

环境检查（不启动 Isaac）::

   bash examples/embodiment/run_robodojo.sh --doctor
