RoboDojo Evaluation
===================

RoboDojo is an Isaac Sim dual-arm benchmark. RLinf evaluates OpenDM on a
single-task env worker (one Isaac process per env) and reports
``eval/success_once``.

Related training doc: :doc:`../../examples/embodied/robodojo`

Official RoboDojo eval (WebSocket policy server) is unchanged; use it as a
physics/observation baseline:

.. code-block:: bash

   bash opendm/script/run_robodojo_eval.sh eval --task stack_bowls

RLinf evaluation
----------------

.. code-block:: bash

   export ROBODOJO_PATH=/kpfs_ssd/data/ruitong_gan/RoboDojo
   export ROBODOJO_PYTHON=/kpfs_ssd/data/ruitong_gan/miniconda/envs/RoboDojo/bin/python
   bash evaluations/run_eval.sh robodojo robodojo_stack_bowls_opendm_dm05_eval

Override the task with Hydra:

.. code-block:: bash

   bash evaluations/run_eval.sh robodojo robodojo_stack_bowls_opendm_dm05_eval \
     env.eval.task_config.task_name=hang_mugs \
     env.eval.max_episode_steps=800

Environment check (no Isaac)::

   bash examples/embodiment/run_robodojo.sh --doctor
