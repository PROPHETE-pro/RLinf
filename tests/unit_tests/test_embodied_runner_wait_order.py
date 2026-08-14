"""Regression: generate_rollouts must wait rollout/env before actor recv."""

from pathlib import Path


def _run_block(source: str, marker: str) -> str:
    start = source.index(marker)
    # Stop at the next top-level timer block after generate_rollouts.
    end = source.index('with self.timer("cal_adv_and_returns")', start)
    return source[start:end]


def test_run_generate_rollouts_wait_order():
    path = Path(__file__).resolve().parents[2] / "rlinf/runners/embodied_runner.py"
    block = _run_block(path.read_text(), 'with self.timer("generate_rollouts"):')

    rollout_pos = block.index("rollout_handle.wait()")
    env_pos = block.index("env_handle.wait()")
    actor_pos = block.index("recv_rollout_trajectories(")

    assert rollout_pos < env_pos < actor_pos, (
        "Expected rollout.wait -> env.wait -> actor.recv in run() generate_rollouts"
    )


def test_run_pipeline_waits_for_env_interact():
    path = Path(__file__).resolve().parents[2] / "rlinf/runners/embodied_runner.py"
    source = path.read_text()
    start = source.index("def run_pipeline(")
    end = source.index("def _save_checkpoint(", start)
    block = source[start:end]

    pipeline_gen = block.index('with self.timer("generate_rollouts"):')
    segment = block[pipeline_gen : pipeline_gen + 400]
    rollout_pos = segment.index("rollout_handle.wait()")
    env_pos = segment.index("env_handle.wait()")

    assert rollout_pos < env_pos, (
        "Expected rollout.wait -> env.wait in run_pipeline() generate_rollouts"
    )
