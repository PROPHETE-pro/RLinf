import numpy as np

from rlinf.envs.robodojo.reward_strip import compose_frame_with_strip, render_reward_strip
from rlinf.envs.robodojo.robodojo_env import RoboDojoEnv


def test_reward_strip_matches_frame_width_and_stacks_below():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    strip = render_reward_strip(
        frame.shape[1],
        horizon=2100,
        scores=[0.1, 0.4],
        chunk_spans=[(0, 2)],
        cursor=2,
        phi=0.8,
    )
    assert strip.shape == (176, 64, 3)
    composed = compose_frame_with_strip(frame, strip)
    assert composed.shape == (48 + 176, 64, 3)
    assert composed.dtype == np.uint8


class _VideoCfg:
    def get(self, key, default=None):
        return getattr(self, key, default)

    video_cfg = {"save_video": True}
    max_episode_steps = 1000


def test_head_video_stacks_missing_and_odd_sized_frames():
    env = RoboDojoEnv.__new__(RoboDojoEnv)
    env.center_crop = False
    env.num_envs = 3
    env.per_env_task_names = ["insert_tubes"] * 3
    env._video_scores = [[], [], []]
    env._video_phis = [[], [], []]
    env._video_chunk_spans = [[], [], []]
    env.cfg = _VideoCfg()
    frames = [
        [np.zeros((32, 48, 3), dtype=np.uint8) for _ in range(2)],
        [],
        [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)],
    ]
    extracted = {"main_images": np.zeros((3, 8, 8, 3), dtype=np.uint8)}
    sequence = env._head_video_obs_list(frames, extracted, infos={}, terminations=None)
    assert sequence[-1] is extracted
    stacked = [item["main_images"] for item in sequence[:-1]]
    assert len(stacked) == 2
    assert stacked[0].shape == stacked[1].shape
    assert stacked[0].shape[0] == 3
    assert stacked[0].dtype == np.uint8
