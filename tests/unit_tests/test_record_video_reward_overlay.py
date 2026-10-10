from types import SimpleNamespace

import numpy as np

from rlinf.envs.wrappers.record_video import RecordVideo


def _wrapper() -> RecordVideo:
    env = SimpleNamespace(seed=0, num_envs=1)
    return RecordVideo(env, {"fps": 25, "info_on_video": False})


def test_extra_policy_frame_reuses_last_reward_scalar():
    video = _wrapper()
    rewards = np.array([[0.1, 0.2]], dtype=np.float32)
    info = video._build_info_item(None, rewards, None, env_id=0, time_idx=2)
    assert info["reward"] == np.float32(0.2)
    video._executor.shutdown(wait=False)


def test_reward_strip_frames_replace_smaller_reset_frame():
    video = _wrapper()
    video.render_images.append(np.zeros((4, 4, 3), dtype=np.uint8))
    taller = np.zeros((8, 4, 3), dtype=np.uint8)
    video._append_compatible_frame(taller)
    assert len(video.render_images) == 1
    assert video.render_images[0].shape == (8, 4, 3)
    raw_policy = np.zeros((4, 4, 3), dtype=np.uint8)
    video._append_compatible_frame(raw_policy)
    assert len(video.render_images) == 1
    video._executor.shutdown(wait=False)
