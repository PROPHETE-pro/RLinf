#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
"""Convert RoboTwin LeRobot Clean/Randomized datasets → LWD demo_buffer.

Your SFT data under starVLA/.../RoboTwin/Clean/<task> is **LeRobot v2.1**,
NOT what ``algorithm.demo_buffer.load_path`` accepts.

This script writes a ``TrajectoryReplayBuffer`` checkpoint directory that
LWD can load via ``demo_buffer.load_path``.

Example (1-task smoke)::

    python examples/embodiment/scripts/convert_robotwin_lerobot_to_demo_buffer.py \\
      --lerobot-root /mnt/pfs/7wsqem/grt/starVLA/playground/Datasets/RoboTwin/Clean \\
      --tasks place_mouse_pad \\
      --output /mnt/pfs/7wsqem/grt/RLinf/data/lwd_demo_buffers/place_mouse_pad \\
      --action-chunk 50 \\
      --image-size 224 \\
      --max-episodes 50

Example (4-task)::

    python examples/embodiment/scripts/convert_robotwin_lerobot_to_demo_buffer.py \\
      --lerobot-root /mnt/pfs/7wsqem/grt/starVLA/playground/Datasets/RoboTwin/Clean \\
      --tasks open_microwave,hanging_mug,place_mouse_pad,blocks_ranking_size \\
      --output /mnt/pfs/7wsqem/grt/RLinf/data/lwd_demo_buffers/robotwin_4task \\
      --action-chunk 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer


CAM_HIGH = "observation.images.cam_high"
CAM_LEFT = "observation.images.cam_left_wrist"
CAM_RIGHT = "observation.images.cam_right_wrist"


def _load_video_frames_rgb(video_path: Path) -> list[np.ndarray]:
    """Decode mp4 to list of HxWxC uint8 RGB frames (via PyAV)."""
    import av

    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def _resize_nhwc(frames: list[np.ndarray], size: int) -> torch.Tensor:
    """[T,H,W,C] uint8 tensor, resized to size×size."""
    out = []
    for fr in frames:
        img = Image.fromarray(fr).convert("RGB").resize((size, size), Image.BILINEAR)
        out.append(np.asarray(img, dtype=np.uint8))
    return torch.from_numpy(np.stack(out, axis=0))  # [T,H,W,C]


def _load_task_prompt(task_dir: Path, task_index: int) -> str:
    tasks_path = task_dir / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return task_dir.name
    for line in tasks_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("task_index", -1)) == int(task_index):
            return str(row.get("task", task_dir.name))
    return task_dir.name


def _episode_to_trajectory(
    *,
    task_dir: Path,
    episode_index: int,
    action_chunk: int,
    image_size: int,
    success_reward: float,
) -> Trajectory | None:
    pq = task_dir / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not pq.exists():
        # Some dumps use multi-chunk layouts; search.
        matches = list(task_dir.glob(f"data/**/episode_{episode_index:06d}.parquet"))
        if not matches:
            return None
        pq = matches[0]

    df = pd.read_parquet(pq)
    T = len(df)
    if T < 2:
        return None

    states = np.stack(df["observation.state"].to_list()).astype(np.float32)  # [T,14]
    actions = np.stack(df["action"].to_list()).astype(np.float32)  # [T,14]
    task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0
    prompt = _load_task_prompt(task_dir, task_index)

    # Videos live under videos/chunk-XXX/<cam>/episode_XXXXXX.mp4
    chunk_name = pq.parent.name  # chunk-000
    high_path = task_dir / "videos" / chunk_name / CAM_HIGH / f"episode_{episode_index:06d}.mp4"
    left_path = task_dir / "videos" / chunk_name / CAM_LEFT / f"episode_{episode_index:06d}.mp4"
    right_path = task_dir / "videos" / chunk_name / CAM_RIGHT / f"episode_{episode_index:06d}.mp4"
    for p in (high_path, left_path, right_path):
        if not p.exists():
            print(f"[skip] missing video {p}", file=sys.stderr)
            return None

    high = _resize_nhwc(_load_video_frames_rgb(high_path), image_size)
    left = _resize_nhwc(_load_video_frames_rgb(left_path), image_size)
    right = _resize_nhwc(_load_video_frames_rgb(right_path), image_size)
    n = min(T, high.shape[0], left.shape[0], right.shape[0])
    if n < action_chunk + 1:
        return None
    states = states[:n]
    actions = actions[:n]
    high, left, right = high[:n], left[:n], right[:n]
    wrist = torch.stack([left, right], dim=1)  # [T,2,H,W,C]

    # Non-overlapping action chunks; last incomplete chunk dropped.
    num_chunks = (n - 1) // action_chunk
    if num_chunks <= 0:
        return None

    curr_main, next_main = [], []
    curr_wrist, next_wrist = [], []
    curr_state, next_state = [], []
    act_list, rew_list = [], []
    term_list, trunc_list, done_list = [], [], []
    intervene = []

    for i in range(num_chunks):
        t0 = i * action_chunk
        t1 = t0 + action_chunk  # action indices [t0, t1)
        # Obs at chunk start / after executing the chunk.
        curr_main.append(high[t0])
        next_main.append(high[t1])
        curr_wrist.append(wrist[t0])
        next_wrist.append(wrist[t1])
        curr_state.append(torch.from_numpy(states[t0]))
        next_state.append(torch.from_numpy(states[t1]))
        act_list.append(torch.from_numpy(actions[t0:t1]))  # [H,14]

        is_last = i == num_chunks - 1
        # Sparse success: Clean demos are successful → reward on last chunk only.
        reward = torch.zeros(action_chunk, dtype=torch.float32)
        terminations = torch.zeros(action_chunk, dtype=torch.bool)
        truncations = torch.zeros(action_chunk, dtype=torch.bool)
        if is_last:
            reward[-1] = float(success_reward)
            terminations[-1] = True
        rew_list.append(reward)
        term_list.append(terminations)
        trunc_list.append(truncations)
        done_list.append(terminations | truncations)
        intervene.append(
            torch.zeros(action_chunk * actions.shape[1], dtype=torch.bool)
        )

    def _stack_env(xs: list[torch.Tensor]) -> torch.Tensor:
        # [traj_len, B=1, ...]
        return torch.stack(xs, dim=0).unsqueeze(1).contiguous()

    traj = Trajectory(
        max_episode_length=n,
        model_weights_id="robotwin_lerobot_demo",
    )
    # Match online rollout layout [T, B, H*action_dim] (not [T, B, H, D]).
    stacked_actions = _stack_env(act_list)  # [T,1,H,14]
    traj.actions = stacked_actions.reshape(
        stacked_actions.shape[0], stacked_actions.shape[1], -1
    )
    traj.rewards = _stack_env(rew_list)
    traj.terminations = _stack_env(term_list)
    traj.truncations = _stack_env(trunc_list)
    traj.dones = _stack_env(done_list)
    traj.intervene_flags = _stack_env(intervene)
    # Dummy placeholders used by buffer shape logic in some paths
    traj.prev_logprobs = torch.zeros(
        traj.actions.shape[0], 1, action_chunk, dtype=torch.float32
    )
    traj.prev_values = torch.zeros(traj.actions.shape[0], 1, dtype=torch.float32)
    traj.versions = torch.zeros(traj.actions.shape[0], 1, dtype=torch.long)

    traj.curr_obs = {
        "main_images": _stack_env(curr_main),  # [T,1,H,W,C]
        "wrist_images": _stack_env(curr_wrist),  # [T,1,2,H,W,C]
        "states": _stack_env(curr_state),  # [T,1,14]
    }
    traj.next_obs = {
        "main_images": _stack_env(next_main),
        "wrist_images": _stack_env(next_wrist),
        "states": _stack_env(next_state),
    }
    # Keep language out of tensors; OpenPI obs_processor reads task_descriptions
    # from live env. Demo QAM/DIVL critic path uses images+states.
    _ = prompt  # reserved for future language fields in buffer
    return traj


def convert(
    lerobot_root: Path,
    tasks: list[str],
    output: Path,
    action_chunk: int,
    image_size: int,
    max_episodes: int | None,
    success_reward: float,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    buffer = TrajectoryReplayBuffer(
        seed=0,
        enable_cache=True,
        cache_size=100000,
        sample_window_size=100000,
        auto_save=False,
        auto_save_path=str(output),
        trajectory_format="pt",
    )

    total = 0
    for task in tasks:
        task_dir = lerobot_root / task
        if not task_dir.exists():
            raise FileNotFoundError(f"Task dir not found: {task_dir}")
        eps = sorted(task_dir.glob("data/**/episode_*.parquet"))
        ep_ids = sorted(
            {int(p.stem.split("_")[-1]) for p in eps}
        )
        if max_episodes is not None:
            ep_ids = ep_ids[: max_episodes]
        print(f"[{task}] converting {len(ep_ids)} episodes → chunks of {action_chunk}")
        for ep_id in ep_ids:
            traj = _episode_to_trajectory(
                task_dir=task_dir,
                episode_index=ep_id,
                action_chunk=action_chunk,
                image_size=image_size,
                success_reward=success_reward,
            )
            if traj is None:
                print(f"  skip episode {ep_id}")
                continue
            buffer.add_trajectories([traj])
            total += 1
            if total % 5 == 0:
                print(f"  added {total} trajectories (last ep={ep_id}, chunks={traj.actions.shape[0]})")

    if total == 0:
        raise RuntimeError("No trajectories converted; check paths / videos.")

    buffer.save_checkpoint(str(output))
    meta = {
        "source_lerobot_root": str(lerobot_root),
        "tasks": tasks,
        "action_chunk": action_chunk,
        "image_size": image_size,
        "num_trajectories": total,
        "note": "Use this directory as algorithm.demo_buffer.load_path",
    }
    (output / "conversion_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Wrote {total} trajectories to {output}")
    print("Set: algorithm.demo_buffer.load_path=" + str(output))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        required=True,
        help="Parent of task folders, e.g. .../RoboTwin/Clean",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated task names under lerobot-root",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-chunk", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--success-reward", type=float, default=1.0)
    args = parser.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    convert(
        lerobot_root=args.lerobot_root,
        tasks=tasks,
        output=args.output,
        action_chunk=args.action_chunk,
        image_size=args.image_size,
        max_episodes=args.max_episodes,
        success_reward=args.success_reward,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
