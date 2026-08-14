# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import queue
import threading
import time
from typing import Any, Iterator, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import concat_batch

logger = get_logger()

_SCALAR_TRAJECTORY_FIELDS = frozenset({"versions", "prev_values"})
_ACTION_ALIGNED_FIELDS = frozenset({"actions", "intervene_flags"})
_CHUNK_ALIGNED_FIELDS = frozenset(
    {"rewards", "terminations", "truncations", "dones", "prev_logprobs"}
)
_IMAGE_OBS_KEYS = frozenset({"main_images", "wrist_images"})
# OpenPI / demo_buffer convention; online RoboTwin env may emit native ~240px.
_OPENPI_IMAGE_SIZE = (224, 224)


def _expand_tensor_last_dim(tensor: torch.Tensor, target: int) -> torch.Tensor:
    """Expand [B, W] (or [B]) tensors to [B, target] when target is a multiple of W."""
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.dim() != 2:
        return tensor
    width = int(tensor.shape[1])
    if width == target:
        return tensor
    if target % width == 0:
        repeat = target // width
        return (
            tensor.unsqueeze(-1)
            .expand(-1, -1, repeat)
            .reshape(tensor.shape[0], -1)
        )
    if width % target == 0:
        return tensor[:, :target]
    return tensor


def _image_spatial_size(tensor: torch.Tensor) -> tuple[int, int] | None:
    if tensor.dim() == 4:
        return int(tensor.shape[1]), int(tensor.shape[2])
    if tensor.dim() == 5:
        return int(tensor.shape[2]), int(tensor.shape[3])
    return None


def _collect_obs_image_sizes(batch: dict) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for obs_key in ("curr_obs", "next_obs"):
        obs = batch.get(obs_key)
        if not isinstance(obs, dict):
            continue
        for key in _IMAGE_OBS_KEYS:
            tensor = obs.get(key)
            if isinstance(tensor, torch.Tensor):
                hw = _image_spatial_size(tensor)
                if hw is not None:
                    sizes.append(hw)
    return sizes


def _pick_unified_image_size(
    sizes: list[tuple[int, int]],
) -> tuple[int, int] | None:
    if not sizes:
        return None
    if len(set(sizes)) == 1:
        return None
    # Prefer OpenPI/demo 224 when mixing native env frames with demo_buffer.
    if any(size == _OPENPI_IMAGE_SIZE for size in sizes):
        return _OPENPI_IMAGE_SIZE
    heights = [h for h, _ in sizes]
    widths = [w for _, w in sizes]
    return min(heights), min(widths)


def _resize_image_obs_tensor(
    tensor: torch.Tensor, target_hw: tuple[int, int]
) -> torch.Tensor:
    target_h, target_w = target_hw
    if _image_spatial_size(tensor) == (target_h, target_w):
        return tensor

    orig_dtype = tensor.dtype
    if tensor.dim() == 4:
        # [B, H, W, C]
        x = tensor.permute(0, 3, 1, 2).float()
        x = F.interpolate(
            x, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return x.permute(0, 2, 3, 1).round().clamp(0, 255).to(orig_dtype)

    if tensor.dim() == 5:
        # [B, N, H, W, C]
        batch, num_cams = tensor.shape[0], tensor.shape[1]
        x = tensor.reshape(batch * num_cams, *tensor.shape[2:]).permute(0, 3, 1, 2)
        x = x.float()
        x = F.interpolate(
            x, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        x = x.permute(0, 2, 3, 1).reshape(batch, num_cams, target_h, target_w, -1)
        return x.round().clamp(0, 255).to(orig_dtype)

    return tensor


def _align_observation_images(batch: dict, target_hw: tuple[int, int]) -> None:
    for obs_key in ("curr_obs", "next_obs"):
        obs = batch.get(obs_key)
        if not isinstance(obs, dict):
            continue
        for key in _IMAGE_OBS_KEYS:
            tensor = obs.get(key)
            if isinstance(tensor, torch.Tensor):
                obs[key] = _resize_image_obs_tensor(tensor, target_hw)


def _batch_action_flat_width(batch: dict) -> int | None:
    if "actions" not in batch:
        return None
    actions = batch["actions"]
    if actions.dim() == 3:
        actions = actions.reshape(actions.shape[0], -1)
        batch["actions"] = actions
    if actions.dim() == 2:
        return int(actions.shape[1])
    return None


def canonicalize_mixed_sample_batch(batch: dict) -> dict:
    """Normalize sampled batch tensors so replay/demo can be concatenated."""
    if not batch:
        return batch

    if "actions" in batch:
        actions = batch["actions"]
        if actions.dim() == 3:
            actions = actions.reshape(actions.shape[0], -1)
        batch["actions"] = actions

    action_width = (
        batch["actions"].shape[1]
        if "actions" in batch and batch["actions"].dim() == 2
        else None
    )

    if action_width is not None and "intervene_flags" in batch:
        flags = batch["intervene_flags"]
        if flags.dim() == 1:
            flags = flags.unsqueeze(-1)
        if flags.dim() == 2 and flags.shape[1] != action_width:
            if action_width % flags.shape[1] == 0:
                repeat = action_width // flags.shape[1]
                flags = (
                    flags.unsqueeze(-1)
                    .expand(-1, -1, repeat)
                    .reshape(flags.shape[0], -1)
                )
        batch["intervene_flags"] = flags

    for field in ("rewards", "terminations", "truncations", "dones", "prev_logprobs"):
        if field not in batch:
            continue
        tensor = batch[field]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        elif tensor.dim() > 2:
            while tensor.dim() > 2 and tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
            if tensor.dim() > 2:
                tensor = tensor.reshape(tensor.shape[0], -1)
        batch[field] = tensor

    for field in _SCALAR_TRAJECTORY_FIELDS:
        if field not in batch:
            continue
        tensor = batch[field]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        elif tensor.dim() >= 2:
            # Online rollout sets versions = full_like(prev_logprobs) which can be
            # [B, H, D] or [B, H*D]; demo stores scalar metadata [B, 1].
            tensor = tensor.reshape(tensor.shape[0], -1)[:, :1]
        batch[field] = tensor

    for obs_key in ("curr_obs", "next_obs"):
        obs = batch.get(obs_key)
        if not isinstance(obs, dict):
            continue
        for key, tensor in obs.items():
            if isinstance(tensor, torch.Tensor) and tensor.dim() == 1:
                obs[key] = tensor.unsqueeze(-1)

    return batch


def align_mixed_batches_for_concat(
    replay_batch: dict, demo_batch: dict
) -> tuple[dict, dict]:
    """Canonicalize and cross-align replay/demo batches before concatenation."""
    replay_batch = canonicalize_mixed_sample_batch(replay_batch)
    demo_batch = canonicalize_mixed_sample_batch(demo_batch)

    action_widths = [
        w
        for w in (
            _batch_action_flat_width(replay_batch),
            _batch_action_flat_width(demo_batch),
        )
        if w is not None
    ]
    action_width = max(action_widths) if action_widths else None

    chunk_widths: list[int] = []
    for batch in (replay_batch, demo_batch):
        for field in _CHUNK_ALIGNED_FIELDS:
            if field not in batch:
                continue
            tensor = batch[field]
            if tensor.dim() == 1:
                chunk_widths.append(1)
            elif tensor.dim() >= 2:
                chunk_widths.append(int(tensor.shape[1]))
    chunk_width = max(chunk_widths) if chunk_widths else None

    image_sizes = _collect_obs_image_sizes(replay_batch) + _collect_obs_image_sizes(
        demo_batch
    )
    target_image_hw = _pick_unified_image_size(image_sizes)

    for batch in (replay_batch, demo_batch):
        if action_width is not None:
            for field in _ACTION_ALIGNED_FIELDS:
                if field in batch:
                    batch[field] = _expand_tensor_last_dim(batch[field], action_width)
        if chunk_width is not None:
            for field in _CHUNK_ALIGNED_FIELDS:
                if field in batch:
                    batch[field] = _expand_tensor_last_dim(batch[field], chunk_width)
        if target_image_hw is not None:
            _align_observation_images(batch, target_image_hw)

    return replay_batch, demo_batch


class ReplayBufferDataset(IterableDataset):
    """Dataset that samples batches from replay and demonstration buffers.

    This dataset provides an infinite iterator that yields batches sampled from
    a replay buffer and optionally a demonstration buffer. When both buffers are
    provided, batches are composed of half replay samples and half demonstration
    samples.

    Attributes:
        replay_buffer: Buffer storing online rollout trajectories.
        demo_buffer: Optional buffer storing offline demonstration trajectories
            and online human-in-the-loop trajectories.
        min_replay_buffer_size: Minimum number of samples required in replay
            buffer before sampling begins.
        min_demo_buffer_size: Minimum number of samples required in demo buffer
            before sampling begins (if demo_buffer is provided).
        batch_size: Total number of samples per batch.
    """

    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: Optional[TrajectoryReplayBuffer],
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        demo_ratio: float = 0.5,
        allow_demo_only: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the ReplayBufferDataset.

        Args:
            replay_buffer: Buffer storing online rollout trajectories.
            demo_buffer: Optional buffer storing demonstration trajectories.
                If None, only replay buffer is used.
            batch_size: Total number of samples per batch.
            min_replay_buffer_size: Minimum number of samples required in replay
                buffer before sampling begins.
            min_demo_buffer_size: Minimum number of samples required in demo
                buffer before sampling begins (ignored if demo_buffer is None).
            demo_ratio: Fraction of each batch drawn from demo_buffer when both
                buffers are used. Defaults to 0.5 (legacy 50/50).
            allow_demo_only: If True, allow sampling purely from demo_buffer when
                replay is below min size (offline LWD stage).
            **kwargs: Additional keyword arguments (unused, for compatibility).
        """
        self.replay_buffer = replay_buffer
        self.demo_buffer = demo_buffer
        self.min_replay_buffer_size = min_replay_buffer_size
        self.min_demo_buffer_size = min_demo_buffer_size

        self.batch_size = batch_size
        self.demo_ratio = float(demo_ratio)
        assert 0.0 <= self.demo_ratio <= 1.0, f"demo_ratio must be in [0,1], got {self.demo_ratio}"
        self.allow_demo_only = bool(allow_demo_only)

    def _sample_mixed_batch(self) -> dict[str, torch.Tensor]:
        """Sample a batch using configured demo_ratio (or single-buffer fallback)."""
        if self.demo_buffer is None:
            return self.replay_buffer.sample(self.batch_size)

        replay_ready = self.replay_buffer.is_ready(self.min_replay_buffer_size)
        demo_ready = self.demo_buffer.is_ready(self.min_demo_buffer_size)

        if self.allow_demo_only and demo_ready and not replay_ready:
            return self.demo_buffer.sample(self.batch_size)
        if self.demo_ratio >= 1.0:
            return self.demo_buffer.sample(self.batch_size)
        if self.demo_ratio <= 0.0:
            return self.replay_buffer.sample(self.batch_size)

        demo_n = int(round(self.batch_size * self.demo_ratio))
        demo_n = max(0, min(self.batch_size, demo_n))
        replay_n = self.batch_size - demo_n
        if replay_n <= 0:
            return self.demo_buffer.sample(self.batch_size)
        if demo_n <= 0:
            return self.replay_buffer.sample(self.batch_size)
        replay_batch = self.replay_buffer.sample(replay_n)
        demo_batch = self.demo_buffer.sample(demo_n)
        replay_batch, demo_batch = align_mixed_batches_for_concat(
            replay_batch, demo_batch
        )
        return concat_batch(replay_batch, demo_batch)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Returns an infinite iterator that yields batches.

        Waits until both buffers (if demo_buffer is provided) reach their
        minimum size requirements before yielding batches. When ready, samples
        from replay buffer only or from both replay and demo buffers.

        Yields:
            Batch dictionary containing sampled trajectories. Keys and structure
            depend on the buffer's trajectory format.
        """
        while True:
            is_ready = True
            replay_ready = self.replay_buffer.is_ready(self.min_replay_buffer_size)
            if not replay_ready:
                if not (
                    self.allow_demo_only
                    and self.demo_buffer is not None
                    and self.demo_buffer.is_ready(self.min_demo_buffer_size)
                ):
                    is_ready = False
            if self.demo_buffer is not None and not self.demo_buffer.is_ready(
                self.min_demo_buffer_size
            ):
                # Offline-only: still require demo when mixing; if demo_ratio==0 skip.
                if self.demo_ratio > 0.0:
                    is_ready = False

            if is_ready:
                yield self._sample_mixed_batch()
            else:
                time.sleep(0.5)

    def close(self) -> None:
        """Releases references to replay and demo buffers."""
        del self.replay_buffer
        del self.demo_buffer

    def __del__(self) -> None:
        """Destructor that ensures buffers are cleaned up."""
        self.close()


class PreloadReplayBufferDataset(ReplayBufferDataset):
    """Dataset that prefetches batches from replay and demo buffers in background.

    This dataset extends ReplayBufferDataset by prefetching batches in a
    background thread, which can improve throughput by overlapping sampling
    with training. Batches are stored in a queue of configurable size.

    Attributes:
        replay_buffer: Buffer storing online rollout trajectories.
        demo_buffer: Optional buffer storing demonstration trajectories.
        min_replay_buffer_size: Minimum number of samples required in replay
            buffer before sampling begins.
        min_demo_buffer_size: Minimum number of samples required in demo buffer
            before sampling begins (if demo_buffer is provided).
        batch_size: Total number of samples per batch.
        prefetch_size: Maximum number of batches to prefetch and store in queue.
        preload_queue: Queue holding prefetched batches.
        sample_thread: Background thread that samples batches.
    """

    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: Optional[TrajectoryReplayBuffer],
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        prefetch_size: int = 5,
        demo_ratio: float = 0.5,
        allow_demo_only: bool = False,
    ) -> None:
        """Initializes the PreloadReplayBufferDataset.

        Args:
            replay_buffer: Buffer storing online rollout trajectories.
            demo_buffer: Optional buffer storing demonstration trajectories.
                If None, only replay buffer is used.
            batch_size: Total number of samples per batch.
            min_replay_buffer_size: Minimum number of samples required in replay
                buffer before sampling begins.
            min_demo_buffer_size: Minimum number of samples required in demo
                buffer before sampling begins (ignored if demo_buffer is None).
            prefetch_size: Maximum number of batches to prefetch and store in
                the queue. Defaults to 10.
            demo_ratio: Fraction of each batch from demo_buffer.
            allow_demo_only: Allow offline sampling from demo only.
        """
        self._stop_event = threading.Event()

        self.replay_buffer = replay_buffer
        self.demo_buffer = demo_buffer
        self.min_replay_buffer_size = min_replay_buffer_size
        self.min_demo_buffer_size = min_demo_buffer_size

        self.batch_size = batch_size
        self.demo_ratio = float(demo_ratio)
        assert 0.0 <= self.demo_ratio <= 1.0, f"demo_ratio must be in [0,1], got {self.demo_ratio}"
        self.allow_demo_only = bool(allow_demo_only)
        self.prefetch_size = prefetch_size
        assert self.prefetch_size > 0, f"{self.prefetch_size=} must be greater than 0"

        self.preload_queue = queue.Queue(maxsize=prefetch_size)
        self.sample_thread = None
        self._exception = None

    def _sample_mixed_batch(self) -> dict[str, torch.Tensor]:
        return ReplayBufferDataset._sample_mixed_batch(self)

    def _sample_buffer(self) -> None:
        """Background thread target that continuously samples batches.

        Runs in a loop until stop event is set. Waits for buffers to be ready,
        samples batches, and puts them in the preload queue. If the queue is
        full, skips the sample and retries. Sleeps when buffers are not ready
        or when errors occur.
        """
        while not self._stop_event.is_set():
            if self.preload_queue.full():
                time.sleep(0.1)
                continue

            is_ready = True
            replay_ready = self.replay_buffer.is_ready(self.min_replay_buffer_size)
            if not replay_ready:
                if not (
                    self.allow_demo_only
                    and self.demo_buffer is not None
                    and self.demo_buffer.is_ready(self.min_demo_buffer_size)
                ):
                    is_ready = False
            if self.demo_buffer is not None and not self.demo_buffer.is_ready(
                self.min_demo_buffer_size
            ):
                if self.demo_ratio > 0.0:
                    is_ready = False

            if is_ready:
                batch = self._sample_mixed_batch()
            else:
                time.sleep(3)
                continue

            try:
                self.preload_queue.put(batch, timeout=1)
            except queue.Full:
                logger.info("Queue is full, skipping sample")
                time.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"Error in ReplayBufferDataset: {e}")
                self._exception = e
                self._stop_event.set()
                break

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Returns an iterator that yields prefetched batches.

        Starts the background sampling thread on first call. Retrieves batches
        from the preload queue and yields them. Stops when the stop event is set.

        Yields:
            Batch dictionary containing sampled trajectories. Keys and structure
            depend on the buffer's trajectory format.
        """
        if self.sample_thread is None:
            self.sample_thread = threading.Thread(
                target=self._sample_buffer, daemon=True
            )
            self.sample_thread.start()

        while not self._stop_event.is_set():
            try:
                batch = self.preload_queue.get(timeout=1)
                yield batch
            except queue.Empty:
                if self._stop_event.is_set():
                    # Check if thread died with exception
                    if hasattr(self, "_exception"):
                        raise RuntimeError(
                            "Sampling thread failed"
                        ) from self._exception
                    break
                continue

    def close(self) -> None:
        """Stops the background sampling thread and cleans up resources.

        Sets the stop event and waits up to 10 seconds for the sampling thread
        to terminate. Logs a warning if the thread does not terminate in time.
        """
        self._stop_event.set()

        thread_timeout = 10
        if self.sample_thread.is_alive():
            self.sample_thread.join(timeout=thread_timeout)
            if self.sample_thread.is_alive():
                logger.warning(
                    f"Sample thread is still alive after {thread_timeout} seconds, force killing"
                )

    def __del__(self) -> None:
        """Destructor that ensures the sampling thread is stopped."""
        if not self._stop_event.is_set():
            self.close()


def replay_buffer_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader that returns the first batch element.

    Since the dataset already yields complete batches, this function simply
    extracts the batch from the list wrapper added by DataLoader.

    Args:
        batch: List containing a single batch dictionary.

    Returns:
        The unwrapped batch dictionary.
    """
    return batch[0]
