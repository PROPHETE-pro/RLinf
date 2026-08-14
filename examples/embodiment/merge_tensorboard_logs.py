#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
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

"""Merge TensorBoard event files produced by resumed RLinf training runs.

When training is resumed multiple times, RLinf writes a new ``events.out.tfevents.*``
file on each process start while keeping the same ``{log_dir}/tensorboard/`` directory.
TensorBoard may then fail to display the full step range (e.g. only the last segment).
This script reads all event files, deduplicates overlapping steps, and writes a single
merged log directory that TensorBoard can load from step 0.

Usage
-----
Basic merge (keep the latest value for duplicate steps)::

    python examples/embodiment/merge_tensorboard_logs.py \\
        --src /path/to/experiment/tensorboard \\
        --dst /path/to/experiment/tensorboard_merged

Then open TensorBoard::

    tensorboard --logdir /path/to/experiment/tensorboard_merged

Concrete example for a resumed robotwin run::

    python examples/embodiment/merge_tensorboard_logs.py \\
        --src /mnt/pfs/7wsqem/grt/RLinf/logs/20260724-19:07:53-robotwin_open_microwave_ppo_openpi_pi05_1/tensorboard \\
        --dst /mnt/pfs/7wsqem/grt/RLinf/logs/20260724-19:07:53-robotwin_open_microwave_ppo_openpi_pi05_1/tensorboard_merged

Options
-------
``--keep earliest``  Keep the first recorded value when the same (tag, step) appears
                       in multiple event files (useful to inspect the original run
                       before a resume overwrote overlapping steps).

``--keep latest``     Keep the value from the event with the greatest wall_time
                       (default; recommended for viewing the final effective curve).

Notes
-----
- Only scalar metrics are merged. Other event types (images, histograms, etc.) are
  skipped.
- The source directory is never modified; output is written to ``--dst``.
- If ``--dst`` already exists, its previous ``events.out.tfevents.*`` files are removed
  before writing the merged result.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge RLinf TensorBoard logs from resumed training runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python examples/embodiment/merge_tensorboard_logs.py \\\n"
            "    --src logs/my_exp/tensorboard \\\n"
            "    --dst logs/my_exp/tensorboard_merged\n"
        ),
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Source TensorBoard directory containing events.out.tfevents.* files.",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Output directory for the merged TensorBoard log.",
    )
    parser.add_argument(
        "--keep",
        choices=("latest", "earliest"),
        default="latest",
        help=(
            "Which value to keep when the same (tag, step) appears in multiple files. "
            "Default: latest."
        ),
    )
    return parser.parse_args()


def _collect_scalar_events(src_dir: str, keep: str) -> dict[tuple[str, int], float]:
    from tensorboard.backend.event_processing import event_accumulator

    event_files = sorted(glob.glob(os.path.join(src_dir, "events.out.tfevents.*")))
    if not event_files:
        raise FileNotFoundError(
            f"No events.out.tfevents.* files found under: {src_dir}"
        )

    # (tag, step) -> (sort_key, value)
    selected: dict[tuple[str, int], tuple[float, float]] = {}

    for path in event_files:
        accumulator = event_accumulator.EventAccumulator(path)
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                key = (tag, event.step)
                sort_key = event.wall_time
                if key not in selected:
                    selected[key] = (sort_key, event.value)
                    continue
                current_sort_key, _ = selected[key]
                if keep == "latest":
                    if sort_key >= current_sort_key:
                        selected[key] = (sort_key, event.value)
                elif sort_key <= current_sort_key:
                    selected[key] = (sort_key, event.value)

    return {key: value for key, (_, value) in selected.items()}


def _write_merged_events(dst_dir: str, scalars: dict[tuple[str, int], float]) -> None:
    from torch.utils.tensorboard import SummaryWriter

    os.makedirs(dst_dir, exist_ok=True)
    for existing in glob.glob(os.path.join(dst_dir, "events.out.tfevents.*")):
        os.remove(existing)

    by_step: dict[int, dict[str, float]] = {}
    for (tag, step), value in scalars.items():
        by_step.setdefault(step, {})[tag] = value

    writer = SummaryWriter(dst_dir)
    try:
        for step in sorted(by_step):
            for tag, value in sorted(by_step[step].items()):
                writer.add_scalar(tag, value, step)
    finally:
        writer.close()


def merge_tensorboard_logs(src_dir: str, dst_dir: str, keep: str = "latest") -> None:
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)

    if not os.path.isdir(src_dir):
        raise NotADirectoryError(f"Source directory does not exist: {src_dir}")

    config_src = os.path.join(src_dir, "config.yaml")
    if os.path.isfile(config_src):
        os.makedirs(dst_dir, exist_ok=True)
        config_dst = os.path.join(dst_dir, "config.yaml")
        if os.path.abspath(config_src) != os.path.abspath(config_dst):
            shutil.copy2(config_src, config_dst)

    scalars = _collect_scalar_events(src_dir, keep=keep)
    if not scalars:
        raise RuntimeError(f"No scalar metrics found under: {src_dir}")

    _write_merged_events(dst_dir, scalars)

    steps = sorted({step for _, step in scalars})
    tags = sorted({tag for tag, _ in scalars})
    num_files = len(glob.glob(os.path.join(src_dir, "events.out.tfevents.*")))
    print(f"Merged {num_files} event file(s) from: {src_dir}")
    print(f"Wrote merged log to: {dst_dir}")
    print(f"Step range: {steps[0]} .. {steps[-1]} ({len(steps)} unique steps)")
    print(f"Scalar tags: {len(tags)}")
    print(f"Deduplication policy: keep {keep}")


def main() -> None:
    args = _parse_args()
    try:
        merge_tensorboard_logs(args.src, args.dst, keep=args.keep)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
