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

"""Head-camera video overlay matching toolkits/robodojo/probe_first_chunk.py."""

from __future__ import annotations

import numpy as np


def render_reward_strip(
    width: int,
    horizon: int,
    scores: list[float],
    chunk_spans: list[tuple[int, int]],
    cursor: int,
    phi: float | None,
    *,
    height: int = 176,
) -> np.ndarray:
    """Cumulative score without the per-step time cost. Up is green, down is red."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (max(int(width), 1), height), (14, 14, 16))
    draw = ImageDraw.Draw(image)
    current = scores[-1] if scores else 0.0
    previous = scores[-2] if len(scores) > 1 else 0.0
    dense_delta = current - previous
    phi_text = f"{phi:.3f}" if phi is not None else "nan"
    draw.text(
        (8, 4),
        (
            f"score(no time cost)={current:.4f}   "
            f"phi={phi_text}   dense={dense_delta:.4f}"
        ),
        fill=(240, 240, 240),
    )
    left, right, top, bottom = 8, max(9, width - 8), 28, height - 22
    slots = max(int(horizon), 1)
    plot_w = right - left
    plot_h = max(bottom - top, 1)
    y_max = 1.0

    def x_at(step: int) -> int:
        return left + int(min(step, slots) * plot_w / slots)

    def y_at(score: float) -> int:
        clipped = min(y_max, max(0.0, float(score)))
        return bottom - int(clipped / y_max * plot_h)

    for span_index, (start, end) in enumerate(chunk_spans):
        x0 = x_at(start)
        x1 = max(x_at(end), x0 + 1)
        shade = (24, 36, 32) if span_index % 2 == 0 else (36, 28, 28)
        draw.rectangle((x0, top, x1, bottom), fill=shade)
    for mark in (0.25, 0.5, 0.75, 1.0):
        y = y_at(mark)
        draw.line((left, y, right, y), fill=(55, 55, 58))
        draw.text((left + 2, y - 11), f"{mark:.2f}", fill=(120, 120, 120))
    draw.line((left, bottom, right, bottom), fill=(90, 90, 90))
    points = [(x_at(0), y_at(0.0))]
    points.extend((x_at(index + 1), y_at(score)) for index, score in enumerate(scores))
    for start, end in zip(points, points[1:]):
        delta = end[1] - start[1]
        if delta < -1:
            color = (70, 190, 90)
        elif delta > 1:
            color = (210, 70, 70)
        else:
            color = (220, 210, 150)
        draw.line((start, end), fill=color, width=3)
    for start, _end in chunk_spans:
        x0 = x_at(start)
        draw.line((x0, top, x0, bottom), fill=(235, 235, 235), width=2)
    cursor_x = x_at(cursor)
    draw.line((cursor_x, top, cursor_x, bottom), fill=(255, 210, 60), width=2)
    draw.text(
        (8, height - 16),
        "curve = cumulative score without time cost    green up / red down    white = chunk start    yellow = now",
        fill=(180, 180, 180),
    )
    return np.asarray(image)


def compose_frame_with_strip(frame: np.ndarray, strip: np.ndarray) -> np.ndarray:
    """Stack the reward strip under one RGB frame."""
    image = np.asarray(frame)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if strip.shape[1] != image.shape[1]:
        from PIL import Image

        strip = np.asarray(
            Image.fromarray(strip).resize((image.shape[1], strip.shape[0]))
        )
    return np.concatenate([image, strip], axis=0)
