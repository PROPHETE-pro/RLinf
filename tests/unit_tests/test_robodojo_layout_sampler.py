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

"""Smoke tests for RoboDojo in-memory task-layout sampling (no Assets writes)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROBODOJO = Path(
    os.environ.get("ROBODOJO_PATH", "/kpfs_ssd/data/ruitong_gan/RoboDojo")
)


def _maybe_import_sampler():
    if not ROBODOJO.is_dir():
        pytest.skip(f"RoboDojo path missing: {ROBODOJO}")
    root = str(ROBODOJO)
    if root not in sys.path:
        sys.path.insert(0, root)
    pytest.importorskip("shapely")
    pytest.importorskip("rtree")
    pytest.importorskip("transforms3d")
    from env.scene_manager.layout_sampler import generate_layout
    from utils.load_file import load_yaml

    return generate_layout, load_yaml


def _load_task_and_scene(load_yaml, task_name: str):
    task = load_yaml(str(ROBODOJO / "task" / "RoboDojo" / "config" / f"{task_name}.yml"))
    scene = load_yaml(str(ROBODOJO / "env_cfg" / "scene" / "default.yml"))
    return task, scene


def _assert_no_eval_layout_write(before_mtime: dict[str, float], pack_root: Path):
    after = {}
    if pack_root.is_dir():
        for path in pack_root.rglob("*"):
            after[str(path)] = path.stat().st_mtime if path.is_file() else 0.0
    created = set(after) - set(before_mtime)
    assert not created, f"sampler wrote Eval_Layout files: {sorted(created)[:5]}"
    for path, mtime in before_mtime.items():
        assert after.get(path) == mtime, f"Eval_Layout file changed: {path}"


def _snapshot_eval_layout() -> tuple[dict[str, float], Path]:
    pack_root = ROBODOJO / "Assets" / "Eval_Layout"
    snap = {}
    if pack_root.is_dir():
        for path in pack_root.rglob("*"):
            snap[str(path)] = path.stat().st_mtime if path.is_file() else 0.0
    return snap, pack_root


def test_build_tower_same_seed_reproducible():
    generate_layout, load_yaml = _maybe_import_sampler()
    task, scene = _load_task_and_scene(load_yaml, "build_tower")
    before, pack_root = _snapshot_eval_layout()
    a = generate_layout(task, scene, seed=7)
    b = generate_layout(task, scene, seed=7)
    c = generate_layout(task, scene, seed=8)
    _assert_no_eval_layout_write(before, pack_root)

    assert "Rigid" in a and "block" in a["Rigid"]
    labels = [inst["label"] for inst in a["Rigid"]["block"]]
    assert "block0" in labels and "block7" in labels
    assert a["Rigid"]["block"][0]["default_pos"] == b["Rigid"]["block"][0]["default_pos"]
    assert a["Rigid"]["block"][0]["default_ori"] == b["Rigid"]["block"][0]["default_ori"]
    assert a["Table"]["default"] == "material_0122"
    assert a["Background"]["category_name"]
    assert "camera_stand" in a.get("Geometry", {})
    assert a["Rigid"]["block"][1]["default_pos"] != c["Rigid"]["block"][1]["default_pos"]


def test_stack_bowls_same_category_idx():
    generate_layout, load_yaml = _maybe_import_sampler()
    task, scene = _load_task_and_scene(load_yaml, "stack_bowls")
    layout = generate_layout(task, scene, seed=11)
    bowls = layout["Rigid"]["bowl"]
    assert len(bowls) == 3
    idxs = {bowl["category_idx"] for bowl in bowls}
    assert len(idxs) == 1
    xs = [bowl["default_pos"][0] for bowl in bowls]
    assert len(set(round(x, 6) for x in xs)) == 3


def test_push_t_category_without_index():
    generate_layout, load_yaml = _maybe_import_sampler()
    task, scene = _load_task_and_scene(load_yaml, "push_T")
    layout = generate_layout(task, scene, seed=4)
    assert "t" in layout["Rigid"]
    assert "t_cushion" in layout["Geometry"]
    assert layout["Geometry"]["t_cushion"][0]["label"] == "target_t"


def test_hang_mugs_random_clutter_and_prohibited():
    generate_layout, load_yaml = _maybe_import_sampler()
    task, scene = _load_task_and_scene(load_yaml, "hang_mugs_random")
    layout = generate_layout(task, scene, seed=3)
    mugs = layout["Rigid"]["mug"]
    assert len(mugs) == 3
    cluttered = [
        inst
        for cat_list in layout["Rigid"].values()
        for inst in cat_list
        if inst.get("type") == "cluttered"
    ]
    assert cluttered, "expected Clutter instances in hang_mugs_random"
    assert all(inst.get("yaml_path") == "Clutter/clutter.yml" for inst in cluttered)
    assert "cup_holder" in layout.get("Geometry", {})
