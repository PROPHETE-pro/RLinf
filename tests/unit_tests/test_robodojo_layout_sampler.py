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


def _sampler_helpers():
    generate_layout, load_yaml = _maybe_import_sampler()
    import numpy as np
    import transforms3d as t3d

    from env.global_configs import OBJECTS_PATH
    from env.scene_manager.layout_sampler import _compose_pose, _record_pose
    from utils.load_file import load_object_metadata

    return {
        "generate_layout": generate_layout,
        "load_yaml": load_yaml,
        "np": np,
        "t3d": t3d,
        "OBJECTS_PATH": OBJECTS_PATH,
        "compose": _compose_pose,
        "record_pose": _record_pose,
        "load_object_metadata": load_object_metadata,
    }


def test_deposit_coin_snaps_to_stand_support():
    helpers = _sampler_helpers()
    task, scene = _load_task_and_scene(helpers["load_yaml"], "deposit_coin")
    layout = helpers["generate_layout"](task, scene, seed=48239)
    stand = layout["Geometry"]["vertical_coin_stand"][0]
    coin = layout["Rigid"]["coin"][0]
    metadata = helpers["load_object_metadata"](
        os.path.join(helpers["OBJECTS_PATH"], "Geometry", "vertical_coin_stand"),
        0,
    )
    support = helpers["np"].asarray(
        metadata["passive"]["support"]["vertical_coin_stand/0"]["center"][0],
        dtype=float,
    )
    expected = helpers["compose"](helpers["record_pose"](stand), support)
    coin_pose = helpers["record_pose"](coin)
    assert helpers["np"].allclose(coin_pose[3:], expected[3:], atol=1e-4)
    delta = coin_pose[:3] - expected[:3]
    assert helpers["np"].allclose(delta[:2], 0.0, atol=1e-4)
    assert float(delta[2]) > 0.004
    rot = helpers["t3d"].quaternions.quat2mat(coin_pose[3:])
    up = rot @ helpers["np"].array([0.0, 1.0, 0.0])
    assert helpers["np"].allclose(up, [0.0, 0.0, 1.0], atol=1e-3)


def test_play_xylophone_mallet_on_support_and_stand_variant():
    helpers = _sampler_helpers()
    task, scene = _load_task_and_scene(helpers["load_yaml"], "play_Xylophone")
    layout = helpers["generate_layout"](task, scene, seed=48239)
    stand = layout["Rigid"]["mallet_stand"][0]
    mallet = layout["Rigid"]["mallet"][0]
    stand_meta = helpers["load_object_metadata"](
        os.path.join(helpers["OBJECTS_PATH"], "Rigid", "mallet_stand"),
        0,
    )
    mallet_meta = helpers["load_object_metadata"](
        os.path.join(helpers["OBJECTS_PATH"], "Rigid", "mallet"),
        0,
    )
    support = helpers["np"].asarray(
        stand_meta["passive"]["support"]["mallet_stand/0"]["center"][0],
        dtype=float,
    )
    side = helpers["np"].asarray(
        mallet_meta["active"]["place"]["side"]["projection_circle"]["center"],
        dtype=float,
    )
    contact = helpers["compose"](helpers["record_pose"](mallet), side)
    expected = helpers["compose"](helpers["record_pose"](stand), support)
    assert helpers["np"].allclose(contact[3:], expected[3:], atol=1e-3)
    assert helpers["np"].allclose(contact[:2], expected[:2], atol=1e-3)
    assert float(contact[2]) + 1e-4 >= float(expected[2])

    tags = set()
    ranges = {"up": (-0.45, -0.2), "up_side": (-0.2, 0.05)}
    for seed in range(20):
        placed = helpers["generate_layout"](task, scene, seed=seed)
        chosen = placed["Rigid"]["mallet_stand"][0]
        tag = chosen["place_tag"]
        assert tag in ranges
        tags.add(tag)
        low, high = ranges[tag]
        assert low - 0.02 <= chosen["default_pos"][0] <= high + 0.02
    assert tags == {"up", "up_side"}


def test_fasten_screws_bolt_index_matches_nut():
    helpers = _sampler_helpers()
    task, scene = _load_task_and_scene(helpers["load_yaml"], "fasten_screws")
    for seed in range(8):
        layout = helpers["generate_layout"](task, scene, seed=seed)
        nuts = {item["label"]: item["category_idx"] for item in layout["Rigid"]["factory_nut"]}
        bolts = {item["label"]: item["category_idx"] for item in layout["Geometry"]["factory_bolt"]}
        indices = []
        for index in range(3):
            assert nuts[f"nut{index}"] == bolts[f"bolt{index}"]
            indices.append(nuts[f"nut{index}"])
        assert len(set(indices)) == 3


def test_pour_balls_stay_in_cup_local_frame():
    helpers = _sampler_helpers()
    task, scene = _load_task_and_scene(helpers["load_yaml"], "pour_balls_into_vase")
    layout = helpers["generate_layout"](task, scene, seed=48239)
    cup = helpers["record_pose"](layout["Rigid"]["cup"][0])
    inverse = helpers["t3d"].quaternions.qinverse(cup[3:])
    assert len(layout["Rigid"]["sphere"]) == 7
    for sphere in layout["Rigid"]["sphere"]:
        delta = helpers["np"].asarray(sphere["default_pos"], dtype=float) - cup[:3]
        local = helpers["t3d"].quaternions.rotate_vector(delta, inverse)
        assert abs(float(local[0])) <= 0.02 + 1e-6
        assert abs(float(local[1])) <= 0.02 + 1e-6
