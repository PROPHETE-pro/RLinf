#!/usr/bin/env python3
"""Roll one RoboDojo env to max_episode_steps with training-style OpenDM chunks.

Matches rollout inference of::

    bash examples/embodiment/run_robodojo.sh robodojo_build_tower_ppo_opendm_dm05_1gpu

Every chunk until ``max_episode_steps`` is kept: the observation sent to the
model, the denormalized absolute joint chunk, and the three cameras after each
action. A full-horizon plot and video line those commands up with the arm.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

WORKSPACE = Path(os.environ.get("WORKSPACE", "/kpfs_ssd/data/ruitong_gan"))
RLINF = WORKSPACE / "RLinf"
JOINT_NAMES = [
    "L_j1",
    "L_j2",
    "L_j3",
    "L_j4",
    "L_j5",
    "L_j6",
    "L_grip",
    "R_j1",
    "R_j2",
    "R_j3",
    "R_j4",
    "R_j5",
    "R_j6",
    "R_grip",
]
ARM_INDEX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def _setup_env() -> None:
    os.environ.setdefault("WORKSPACE", str(WORKSPACE))
    os.environ.setdefault("ROBODOJO_PATH", str(WORKSPACE / "RoboDojo"))
    os.environ.setdefault(
        "ROBODOJO_RUNTIME_LIBS", str(WORKSPACE / "robodojo_runtime" / "runtime_libs")
    )
    os.environ.setdefault(
        "WARP_CACHE_PATH", str(WORKSPACE / "robodojo_runtime" / "caches" / "warp")
    )
    os.environ.setdefault(
        "ROBODOJO_XDG_CACHE_HOME", str(WORKSPACE / "robodojo_runtime" / "caches")
    )
    os.environ.setdefault(
        "ROBODOJO_EVAL_ROOT",
        str(WORKSPACE / "robodojo_runtime" / "eval_result" / "RoboDojo" / "rlinf"),
    )
    os.environ.setdefault("EMBODIED_PATH", str(RLINF / "examples" / "embodiment"))
    os.environ.setdefault("REPO_PATH", str(RLINF))
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    if not os.environ.get("ROBODOJO_PYTHON"):
        for candidate in (
            WORKSPACE / "miniconda" / "envs" / "RoboDojo" / "bin" / "python",
            WORKSPACE / "miniconda" / "envs" / "robodojo" / "bin" / "python",
        ):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                os.environ["ROBODOJO_PYTHON"] = str(candidate)
                break
    repo = str(RLINF)
    pythonpath = os.environ.get("PYTHONPATH", "")
    if repo not in pythonpath.split(":"):
        os.environ["PYTHONPATH"] = repo if not pythonpath else f"{repo}:{pythonpath}"
    if str(RLINF) not in sys.path:
        sys.path.insert(0, str(RLINF))


def _load_train_cfg(config_name: str, extra_overrides: list[str] | None = None):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_dir = str(RLINF / "examples" / "embodiment" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(
            config_name=config_name,
            overrides=[
                "env.train.total_num_envs=1",
                "env.train.video_cfg.save_video=false",
                *(extra_overrides or []),
            ],
        )


def _as_uint8_hwc(image) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def _split_obs(obs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    main = _as_uint8_hwc(obs["main_images"][0])
    wrists = obs["wrist_images"][0]
    if hasattr(wrists, "detach"):
        wrists = wrists.detach().cpu().numpy()
    left = _as_uint8_hwc(wrists[0])
    right = _as_uint8_hwc(wrists[1])
    state = obs["states"][0]
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    return main, left, right, np.asarray(state, dtype=np.float32).reshape(-1)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(image).save(path)


def _panel(head, left, right, title: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    tiles = [Image.fromarray(img) for img in (head, left, right)]
    height = min(tile.height for tile in tiles)
    resized = []
    for tile in tiles:
        if tile.height != height:
            width = max(1, int(tile.width * height / tile.height))
            tile = tile.resize((width, height))
        resized.append(tile)
    labels = ("head", "left wrist", "right wrist")
    banner_h = 36
    width = sum(tile.width for tile in resized)
    canvas = Image.new("RGB", (width, height + banner_h), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for tile, label in zip(resized, labels):
        canvas.paste(tile, (x, banner_h))
        draw.text((x + 8, 8), label, fill=(220, 220, 220))
        x += tile.width
    draw.text((width // 2, 8), title, fill=(255, 220, 80))
    return np.asarray(canvas)


def _info_float(infos, key: str) -> float | None:
    if not isinstance(infos, dict) or key not in infos:
        return None
    value = infos[key]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr[0]):
        return None
    return float(arr[0])


def _tensor_float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(arr[0]) if arr.size else 0.0


def _reward_strip(
    width: int,
    horizon: int,
    scores: list[float],
    chunk_spans: list[tuple[int, int, float | None]],
    cursor: int,
    phi: float | None,
    height: int = 176,
) -> np.ndarray:
    """Cumulative score without the per-step time cost. Up is green, down is red."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), (14, 14, 16))
    draw = ImageDraw.Draw(image)
    current = scores[-1] if scores else 0.0
    title = (
        f"score(no time cost)={current:.4f}   "
        f"phi={phi if phi is not None else float('nan'):.3f}"
    )
    draw.text((8, 4), title, fill=(240, 240, 240))
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

    for span_index, (start, end, _gain) in enumerate(chunk_spans):
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
    for start, _end, _gain in chunk_spans:
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


def _compose_frame(cameras: np.ndarray, strip: np.ndarray) -> np.ndarray:
    if cameras.shape[1] != strip.shape[1]:
        from PIL import Image

        strip_img = Image.fromarray(strip).resize((cameras.shape[1], strip.shape[0]))
        strip = np.asarray(strip_img)
    return np.concatenate([cameras, strip], axis=0)


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    import imageio.v2 as imageio

    writer = imageio.get_writer(path, fps=fps)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def _plot_episode(
    path: Path, actions: np.ndarray, states: np.ndarray, chunk_len: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(actions.shape[0])
    fig, axes = plt.subplots(7, 2, figsize=(14, 16), sharex=True)
    axes = axes.T.reshape(-1)
    for joint, ax in enumerate(axes):
        ax.plot(steps, states[:-1, joint], "--", color="0.45", label="state before")
        ax.plot(steps, actions[:, joint], color="C1", linewidth=1.8, label="command")
        ax.plot(steps, states[1:, joint], color="C0", label="state after")
        ax.set_ylabel(JOINT_NAMES[joint])
        ax.grid(True, alpha=0.3)
    for boundary in range(chunk_len, actions.shape[0], chunk_len):
        for ax in axes:
            ax.axvline(boundary - 0.5, color="0.7", linewidth=0.6)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("control step")
    axes[-2].set_xlabel("control step")
    fig.suptitle(
        "Full episode: commanded absolute joints vs measured joints "
        "(vertical lines are chunk boundaries)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _chunk_metrics(actions: np.ndarray, states: np.ndarray) -> dict:
    """``states`` includes the pose before action 0, so its length is T+1."""
    arm = np.array(ARM_INDEX)
    before = states[:-1, arm]
    after = states[1:, arm]
    commanded = actions[:, arm]
    step_delta = np.abs(np.diff(commanded, axis=0))
    max_jump = float(step_delta.max()) if step_delta.size else 0.0
    mean_motion = float(step_delta.mean()) if step_delta.size else 0.0
    boundary = float(np.abs(commanded[0] - before[0]).max()) if len(commanded) else 0.0
    max_snap = float(np.abs(commanded - before).max()) if commanded.size else 0.0
    max_follow = float(np.abs(after - commanded).max()) if commanded.size else 0.0
    if mean_motion < 0.01 and max_jump < 0.05:
        kind = "repeat"
    elif max_jump >= 0.35 or boundary >= 0.5:
        kind = "jitter"
    elif max_follow >= 0.35 and max_jump < 0.2:
        kind = "execution"
    else:
        kind = "smooth"
    return {
        "kind": kind,
        "max_arm_command_jump_rad": max_jump,
        "mean_arm_command_step_rad": mean_motion,
        "boundary_snap_rad": boundary,
        "max_arm_command_vs_state_before_rad": max_snap,
        "max_arm_state_after_vs_command_rad": max_follow,
    }


def _summarize(chunks: list[dict]) -> dict:
    kinds: dict[str, list[int]] = {}
    for row in chunks:
        kinds.setdefault(row["kind"], []).append(int(row["chunk_index"]))
    jitter = kinds.get("jitter", [])
    repeat = kinds.get("repeat", [])
    execution = kinds.get("execution", [])
    if jitter and not execution:
        cause = "model_chunk"
        note = (
            f"有 {len(jitter)} 个 chunk 的关节指令自己在跳，或新 chunk 的第一步"
            f"相对当前姿态跳得很大（chunk {jitter}）。蓝线贴着橙线时，抖动就是模型输出。"
        )
    elif repeat and not jitter and not execution:
        cause = "model_repeat"
        note = (
            f"有 {len(repeat)} 个 chunk 的手臂指令几乎不变（chunk {repeat}）。"
            "这段重复来自模型一直给出同一个姿态。"
        )
    elif execution and not jitter:
        cause = "execution"
        note = (
            f"chunk {execution} 的指令比较连续，但执行后的关节跟指令差得很远。"
            "这段更像控制器或仿真没有跟上。"
        )
    elif jitter or repeat:
        cause = "model_chunk"
        note = (
            f"重复 chunk {repeat or '无'}，指令跳变 chunk {jitter or '无'}，"
            f"执行偏离 chunk {execution or '无'}。"
            "垂直线是 chunk 边界；边界处橙线突然离开蓝线，就是新 chunk 和上一段接不上。"
        )
    else:
        cause = "smooth"
        note = "整段 episode 的手臂指令连续，执行也跟着走，没有明显的抖动或原地重复。"
    return {
        "cause": cause,
        "note": note,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "jitter_chunks": jitter,
        "repeat_chunks": repeat,
        "execution_chunks": execution,
        "inference_mode": "train",
        "noise_method": "flow_sde",
        "action_meaning": "denormalized absolute joint targets sent to the env",
    }


def _denoise_step(result) -> int | None:
    forward = result.get("forward_inputs") if isinstance(result, dict) else None
    if not isinstance(forward, dict) or "denoise_inds" not in forward:
        return None
    inds = forward["denoise_inds"]
    if hasattr(inds, "detach"):
        inds = inds.detach().cpu().numpy()
    inds = np.asarray(inds).reshape(-1)
    if inds.size == 0:
        return None
    return int(inds[0])


def _info_flag(infos, key: str) -> bool:
    if not isinstance(infos, dict) or key not in infos:
        return False
    value = infos[key]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return False
    return bool(arr[0])


def _plot_rewards(
    path: Path,
    scores: np.ndarray,
    dense_phi: np.ndarray,
    chunk_len: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(1, scores.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(steps, scores, color="C0", linewidth=1.6, label="score without time cost")
    ax.plot(steps, dense_phi, color="C1", linewidth=1.0, alpha=0.7, label="dense phi")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("cumulative score")
    ax.set_xlabel("control step")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    for boundary in range(chunk_len, scores.shape[0], chunk_len):
        ax.axvline(boundary + 0.5, color="0.7", linewidth=0.6)
    fig.suptitle("Cumulative score without time cost (vertical lines are chunk boundaries)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    _setup_env()
    import imageio.v2 as imageio
    import torch

    from rlinf.envs.robodojo.robodojo_env import RoboDojoEnv
    from rlinf.models.embodiment.opendm_dm05 import get_model

    config_name = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "robodojo_insert_key_ppo_opendm_dm05_1gpu"
    )
    cfg = _load_train_cfg(config_name, sys.argv[2:])
    task_name = str(cfg.env.train.task_config.task_name)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RLINF / "logs" / "probe_episode" / f"{task_name}_{stamp}"
    model_input_dir = out / "model_inputs"
    image_dir = out / "images"
    model_input_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ROBODOJO_WORKER_LOG_DIR"] = str(out)
    print(f"[probe] config={config_name} task={task_name}", flush=True)
    print(f"[probe] output={out}", flush=True)
    env_cfg = cfg.env.train
    chunk_len = int(cfg.actor.model.num_action_chunks)
    print("[probe] starting one Isaac env", flush=True)
    env = RoboDojoEnv(
        cfg=env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    horizon = int(env.cfg.max_episode_steps)
    print(
        f"[probe] horizon={horizon} chunk_len={chunk_len} "
        f"chunks={int(np.ceil(horizon / chunk_len))}",
        flush=True,
    )
    print("[probe] offloading Isaac so the policy can load on the same GPU", flush=True)
    env.offload()

    print(f"[probe] loading {cfg.actor.model.model_path}", flush=True)
    model = get_model(cfg.actor.model)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()

    print("[probe] reset, then roll until max_episode_steps", flush=True)
    obs, _info = env.reset()
    head, left, right, state = _split_obs(obs)
    states = [state]
    action_chunks: list[np.ndarray] = []
    chunk_rows: list[dict] = []
    action_rewards: list[float] = []
    score_trace: list[float] = []
    dense_phi: list[float] = []
    chunk_spans: list[tuple[int, int, float | None]] = []
    success_at = None
    executed = 0
    episode_return = 0.0
    stopped = "horizon"

    video_path = out / "execution.mp4"
    writer = imageio.get_writer(video_path, fps=10)
    try:
        opening = _panel(head, left, right, "t=0 before any action")
        writer.append_data(
            _compose_frame(
                opening,
                _reward_strip(opening.shape[1], horizon, [], [], 0, None),
            )
        )
        _save_rgb(image_dir / "step_0000_head.jpg", head)
        _save_rgb(image_dir / "step_0000_left_wrist.jpg", left)
        _save_rgb(image_dir / "step_0000_right_wrist.jpg", right)

        chunk_index = 0
        while executed < horizon:
            _save_rgb(model_input_dir / f"chunk_{chunk_index:03d}_head.png", head)
            _save_rgb(model_input_dir / f"chunk_{chunk_index:03d}_left_wrist.png", left)
            _save_rgb(model_input_dir / f"chunk_{chunk_index:03d}_right_wrist.png", right)
            with torch.no_grad():
                actions, result = model.predict_action_batch(
                    env_obs=obs, mode="train", compute_values=True
                )
            chunk = actions[0].detach().float().cpu().numpy().astype(np.float32)
            remain = horizon - executed
            chunk = chunk[:remain]
            denoise_step = _denoise_step(result)
            state_before = state.copy()
            chunk_states = [state_before]
            chunk_success = False
            chunk_start = executed
            score_before = _tensor_float(env.prev_step_reward)
            return_before = episode_return
            chunk_action_rewards: list[float] = []
            print(
                f"[probe] chunk {chunk_index} predict shape={chunk.shape} "
                f"denoise_step={denoise_step} step {executed}/{horizon} "
                f"score_before={score_before:.4f}",
                flush=True,
            )
            for index in range(chunk.shape[0]):
                obs_list, rewards, terms, truncs, infos_list = env.chunk_step(
                    chunk[index][None, None, :]
                )
                infos = infos_list[-1] if infos_list else {}
                obs = obs_list[-1]
                head, left, right, state = _split_obs(obs)
                states.append(state)
                chunk_states.append(state)
                executed += 1
                delta = _tensor_float(rewards)
                phi = _info_float(infos, "dense_progress")
                score_now = _tensor_float(env.prev_step_reward)
                action_rewards.append(delta)
                chunk_action_rewards.append(delta)
                score_trace.append(score_now)
                dense_phi.append(phi if phi is not None else float("nan"))
                episode_return += delta
                chunk_gain = float(sum(chunk_action_rewards))
                if _info_flag(infos, "success"):
                    chunk_success = True
                    if success_at is None:
                        success_at = executed
                done = _info_flag({"terminated": terms}, "terminated") or _info_flag(
                    {"truncated": truncs}, "truncated"
                )
                cameras = _panel(
                    head,
                    left,
                    right,
                    f"chunk {chunk_index:02d}  action {index:02d}/{chunk.shape[0] - 1:02d}  t={executed}",
                )
                writer.append_data(
                    _compose_frame(
                        cameras,
                        _reward_strip(
                            cameras.shape[1],
                            horizon,
                            score_trace,
                            chunk_spans + [(chunk_start, executed, chunk_gain)],
                            executed,
                            phi,
                        ),
                    )
                )
                step_name = f"step_{executed:04d}"
                _save_rgb(image_dir / f"{step_name}_head.jpg", head)
                _save_rgb(image_dir / f"{step_name}_left_wrist.jpg", left)
                _save_rgb(image_dir / f"{step_name}_right_wrist.jpg", right)
                if done:
                    stopped = "done"
                    break
            score_after = _tensor_float(env.prev_step_reward)
            chunk_gain = float(sum(chunk_action_rewards))
            chunk_spans.append((chunk_start, executed, chunk_gain))
            metrics = _chunk_metrics(chunk[: len(chunk_action_rewards)], np.stack(chunk_states, axis=0))
            metrics.update(
                {
                    "chunk_index": chunk_index,
                    "start_step": chunk_start,
                    "end_step": executed,
                    "num_actions": int(len(chunk_action_rewards)),
                    "denoise_step_with_sde_noise": denoise_step,
                    "env_success_during_chunk": chunk_success,
                    "score_before": score_before,
                    "score_after": score_after,
                    "score_gain": score_after - score_before,
                    "return_before": return_before,
                    "return_after": episode_return,
                    "chunk_reward_gain": chunk_gain,
                    "phi_before": dense_phi[chunk_start - 1] if chunk_start else None,
                    "phi_after": dense_phi[executed - 1] if executed else None,
                }
            )
            chunk_rows.append(metrics)
            action_chunks.append(chunk[: len(chunk_action_rewards)])
            print(
                f"[probe] chunk {chunk_index} reward {score_before:.4f} -> {score_after:.4f} "
                f"gain={chunk_gain:+.4f} phi={metrics['phi_after']}",
                flush=True,
            )
            np.savez_compressed(
                out / "trace.npz",
                action_chunks=np.concatenate(action_chunks, axis=0),
                chunk_lengths=np.array([c.shape[0] for c in action_chunks]),
                joint_names=np.array(JOINT_NAMES),
                measured_states=np.stack(states, axis=0),
                dense_phi=np.asarray(dense_phi, dtype=np.float32),
                action_reward=np.asarray(action_rewards, dtype=np.float32),
                score_without_time=np.asarray(score_trace, dtype=np.float32),
            )
            chunk_index += 1
            if stopped == "done":
                break
    finally:
        writer.close()

    summary = _summarize(chunk_rows) if chunk_rows else {"cause": "empty", "note": "", "chunks": []}
    summary["config"] = config_name
    summary["task"] = task_name
    summary["horizon"] = horizon
    summary["executed_steps"] = executed
    summary["stopped"] = stopped
    summary["success_step"] = success_at
    summary["episode_return"] = episode_return
    summary["reward_meaning"] = (
        "The video curve and score_without_time are env.prev_step_reward: "
        "shaping_coef * dense phi, or the success score, with no time cost. "
        "action_reward is still the training step reward, that change minus time_cost."
    )
    _write_summary(out / "summary.json", summary)
    if action_chunks:
        _plot_episode(
            out / "action_vs_state.png",
            np.concatenate(action_chunks, axis=0),
            np.stack(states, axis=0),
            chunk_len,
        )
        _plot_rewards(
            out / "reward_trace.png",
            np.asarray(score_trace, dtype=np.float32),
            np.asarray(dense_phi, dtype=np.float32),
            chunk_len,
        )
    env.offload()
    print(f"[probe] cause={summary['cause']}", flush=True)
    print(summary["note"], flush=True)
    print(f"[probe] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
