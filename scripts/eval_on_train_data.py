#!/usr/bin/env python3
"""Evaluate Beta-VLA on TRAINING data (physical-intelligence/libero train split).

Use this to verify the eval pipeline correctness: if the model performs well on
training episodes (same init states, same tasks), the pipeline is likely correct.

Usage:
  python scripts/eval_on_train_data.py --checkpoint checkpoints/beta_vla_libero/best --max_episodes 5
  python scripts/eval_on_train_data.py --checkpoint ... --max_episodes 50 --no_video  # faster

Speed: Use MUJOCO_GL=egl (GPU render, ~5-10 min/ep) instead of osmesa (CPU, ~2h/ep).

Output:
  - Videos saved to --video_out_path (default: data/libero/videos/eval_on_train/)
  - JSON results with all metadata: task_description, episode_index, task_index, success, etc.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure LIBERO is importable
_libero_root = Path(__file__).resolve().parent.parent.parent / "LIBERO"
if _libero_root.exists() and str(_libero_root) not in sys.path:
    sys.path.insert(0, str(_libero_root))

_config_dir = Path(__file__).resolve().parent.parent / "configs" / "libero_config"
_config_dir.mkdir(parents=True, exist_ok=True)
_config_file = _config_dir / "config.yaml"
if _libero_root.exists():
    _libero_benchmark = _libero_root / "libero" / "libero"
    _datasets_dir = _libero_root / "datasets"
    _datasets_dir.mkdir(exist_ok=True)
    _libero_paths = {
        "benchmark_root": str(_libero_benchmark),
        "bddl_files": str(_libero_benchmark / "bddl_files"),
        "init_states": str(_libero_benchmark / "init_files"),
        "datasets": str(_datasets_dir),
        "assets": str(_libero_benchmark / "assets"),
    }
    _config_file.write_text("\n".join(f"{k}: {v}" for k, v in _libero_paths.items()), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(_config_dir)

import argparse
import logging
import time
from collections import deque

import numpy as np
import torch
from datasets import load_dataset
from libero.libero import benchmark
from tqdm import tqdm

from betavla.eval.inference import (
    get_tokenizer,
    load_model,
    load_norm_stats,
    predict,
    process_action_for_env,
)
from betavla.eval.libero_utils import (
    LIBERO_DUMMY_ACTION,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    prepare_state,
    save_rollout_video,
)
from betavla.training.config_beta_vla import load_beta_vla_config

# physical-intelligence/libero: 40 tasks = libero_spatial(10) + libero_object(10) + libero_goal(10) + libero_10(10)
TASK_INDEX_TO_SUITE_AND_ID = []
for suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    suite_cls = benchmark.get_benchmark_dict()[suite_name]
    suite = suite_cls()
    for task_id in range(suite.n_tasks):
        TASK_INDEX_TO_SUITE_AND_ID.append((suite_name, task_id, suite))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_task_for_index(task_index: int):
    """Map dataset task_index to (task, task_description)."""
    if task_index < 0 or task_index >= len(TASK_INDEX_TO_SUITE_AND_ID):
        raise ValueError(f"task_index {task_index} out of range [0, {len(TASK_INDEX_TO_SUITE_AND_ID)})")
    suite_name, task_id, suite = TASK_INDEX_TO_SUITE_AND_ID[task_index]
    task = suite.get_task(task_id)
    return task, task.language


def run_episode(
    env,
    initial_state,
    task_description: str,
    model,
    tokenizer,
    norm_stats: dict | None,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    invert_gripper: bool,
    device: torch.device,
    save_video: bool = False,
    num_ode_steps: int = 5,
):
    """Run one episode. Returns (success, replay_images)."""
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
    action_queue = deque(maxlen=replan_steps)
    replay_images = [] if save_video else None
    t = 0
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue
        base_img = get_libero_image(obs)
        if save_video:
            replay_images.append(base_img)
        wrist_img = get_libero_wrist_image(obs)
        state = prepare_state(obs)
        if len(action_queue) == 0:
            actions = predict(
                model,
                base_img,
                wrist_img,
                task_description,
                "",
                state,
                device,
                norm_stats=norm_stats,
                replan_steps=replan_steps,
                tokenizer=tokenizer,
                num_ode_steps=num_ode_steps,
            )
            action_queue.extend(actions)
        action = action_queue.popleft()
        action_for_env = process_action_for_env(action, invert_gripper=invert_gripper)
        obs, reward, done, info = env.step(action_for_env.tolist())
        if done:
            return True, replay_images or []
        t += 1
    return False, replay_images or []


def _to_scalar(x):
    """Convert HF dataset value (possibly array) to scalar."""
    if isinstance(x, (list, np.ndarray)):
        return int(x[0]) if len(x) > 0 else 0
    return int(x)


def load_train_episodes(repo_id: str = "physical-intelligence/libero", max_episodes: int | None = None):
    """Load training dataset and return list of (episode_index, task_index) for first frame of each episode."""
    ds = load_dataset(repo_id, split="train")
    # Filter to first frame of each episode (frame_index == 0)
    first_frames = []
    for i in range(len(ds)):
        row = ds[i]
        frame_idx = _to_scalar(row.get("frame_index", 0))
        if frame_idx == 0:
            ep_idx = _to_scalar(row.get("episode_index", i))
            task_idx = _to_scalar(row.get("task_index", 0))
            first_frames.append((ep_idx, task_idx))
        if max_episodes is not None and len(first_frames) >= max_episodes:
            break
    return first_frames[:max_episodes] if max_episodes else first_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/train_beta_vla_libero.yaml"))
    p.add_argument("--max_episodes", type=int, default=50, help="Max episodes to eval (default 50)")
    p.add_argument("--repo_id", type=str, default="physical-intelligence/libero")
    p.add_argument("--replan_steps", type=int, default=5)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--num_ode_steps", type=int, default=15)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--resolution", type=int, default=256, help="Env render resolution (default 256)")
    p.add_argument(
        "--group_by_task",
        action="store_true",
        default=True,
        help="Sort selected episodes by task_index to reduce env recreation overhead",
    )
    p.add_argument("--no_group_by_task", action="store_false", dest="group_by_task")
    p.add_argument(
        "--quick_debug",
        action="store_true",
        help="Fast sanity-check preset: disable video, lower steps/ODE, lower resolution",
    )
    p.add_argument("--invert_gripper", action="store_true", default=True)
    p.add_argument("--no_invert_gripper", action="store_false", dest="invert_gripper")
    p.add_argument("--norm_stats", type=Path, default=None)
    p.add_argument("--no_norm_stats", action="store_true")
    p.add_argument("--video_out_path", type=Path, default=Path("data/libero/videos/eval_on_train"))
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--log_dir", type=Path, default=Path("./experiments/logs"))
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    if args.quick_debug:
        # Keep this preset focused on validating pipeline correctness quickly.
        args.no_video = True
        args.max_steps = min(args.max_steps, 150)
        args.num_ode_steps = min(args.num_ode_steps, 2)
        args.replan_steps = max(args.replan_steps, 10)
        args.resolution = min(args.resolution, 128)

    cfg = load_beta_vla_config(args.config)
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model, _ = load_model(args.checkpoint, device, model_config=cfg.model)

    if args.no_norm_stats:
        norm_stats = None
    else:
        norm_stats = load_norm_stats(args.norm_stats)
        if norm_stats is None:
            for cand_dir in (args.checkpoint, args.checkpoint.parent):
                cand = Path(cand_dir) / "norm_stats.json"
                if cand.exists():
                    norm_stats = load_norm_stats(cand)
                    break
        if norm_stats is None:
            norm_stats = load_norm_stats(cfg.checkpoint_dir / "norm_stats.json")

    tokenizer = get_tokenizer(cfg.model.language.model_name)
    save_video = not args.no_video
    if save_video:
        try:
            import imageio  # noqa: F401
        except ImportError:
            logger.warning("imageio not installed, disabling video save")
            save_video = False

    logger.info("Loading training episodes...")
    episodes = load_train_episodes(repo_id=args.repo_id, max_episodes=args.max_episodes)
    if args.group_by_task:
        episodes = sorted(episodes, key=lambda x: (x[1], x[0]))
    n_tasks = len({task_idx for _, task_idx in episodes})
    logger.info(f"Eval on {len(episodes)} training episodes")
    logger.info(
        "Eval settings: no_video=%s, max_steps=%d, num_ode_steps=%d, replan_steps=%d, resolution=%d, tasks=%d",
        args.no_video,
        args.max_steps,
        args.num_ode_steps,
        args.replan_steps,
        args.resolution,
        n_tasks,
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.video_out_path / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    meta_path = out_dir / "metadata.json"

    results = []
    total_success = 0
    last_task_idx = -1
    env = None

    for ep_idx, task_idx in tqdm(episodes, desc="Episodes"):
        ep_start = time.time()
        task, task_description = get_task_for_index(task_idx)
        suite_name = TASK_INDEX_TO_SUITE_AND_ID[task_idx][0]
        suite = TASK_INDEX_TO_SUITE_AND_ID[task_idx][2]
        task_id_in_suite = TASK_INDEX_TO_SUITE_AND_ID[task_idx][1]

        # Create new env when task changes (each task has different bddl)
        if env is None or last_task_idx != task_idx:
            if env is not None:
                env.close()
            env_switch_start = time.time()
            env, _ = get_libero_env(task, resolution=args.resolution)
            last_task_idx = task_idx
            logger.info(
                "Switched env to task_index=%d (suite=%s, task_id=%d) in %.2fs",
                task_idx,
                suite_name,
                task_id_in_suite,
                time.time() - env_switch_start,
            )

        initial_states = suite.get_task_init_states(task_id_in_suite)
        init_state_idx = ep_idx % len(initial_states)
        initial_state = initial_states[init_state_idx]

        success, replay_images = run_episode(
            env,
            initial_state,
            task_description,
            model,
            tokenizer,
            norm_stats,
            args.replan_steps,
            args.num_steps_wait,
            args.max_steps,
            args.invert_gripper,
            device,
            save_video=save_video,
            num_ode_steps=args.num_ode_steps,
        )

        total_success += int(success)
        record = {
            "episode_index": ep_idx,
            "task_index": task_idx,
            "task_suite": suite_name,
            "task_id_in_suite": task_id_in_suite,
            "task_description": task_description,
            "init_state_idx": init_state_idx,
            "success": success,
        }
        results.append(record)
        logger.info(
            "Episode done: ep=%d task=%d success=%s elapsed=%.2fs",
            ep_idx,
            task_idx,
            success,
            time.time() - ep_start,
        )

        if save_video and replay_images:
            save_rollout_video(
                replay_images,
                ep_idx,
                success,
                task_description,
                out_dir,
                None,
            )

    if env is not None:
        env.close()

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    metadata = {
        "run_ts": run_ts,
        "checkpoint": str(args.checkpoint),
        "max_episodes": args.max_episodes,
        "total_episodes": len(results),
        "total_success": total_success,
        "success_rate": total_success / len(results) if results else 0,
        "video_out_path": str(out_dir),
        "results_path": str(results_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    sr = total_success / len(results) if results else 0
    logger.info(f"Success rate: {total_success}/{len(results)} = {sr:.2%}")
    logger.info(f"Results: {results_path}")
    logger.info(f"Metadata: {meta_path}")
    logger.info(f"Videos: {out_dir}")


if __name__ == "__main__":
    main()
