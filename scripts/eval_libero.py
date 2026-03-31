#!/usr/bin/env python3
"""Evaluate Beta-VLA on LIBERO benchmark.

Single GPU:
    python scripts/eval_libero.py --checkpoint checkpoints/beta-action-chunk-0316/best --gpus 0 --video_out_path data/libero/action-chunk-0316

Multi-GPU (parallel workers):
    python scripts/eval_libero.py --checkpoint checkpoints/beta-action-chunk-0316/best --gpus 0 --video_out_path data/libero/action-chunk-0316

Or use the shell wrapper:
    bash scripts/eval_libero.sh 0,1,2,3 checkpoints/libero_vggt/best libero_10
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

# Ensure LIBERO is importable
_libero_root = Path(__file__).resolve().parent.parent.parent / "LIBERO"
if _libero_root.exists() and str(_libero_root) not in sys.path:
    sys.path.insert(0, str(_libero_root))

# Set LIBERO paths to avoid using ~/.libero from other users
_libero_benchmark = _libero_root / "libero" / "libero"
_config_dir = Path(__file__).resolve().parent.parent / "configs" / "libero_config"
_config_dir.mkdir(parents=True, exist_ok=True)
_config_file = _config_dir / "config.yaml"
if _libero_root.exists():
    _datasets_dir = _libero_root / "datasets"
    _datasets_dir.mkdir(exist_ok=True)
    _libero_paths = {
        "benchmark_root": str(_libero_benchmark),
        "bddl_files": str(_libero_benchmark / "bddl_files"),
        "init_states": str(_libero_benchmark / "init_files"),
        "datasets": str(_datasets_dir),
        "assets": str(_libero_benchmark / "assets"),
    }
    _config_file.write_text(
        "\n".join(f"{k}: {v}" for k, v in _libero_paths.items()),
        encoding="utf-8",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(_config_dir)

import argparse
import logging
import time

import numpy as np
import torch
from libero.libero import benchmark
from tqdm import tqdm

from betavla.data.normalize import load_norm_stats
from betavla.eval.inference import (
    clear_caches,
    get_tokenizer,
    load_model,
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
from betavla.training.config import load_config


TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
# ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_episode(
    env,
    initial_state,
    task_description: str,
    model,
    tokenizer,
    norm_stats,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    device: torch.device,
    save_video: bool = True,
    num_ode_steps: int = 5,
    debug_log_path: Path | None = None,
) -> tuple[bool, list]:
    """Run one episode. Returns (success, replay_images)."""
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    action_queue: deque = deque(maxlen=replan_steps)
    replay_images = [] if save_video else None
    debug_file = None

    if debug_log_path is not None:
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        debug_file = open(debug_log_path, "w", encoding="utf-8")
        debug_file.write(f"task: {task_description[:60]}\n")
        debug_file.write("t,action_norm,reward,done\n")

    # Clear per-episode language caches to ensure fresh encoding for each task
    clear_caches()

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
                "",  # legacy positional arg
                state,
                device,
                norm_stats=norm_stats,
                replan_steps=replan_steps,
                tokenizer=tokenizer,
                num_ode_steps=num_ode_steps,
            )
            # if (t - num_steps_wait) % 50 == 0:
            #     logger.info(
            #         "step=%d action_norm=%.4f max_abs=%.4f",
            #         t, float(np.linalg.norm(actions)), float(np.abs(actions).max()),
            #     )
            action_queue.extend(actions)

        action = action_queue.popleft()
        action_env = process_action_for_env(action)
        obs, reward, done, _ = env.step(action_env.tolist())

        if debug_file is not None:
            debug_file.write(f"{t},{np.linalg.norm(action):.4f},{reward},{done}\n")
            debug_file.flush()

        if done:
            if debug_file is not None:
                debug_file.write(f"# SUCCESS at step {t}\n")
                debug_file.close()
            return True, replay_images or []
        t += 1

    if debug_file is not None:
        debug_file.write(f"# FAILED at step {t}\n")
        debug_file.close()
    return False, replay_images or []


def _run_worker(args, task_suite, num_tasks: int, max_steps: int, save_video: bool,
                video_out_path, model, tokenizer, norm_stats, device) -> None:
    n_trials = args.num_trials_per_task
    rank, total = args.worker_rank, args.worker_total
    results = []
    worker_video_dir = (Path(video_out_path) / f"worker{rank}") if video_out_path else None

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)
        for trial_id in range(n_trials):
            if (task_id * n_trials + trial_id) % total != rank:
                continue
            success, replay_images = run_episode(
                env, initial_states[trial_id], task_description,
                model, tokenizer, norm_stats,
                args.replan_steps, args.num_steps_wait, max_steps,
                device,
                save_video=save_video,
                num_ode_steps=args.num_ode_steps,
            )
            results.append({"task_id": task_id, "trial_id": trial_id, "success": success})
            if save_video and replay_images and worker_video_dir:
                save_rollout_video(replay_images, trial_id, success, task_description, worker_video_dir, None)
        env.close()

    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_file, "w") as f:
        json.dump(results, f)


def _run_multi_gpu_launcher(gpu_list: list[int], args) -> None:
    run_dir = args.log_dir / args.run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{args.task_suite}.txt"

    with tempfile.TemporaryDirectory() as tmpdir:
        procs = []
        for rank, gpu_id in enumerate(gpu_list):
            result_file = Path(tmpdir) / f"worker_{rank}.json"
            cmd = [
                sys.executable, "-u", __file__,
                "--checkpoint", str(args.checkpoint),
                "--config", str(args.config),
                "--task_suite", args.task_suite,
                "--num_trials_per_task", str(args.num_trials_per_task),
                "--replan_steps", str(args.replan_steps),
                "--num_steps_wait", str(args.num_steps_wait),
                "--num_ode_steps", str(args.num_ode_steps),
                "--seed", str(args.seed),
                "--log_dir", str(args.log_dir),
                "--result_file", str(result_file),
                "--worker_rank", str(rank),
                "--worker_total", str(len(gpu_list)),
                "--gpu", "0",
            ]
            if args.norm_stats is not None:
                cmd += ["--norm_stats", str(args.norm_stats)]
            if args.no_norm_stats:
                cmd.append("--no_norm_stats")
            if args.max_tasks is not None:
                cmd += ["--max_tasks", str(args.max_tasks)]
            if args.no_video:
                cmd.append("--no_video")
            else:
                cmd += ["--video_out_path", str(args.video_out_path)]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            procs.append(subprocess.Popen(
                cmd, env=env,
                cwd=Path(__file__).resolve().parent.parent,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            ))

        for p in procs:
            p.wait()
            if p.returncode != 0:
                err = (p.stderr.read() or b"").decode()
                raise RuntimeError(f"Worker failed (code {p.returncode}): {err[:500]}")

        task_suite_cls = benchmark.get_benchmark_dict()[args.task_suite]
        task_suite = task_suite_cls()
        num_tasks = min(task_suite.n_tasks, args.max_tasks or task_suite.n_tasks)
        n_trials = args.num_trials_per_task
        task_successes = [0] * num_tasks
        total_successes = total_episodes = 0
        for rank in range(len(gpu_list)):
            for row in json.load(open(Path(tmpdir) / f"worker_{rank}.json")):
                task_successes[row["task_id"]] += int(row["success"])
                total_successes += int(row["success"])
                total_episodes += 1

        with open(log_path, "w") as lf:
            for task_id in range(num_tasks):
                task = task_suite.get_task(task_id)
                msg = f"Task {task_id} ({task.language[:40]}): {task_successes[task_id]}/{n_trials}"
                logger.info(msg)
                lf.write(msg + "\n")
            sr = total_successes / total_episodes if total_episodes else 0
            summary = f"Overall: {total_successes}/{total_episodes} = {sr:.2%}"
            logger.info(summary)
            lf.write(summary + "\n")
        logger.info("Log written to %s", log_path)


def main():
    p = argparse.ArgumentParser(description="Evaluate Beta-VLA on LIBERO")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/libero_vggt.yaml"))
    p.add_argument("--task_suite", default="all", help="Suite name or 'all'")
    p.add_argument("--num_trials_per_task", type=int, default=20)
    p.add_argument("--replan_steps", type=int, default=5)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--num_ode_steps", type=int, default=5)
    p.add_argument("--norm_stats", type=Path, default=None)
    p.add_argument("--no_norm_stats", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--video_out_path", type=Path, default=Path("data/libero/videos"))
    p.add_argument("--no_video", action="store_true")
    p.add_argument("--log_dir", type=Path, default=Path("./experiments/logs/eval"))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated GPU ids for multi-GPU eval")
    p.add_argument("--worker_rank", type=int, default=None)
    p.add_argument("--worker_total", type=int, default=None)
    p.add_argument("--result_file", type=Path, default=None)
    args = p.parse_args()

    suites = ALL_SUITES if args.task_suite.lower() == "all" else [args.task_suite]

    gpu_list = None
    if args.gpus:
        gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if gpu_list and len(gpu_list) > 1:
        args.run_ts = time.strftime("%Y%m%d_%H%M%S")
        for suite in suites:
            args.task_suite = suite
            _run_multi_gpu_launcher(gpu_list, args)
        return

    if gpu_list and len(gpu_list) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[0])

    cfg = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(args.checkpoint, device, cfg.model)

    if args.no_norm_stats:
        norm_stats = None
    else:
        norm_stats = load_norm_stats(args.norm_stats)
        if norm_stats is None:
            for cand in (args.checkpoint, args.checkpoint.parent):
                cand_file = Path(cand) / "norm_stats.json"
                if cand_file.exists():
                    norm_stats = load_norm_stats(cand_file)
                    break
        if norm_stats is None and cfg.data.norm_stats_path:
            norm_stats = load_norm_stats(cfg.data.norm_stats_path)

    logger.info("Norm stats: %s", "enabled" if norm_stats else "disabled")
    tokenizer = get_tokenizer(cfg.model.language.model_name)

    save_video = not args.no_video
    if save_video:
        try:
            import imageio  # noqa: F401
        except ImportError:
            logger.warning("imageio not installed, disabling video")
            save_video = False
    video_out_path = args.video_out_path if save_video else None

    if args.worker_rank is not None and args.result_file is not None:
        task_suite_cls = benchmark.get_benchmark_dict()[args.task_suite]
        task_suite = task_suite_cls()
        num_tasks = min(task_suite.n_tasks, args.max_tasks or task_suite.n_tasks)
        max_steps = TASK_MAX_STEPS.get(args.task_suite, 300)
        _run_worker(args, task_suite, num_tasks, max_steps, save_video, video_out_path,
                    model, tokenizer, norm_stats, device)
        return

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    args.run_ts = run_ts
    run_dir = args.log_dir / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)

    for suite in suites:
        args.task_suite = suite
        task_suite_cls = benchmark.get_benchmark_dict()[suite]
        task_suite = task_suite_cls()
        num_tasks = min(task_suite.n_tasks, args.max_tasks or task_suite.n_tasks)
        max_steps = TASK_MAX_STEPS.get(suite, 300)

        log_path = run_dir / f"{suite}.txt"
        log_file = open(log_path, "w", encoding="utf-8")
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.info("=" * 60)
        logger.info("Suite: %s (%d tasks)", suite, num_tasks)
        logger.info("=" * 60)

        total_episodes = total_successes = 0
        task_successes = []

        for task_id in tqdm(range(num_tasks), desc="Tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=256)
            task_ok = 0

            for ep in tqdm(range(args.num_trials_per_task), desc=f"T{task_id}", leave=False):
                success, replay_images = run_episode(
                    env, initial_states[ep], task_description,
                    model, tokenizer, norm_stats,
                    args.replan_steps, args.num_steps_wait, max_steps,
                    device,
                    save_video=save_video,
                    num_ode_steps=args.num_ode_steps,
                )
                if save_video and replay_images and video_out_path:
                    suite_video_path = Path(video_out_path) / suite
                    save_rollout_video(replay_images, ep, success, task_description, suite_video_path, log_file)
                task_ok += int(success)
                total_episodes += 1
                total_successes += int(success)

            task_successes.append(task_ok / args.num_trials_per_task)
            msg = f"Task {task_id} ({task_description[:40]}): {task_ok}/{args.num_trials_per_task}"
            logger.info(msg)
            env.close()

        sr = total_successes / total_episodes if total_episodes else 0
        summary = f"Overall: {total_successes}/{total_episodes} = {sr:.2%}"
        logger.info(summary)
        log_file.close()
        logger.removeHandler(handler)
        logger.info("Log written to %s", log_path)


if __name__ == "__main__":
    main()
