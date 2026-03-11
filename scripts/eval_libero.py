#!/usr/bin/env python3
# 旧模型（训练时未做 action 归一化）:
#   python scripts/eval_libero.py --checkpoint checkpoints/beta_vla_libero/best --no_norm_stats --gpus 7
# 新模型（训练时做了 action 归一化，checkpoint 内含 norm_stats.json）:
#   python scripts/eval_libero.py --checkpoint checkpoints/beta_vla_libero/6000 ...
# Multi-GPU: python scripts/eval_libero.py --checkpoint ... --gpus 4,5,6,7
# bash scripts/run_eval_libero.sh --checkpoint checkpoints/beta_vla_libero/best --no_norm_stats --task_suite all --gpus 7
#
# 加速建议（eval 慢时）:
#   --no_video         关闭视频保存（每 episode 省 ~30s）
#   --num_ode_steps 4  减少 ODE 步数（默认 5，10 更准但慢一倍）
#   MUJOCO_GL=egl      用 GPU 渲染（需 headless，比 osmesa 快很多）
#   --gpus 4,5,6,7    多卡并行
#   --verbose         保存每 episode 的 step-by-step debug log（排查提前停止）
"""Evaluate Beta-VLA on LIBERO benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure LIBERO is importable (editable install can fail on NFS)
_libero_root = Path(__file__).resolve().parent.parent.parent / "LIBERO"
if _libero_root.exists() and str(_libero_root) not in sys.path:
    sys.path.insert(0, str(_libero_root))

# 设置 LIBERO 路径，避免使用 ~/.libero 中其他用户的配置
_libero_benchmark = _libero_root / "libero" / "libero"
_config_dir = Path(__file__).resolve().parent.parent / "configs" / "libero_config"
_config_dir.mkdir(parents=True, exist_ok=True)
_config_file = _config_dir / "config.yaml"
if _libero_root.exists():
    _datasets_dir = _libero_root / "datasets"
    _datasets_dir.mkdir(exist_ok=True)  # 避免 LIBERO 报 "datasets path does not exist"（eval 不需 demos）
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
from collections import deque

import numpy as np
import torch
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

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
    "libero_100": 300,
}

ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90", "libero_100"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
    debug_log_path: Path | None = None,
):
    """Run one episode. Returns (success, replay_images). replay_images is empty if save_video=False.
    When debug_log_path is set, write step-by-step log for debugging."""
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
    action_queue = deque(maxlen=replan_steps)
    replay_images = [] if save_video else None
    debug_file = None
    if debug_log_path is not None:
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        debug_file = open(debug_log_path, "w", encoding="utf-8")
        debug_file.write(f"task: {task_description[:60]}...\n")
        debug_file.write("t,action_norm,action_max,reward,done,action_0,action_1,action_2,action_3,action_4,action_5,action_6\n")
        debug_file.flush()
    t = 0
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        if save_video:
            replay_images.append(get_libero_image(obs))
        base_img = get_libero_image(obs)
        wrist_img = get_libero_wrist_image(obs)
        state = prepare_state(obs)

        if len(action_queue) == 0:
            actions = predict(
                model,
                base_img,
                wrist_img,
                task_description,
                "",  # unused when tokenizer passed
                state,
                device,
                norm_stats=norm_stats,
                replan_steps=replan_steps,
                tokenizer=tokenizer,
                num_ode_steps=num_ode_steps,
            )
            # 诊断：每 50 步打印一次 action 幅度，排查提前停止
            if (t - num_steps_wait) % 50 == 0:
                logger.info(
                    "step %d action norm=%.4f max_abs=%.4f",
                    t,
                    float(np.linalg.norm(actions)),
                    float(np.abs(actions).max()),
                )
            action_queue.extend(actions)

        action = action_queue.popleft()
        action_for_env = process_action_for_env(action, invert_gripper=invert_gripper)
        obs, reward, done, info = env.step(action_for_env.tolist())
        if debug_file is not None:
            a_str = ",".join(f"{x:.6f}" for x in action)
            debug_file.write(f"{t},{np.linalg.norm(action):.4f},{np.abs(action).max():.4f},{reward},{done},{a_str}\n")
            debug_file.flush()
        if done:
            if debug_file is not None:
                debug_file.write(f"# SUCCESS at step {t}\n")
                debug_file.close()
            return True, replay_images or []
        t += 1
    if debug_file is not None:
        debug_file.write(f"# FAILED (max_steps) at step {t}\n")
        debug_file.close()
    return False, replay_images or []


def _run_worker(
    args,
    task_suite,
    num_tasks: int,
    max_steps: int,
    save_video: bool,
    video_out_path: Path | None,
    model,
    tokenizer,
    norm_stats: dict | None,
    device: torch.device,
) -> None:
    """Run this process's share of (task_id, trial_id) and write results to args.result_file."""
    n_trials = args.num_trials_per_task
    rank, total = args.worker_rank, args.worker_total
    results = []
    worker_video_dir = (Path(video_out_path) / f"worker{rank}") if video_out_path else None
    debug_log_dir = None
    if getattr(args, "verbose", False):
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        debug_log_dir = args.log_dir / f"eval_{args.task_suite}_{args.checkpoint.name}_{run_ts}" / "debug" / f"worker{rank}"
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)
        for trial_id in range(n_trials):
            if (task_id * n_trials + trial_id) % total != rank:
                continue
            debug_path = None
            if debug_log_dir is not None:
                safe_desc = task_description[:30].replace(" ", "_").replace(".", "_")
                debug_path = debug_log_dir / f"task{task_id}_ep{trial_id}_{safe_desc}.log"
            success, replay_images = run_episode(
                env,
                initial_states[trial_id],
                task_description,
                model,
                tokenizer,
                norm_stats,
                args.replan_steps,
                args.num_steps_wait,
                max_steps,
                args.invert_gripper,
                device,
                save_video=save_video,
                num_ode_steps=args.num_ode_steps,
                debug_log_path=debug_path,
            )
            results.append({"task_id": task_id, "trial_id": trial_id, "success": success})
            if save_video and replay_images and worker_video_dir:
                save_rollout_video(
                    replay_images,
                    trial_id,
                    success,
                    task_description,
                    worker_video_dir,
                    None,
                )
        env.close()
    args.result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.result_file, "w") as f:
        json.dump(results, f)


def _run_multi_gpu_launcher(gpu_list: list[int], args) -> None:
    """Spawn one Python process per GPU and aggregate results."""
    args.log_dir.mkdir(parents=True, exist_ok=True)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = args.log_dir / f"eval_{args.task_suite}_{args.checkpoint.name}_{run_ts}.txt"
    with tempfile.TemporaryDirectory() as tmpdir:
        procs = []
        for rank, gpu_id in enumerate(gpu_list):
            result_file = Path(tmpdir) / f"worker_{rank}.json"
            cmd = [
                sys.executable,
                "-u",
                __file__,
                "--checkpoint",
                str(args.checkpoint),
                "--config",
                str(args.config),
                "--task_suite",
                args.task_suite,
                "--num_trials_per_task",
                str(args.num_trials_per_task),
                "--replan_steps",
                str(args.replan_steps),
                "--num_steps_wait",
                str(args.num_steps_wait),
                "--num_ode_steps",
                str(args.num_ode_steps),
                "--seed",
                str(args.seed),
                "--log_dir",
                str(args.log_dir),
                "--result_file",
                str(result_file),
                "--worker_rank",
                str(rank),
                "--worker_total",
                str(len(gpu_list)),
                "--gpu",
                "0",
            ]
            if args.norm_stats is not None:
                cmd += ["--norm_stats", str(args.norm_stats)]
            if args.no_norm_stats:
                cmd.append("--no_norm_stats")
            if args.max_tasks is not None:
                cmd += ["--max_tasks", str(args.max_tasks)]
            if args.invert_gripper:
                cmd.append("--invert_gripper")
            else:
                cmd.append("--no_invert_gripper")
            if args.no_video:
                cmd.append("--no_video")
            else:
                cmd += ["--video_out_path", str(args.video_out_path)]
            if getattr(args, "verbose", False):
                cmd.append("--verbose")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            procs.append(
                subprocess.Popen(
                    cmd,
                    env=env,
                    cwd=Path(__file__).resolve().parent.parent,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            )
        for p in procs:
            p.wait()
            if p.returncode != 0:
                err = p.stderr.read().decode() if p.stderr else ""
                raise RuntimeError(f"Worker failed with code {p.returncode}: {err}")

        # Aggregate results
        task_suite_cls = benchmark.get_benchmark_dict()[args.task_suite]
        task_suite = task_suite_cls()
        num_tasks = min(task_suite.n_tasks, args.max_tasks or task_suite.n_tasks)
        n_trials = args.num_trials_per_task
        task_successes = [0] * num_tasks
        total_successes = 0
        total_episodes = 0
        for rank in range(len(gpu_list)):
            result_file = Path(tmpdir) / f"worker_{rank}.json"
            with open(result_file) as f:
                for row in json.load(f):
                    task_id = row["task_id"]
                    task_successes[task_id] += int(row["success"])
                    total_successes += int(row["success"])
                    total_episodes += 1
        with open(log_path, "w") as log_file:
            for task_id in range(num_tasks):
                task = task_suite.get_task(task_id)
                desc = task.language[:40] + "..." if len(task.language) > 40 else task.language
                msg = f"Task {task_id} ({desc}): {task_successes[task_id]}/{n_trials}"
                logger.info(msg)
                log_file.write(msg + "\n")
            sr = total_successes / total_episodes if total_episodes else 0
            summary = f"Overall success rate: {total_successes}/{total_episodes} = {sr:.2%}"
            logger.info(summary)
            log_file.write(summary + "\n")
        logger.info(f"Log written to {log_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint dir (e.g. checkpoints/beta_vla_libero/6000)")
    p.add_argument("--config", type=Path, default=Path("configs/train_beta_vla_libero.yaml"))
    p.add_argument("--task_suite", "--task_suite_name", dest="task_suite", default="libero_spatial",
                   help="Task suite name, or 'all' for all 6 suites")
    p.add_argument("--num_trials_per_task", type=int, default=50)
    p.add_argument("--replan_steps", type=int, default=5)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--num_ode_steps", type=int, default=15, help="ODE steps in action head (default 5 for eval speed, 10 for quality)")
    p.add_argument("--invert_gripper", action="store_true", default=True)
    p.add_argument("--no_invert_gripper", action="store_false", dest="invert_gripper")
    p.add_argument("--norm_stats", type=Path, default=None, help="norm_stats.json path (for model trained WITH action norm)")
    p.add_argument("--no_norm_stats", action="store_true", help="Disable norm_stats (for model trained WITHOUT action norm)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--video_out_path", type=Path, default=Path("data/libero/videos/7kchkptbest_v2"), help="Save rollout videos to this dir")
    p.add_argument("--no_video", action="store_true", help="Disable video saving")
    p.add_argument("--log_dir", type=Path, default=Path("./experiments/logs"))
    p.add_argument("--verbose", action="store_true", help="Save step-by-step debug log per episode")
    p.add_argument("--max_tasks", type=int, default=None, help="Limit tasks for quick testing")
    p.add_argument("--gpu", type=int, default=0, help="GPU index (single-GPU mode).")
    p.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids for multi-GPU eval, e.g. 4,5,6,7")
    p.add_argument("--worker_rank", type=int, default=None, help="Internal: worker index (0..worker_total-1).")
    p.add_argument("--worker_total", type=int, default=None, help="Internal: total number of workers.")
    p.add_argument("--result_file", type=Path, default=None, help="Internal: JSON file to write (task_id, trial_id, success).")
    args = p.parse_args()

    # Resolve suites to run
    if args.task_suite.lower() == "all":
        suites_to_run = ALL_SUITES
    else:
        suites_to_run = [args.task_suite]

    # Multi-GPU launcher: spawn one process per GPU (per suite)
    gpu_list = None
    if args.gpus:
        gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if gpu_list is not None and len(gpu_list) > 1:
        for suite_name in suites_to_run:
            args.task_suite = suite_name
            _run_multi_gpu_launcher(gpu_list, args)
        return

    # Single-GPU: --gpus 7 需在 import torch 后、首次 cuda 使用前设置
    if gpu_list is not None and len(gpu_list) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[0])

    # Single-GPU or worker mode
    cfg = load_beta_vla_config(args.config)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    model, _ = load_model(args.checkpoint, device, model_config=cfg.model)
    if args.no_norm_stats:
        norm_stats = None  # 旧模型：训练时未做 action 归一化，eval 也不做反归一化
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
    if norm_stats is not None:
        logger.info("Using norm_stats for action unnormalization (model trained WITH action norm)")
    else:
        logger.info("No norm_stats (model trained WITHOUT action norm)")
    tokenizer = get_tokenizer(cfg.model.language.model_name)

    save_video = not args.no_video
    if save_video:
        try:
            import imageio  # noqa: F401
        except ImportError:
            logger.warning("imageio not installed, disabling video save. pip install imageio")
            save_video = False
    video_out_path = args.video_out_path if save_video else None

    # Worker mode: only run our share, write results to result_file, then exit
    if args.worker_rank is not None and args.worker_total is not None and args.result_file is not None:
        task_suite_cls = benchmark.get_benchmark_dict()[args.task_suite]
        task_suite = task_suite_cls()
        num_tasks = min(task_suite.n_tasks, args.max_tasks or task_suite.n_tasks)
        # max_steps = TASK_MAX_STEPS.get(args.task_suite, 300)

        max_steps = 600
        _run_worker(
            args=args,
            task_suite=task_suite,
            num_tasks=num_tasks,
            max_steps=max_steps,
            save_video=save_video,
            video_out_path=video_out_path,
            model=model,
            tokenizer=tokenizer,
            norm_stats=norm_stats,
            device=device,
        )
        return

    args.log_dir.mkdir(parents=True, exist_ok=True)
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    # Single-GPU: loop over suites (model loaded once, reused)
    for suite_name in suites_to_run:
        args.task_suite = suite_name
        task_suite_cls = benchmark.get_benchmark_dict()[args.task_suite]
        task_suite = task_suite_cls()
        num_tasks = min(task_suite.n_tasks, args.max_tasks or task_suite.n_tasks)
        max_steps = TASK_MAX_STEPS.get(args.task_suite, 300)

        log_path = args.log_dir / f"eval_{args.task_suite}_{args.checkpoint.name}_{run_ts}.txt"
        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write(f"Eval started at {run_ts}\n")
        log_file.flush()
        # 将 logger 输出同时写入文件
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

        debug_log_dir = None
        if args.verbose:
            debug_log_dir = args.log_dir / f"eval_{args.task_suite}_{args.checkpoint.name}_{run_ts}" / "debug"

        logger.info("=" * 60)
        logger.info(f"Eval suite: {args.task_suite} ({num_tasks} tasks)")
        logger.info("=" * 60)

        total_episodes, total_successes = 0, 0
        task_successes = []

        for task_id in tqdm(range(num_tasks), desc="Tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=256)

            task_ok = 0
            for ep in tqdm(range(args.num_trials_per_task), desc=f"T{task_id} trials", leave=False):
                debug_path = None
                if debug_log_dir is not None:
                    safe_desc = task_description[:30].replace(" ", "_").replace(".", "_")
                    debug_path = debug_log_dir / f"task{task_id}_ep{ep}_{safe_desc}.log"
                success, replay_images = run_episode(
                    env,
                    initial_states[ep],
                    task_description,
                    model,
                    tokenizer,
                    norm_stats,
                    args.replan_steps,
                    args.num_steps_wait,
                    max_steps,
                    args.invert_gripper,
                    device,
                    save_video=save_video,
                    num_ode_steps=args.num_ode_steps,
                    debug_log_path=debug_path,
                )
                if save_video and replay_images and video_out_path:
                    save_rollout_video(
                        replay_images,
                        ep,
                        success,
                        task_description,
                        video_out_path,
                        log_file,
                    )
                task_ok += int(success)
                total_episodes += 1
                total_successes += int(success)

            task_successes.append(task_ok / args.num_trials_per_task)
            msg = f"Task {task_id} ({task_description[:40]}...): {task_ok}/{args.num_trials_per_task}"
            logger.info(msg)
            env.close()

        sr = total_successes / total_episodes if total_episodes else 0
        summary = f"Overall success rate: {total_successes}/{total_episodes} = {sr:.2%}"
        logger.info(summary)
        log_file.close()
        logger.removeHandler(file_handler)
        logger.info(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
