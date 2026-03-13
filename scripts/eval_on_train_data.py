#!/usr/bin/env python3
"""Evaluate Beta-VLA on TRAINING data (physical-intelligence/libero train split).

Use this to verify the eval pipeline correctness: if the model performs well on
training episodes (same init states, same tasks), the pipeline is likely correct.

Usage:
  python scripts/eval_on_train_data.py --checkpoint checkpoints/beta_vla_libero/best --max_episodes 1
  python scripts/eval_on_train_data.py --checkpoint ... --max_episodes 50 --no_video  # faster

Speed: Use MUJOCO_GL=egl (GPU render, ~5-10 min/ep) instead of osmesa (CPU, ~2h/ep).

MUJOCO_GL=egl bash scripts/run_eval_libero.sh \
  --checkpoint checkpoints/beta_gripper/best \
  --task_suite libero_spatial \
  --num_trials_per_task 1 \
  --gpus 0 \
  --num_ode_steps 5




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

from betavla.data.types import ObservationBatch
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
from betavla.training.config import load_config as load_beta_vla_config

# physical-intelligence/libero: 40 tasks = libero_spatial(10) + libero_object(10) + libero_goal(10) + libero_10(10)
TASK_INDEX_TO_SUITE_AND_ID = []
for suite_name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    suite_cls = benchmark.get_benchmark_dict()[suite_name]
    suite = suite_cls()
    for task_id in range(suite.n_tasks):
        TASK_INDEX_TO_SUITE_AND_ID.append((suite_name, task_id, suite))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _pick_first(sample: dict, keys: list[str]):
    for k in keys:
        if k in sample and sample[k] is not None:
            return sample[k]
    return None


def _to_tensor_image(x) -> torch.Tensor:
    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() > 1.0:
            arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr.astype(np.float32))


def _quantile_normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    span = np.asarray(q99, dtype=np.float32) - np.asarray(q01, dtype=np.float32) + 1e-6
    return (np.asarray(x, dtype=np.float32) - np.asarray(q01, dtype=np.float32)) / span * 2.0 - 1.0


def _ensure_2d_actions(actions: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.shape[0] < action_horizon:
        pad = np.zeros((action_horizon - actions.shape[0], actions.shape[1]), dtype=actions.dtype)
        actions = np.concatenate([actions, pad], axis=0)
    actions = actions[:action_horizon, :]
    if actions.shape[1] < action_dim:
        pad = np.zeros((actions.shape[0], action_dim - actions.shape[1]), dtype=actions.dtype)
        actions = np.concatenate([actions, pad], axis=1)
    return actions[:, :action_dim]


def _tensor_to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return x


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
    save_video: bool = True,
    num_ode_steps: int = 5,
):
    """Run one episode. Returns (success, replay_images, stats)."""
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()
    action_queue = deque(maxlen=replan_steps)
    replay_images = [] if save_video else None
    t = 0
    env_step_time = 0.0
    predict_calls = 0
    predict_total_time = 0.0
    start_time = time.time()
    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            env_t0 = time.time()
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            env_step_time += time.time() - env_t0
            t += 1
            continue
        base_img = get_libero_image(obs)
        if save_video:
            replay_images.append(base_img)
        wrist_img = get_libero_wrist_image(obs)
        state = prepare_state(obs)
        if len(action_queue) == 0:
            pred_t0 = time.time()
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
                log_chunk=(predict_calls == 0),  # 只在第一次 replan 打印完整 chunk
            )
            predict_total_time += time.time() - pred_t0
            predict_calls += 1
            action_queue.extend(actions)
        action = action_queue.popleft()
        action_for_env = process_action_for_env(action, invert_gripper=invert_gripper)
        env_t0 = time.time()
        obs, reward, done, info = env.step(action_for_env.tolist())
        env_step_time += time.time() - env_t0
        if done:
            elapsed = time.time() - start_time
            return True, replay_images or [], {
                "steps_executed": t + 1,
                "elapsed_sec": elapsed,
                "env_step_sec_total": env_step_time,
                "predict_calls": predict_calls,
                "predict_sec_total": predict_total_time,
                "predict_sec_avg": (predict_total_time / predict_calls) if predict_calls > 0 else 0.0,
            }
        t += 1
    elapsed = time.time() - start_time
    return False, replay_images or [], {
        "steps_executed": t,
        "elapsed_sec": elapsed,
        "env_step_sec_total": env_step_time,
        "predict_calls": predict_calls,
        "predict_sec_total": predict_total_time,
        "predict_sec_avg": (predict_total_time / predict_calls) if predict_calls > 0 else 0.0,
    }


def _to_scalar(x):
    """Convert HF dataset value (possibly array) to scalar."""
    if isinstance(x, (list, np.ndarray)):
        return int(x[0]) if len(x) > 0 else 0
    return int(x)


def load_train_episodes(repo_id: str = "physical-intelligence/libero", max_episodes: int | None = None):
    """Load training dataset and return selected first-frame rows."""
    ds = load_dataset(repo_id, split="train")
    # Filter to first frame of each episode (frame_index == 0).
    # Break as soon as we have enough episodes (frame_index==0 rows are sparse —
    # each episode has many frames, so we stop scanning early instead of always
    # walking to the end of the dataset).
    first_frames = []
    for i in range(len(ds)):
        row = ds[i]
        frame_idx = _to_scalar(row.get("frame_index", 0))
        if frame_idx == 0:
            ep_idx = _to_scalar(row.get("episode_index", i))
            task_idx = _to_scalar(row.get("task_index", 0))
            first_frames.append(
                {
                    "row_index": i,
                    "episode_index": ep_idx,
                    "task_index": task_idx,
                    "frame_index": frame_idx,
                    "prompt": str(_pick_first(row, ["prompt", "task", "instruction"]) or ""),
                }
            )
            if max_episodes is not None and len(first_frames) >= max_episodes:
                break
    return ds, first_frames


def compute_supervised_loss_for_row(
    row: dict,
    *,
    model,
    tokenizer,
    device: torch.device,
    norm_stats: dict | None,
    state_dim: int,
    action_horizon: int,
    action_dim: int,
):
    base_img = _pick_first(row, ["observation/image", "image", "observation.image"])
    wrist_img = _pick_first(row, ["observation/wrist_image", "wrist_image", "observation.wrist_image"])
    if base_img is None:
        return None
    if wrist_img is None:
        wrist_img = base_img

    state = _pick_first(row, ["observation/state", "state", "observation.state"])
    if state is None:
        state = np.zeros((state_dim,), dtype=np.float32)
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] < state_dim:
        state = np.pad(state, (0, state_dim - state.shape[0]))
    state = state[:state_dim]
    if norm_stats is not None and "state" in norm_stats:
        ns = norm_stats["state"]
        state = _quantile_normalize(state, ns["q01"], ns["q99"])

    actions = _pick_first(row, ["actions", "action"])
    if actions is None:
        return None
    actions = _ensure_2d_actions(np.asarray(actions, dtype=np.float32), action_horizon, action_dim)
    if norm_stats is not None and "action" in norm_stats:
        ns = norm_stats["action"]
        actions = _quantile_normalize(actions, ns["q01"], ns["q99"])

    prompt = str(_pick_first(row, ["prompt", "task", "instruction"]) or "")
    tokenized = tokenizer(
        [prompt],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    obs = ObservationBatch(
        images={
            "base_0_rgb": _to_tensor_image(base_img).unsqueeze(0),
            "left_wrist_0_rgb": _to_tensor_image(wrist_img).unsqueeze(0),
        },
        image_masks={
            "base_0_rgb": torch.ones(1, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool),
        },
        state=torch.from_numpy(state.reshape(1, -1).astype(np.float32)),
        tokenized_prompt=tokenized["input_ids"],
        tokenized_prompt_mask=tokenized["attention_mask"].to(torch.bool),
    ).to(device)
    actions_t = torch.from_numpy(actions).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        out = model(obs, actions_t, return_loss_details=True)
    if out is None:
        return None
    return {
        "loss": float(out["loss"].item()),
        "weighted_per_dim": _tensor_to_list(out["weighted_per_dim"]),
        "unweighted_per_dim": _tensor_to_list(out["unweighted_per_dim"]),
        "weighted_per_step": _tensor_to_list(out["weighted_per_step"]),
        "unweighted_per_step": _tensor_to_list(out["unweighted_per_step"]),
        "weighted_per_step_dim": _tensor_to_list(out["weighted_per_step_dim"]),
        "unweighted_per_step_dim": _tensor_to_list(out["unweighted_per_step_dim"]),
        "gt_action_mean_per_dim": _tensor_to_list(out["gt_action_mean_per_dim"]),
        "gt_action_std_per_dim": _tensor_to_list(out["gt_action_std_per_dim"]),
        "gripper_dim": int(out["gripper_dim"]),
        "gripper_loss_weight": float(out["gripper_loss_weight"]),
    }


def build_episode_row_index_map(ds, target_episode_indices: set[int]) -> dict[int, list[int]]:
    ep_to_rows = {ep: [] for ep in target_episode_indices}
    remaining = set(target_episode_indices)
    # Track which episodes we've seen *all* frames for (episode_index increases
    # monotonically in the dataset, so once we've moved past an episode we won't
    # see it again). Stop as soon as all target episodes are fully collected.
    last_ep_seen = -1
    for i in range(len(ds)):
        row = ds[i]
        ep_idx = _to_scalar(row.get("episode_index", -1))
        if ep_idx in ep_to_rows:
            ep_to_rows[ep_idx].append(i)
        # When we advance to a new episode, the previous one is complete.
        if ep_idx != last_ep_seen:
            if last_ep_seen in remaining:
                remaining.discard(last_ep_seen)
                if not remaining:
                    break
            last_ep_seen = ep_idx
    return ep_to_rows


def aggregate_row_losses(row_losses: list[dict]) -> dict | None:
    if not row_losses:
        return None

    def _mean_array(key: str):
        arr = np.asarray([x[key] for x in row_losses], dtype=np.float64)
        return arr.mean(axis=0).tolist()

    agg = {
        "rows_used": len(row_losses),
        "loss": float(np.mean([x["loss"] for x in row_losses])),
        "weighted_per_dim": _mean_array("weighted_per_dim"),
        "unweighted_per_dim": _mean_array("unweighted_per_dim"),
        "weighted_per_step": _mean_array("weighted_per_step"),
        "unweighted_per_step": _mean_array("unweighted_per_step"),
        "weighted_per_step_dim": _mean_array("weighted_per_step_dim"),
        "unweighted_per_step_dim": _mean_array("unweighted_per_step_dim"),
        "gt_action_mean_per_dim": _mean_array("gt_action_mean_per_dim"),
        "gt_action_std_per_dim": _mean_array("gt_action_std_per_dim"),
        "gripper_dim": int(row_losses[0]["gripper_dim"]),
        "gripper_loss_weight": float(row_losses[0]["gripper_loss_weight"]),
    }
    gidx = agg["gripper_dim"]
    agg["weighted_gripper_dim_loss"] = float(agg["weighted_per_dim"][gidx])
    agg["unweighted_gripper_dim_loss"] = float(agg["unweighted_per_dim"][gidx])
    agg["gripper_value_mean"] = float(agg["gt_action_mean_per_dim"][gidx])
    agg["gripper_value_std"] = float(agg["gt_action_std_per_dim"][gidx])
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/beta_gripper.yaml"))
    p.add_argument("--max_episodes", type=int, default=50, help="Max episodes to eval (default 50)")
    p.add_argument("--repo_id", type=str, default="physical-intelligence/libero")
    p.add_argument("--replan_steps", type=int, default=5)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--num_ode_steps", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=220)
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
    p.add_argument("--log_dir", type=Path, default=Path("./experiments/eval_on_train/logs"))
    p.add_argument("--gpu", type=int, default=int(os.environ.get("EVAL_GPU", "0")))
    p.add_argument(
        "--log_loss",
        action="store_true",
        default=True,
        help="Compute supervised train loss for each selected episode row (default on)",
    )
    p.add_argument("--no_log_loss", action="store_false", dest="log_loss")
    p.add_argument(
        "--loss_scope",
        type=str,
        choices=["first_frame", "all_frames"],
        default="first_frame",
        help="When --log_loss is set: first frame only or all frames in each selected episode",
    )
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
    ds, episodes = load_train_episodes(repo_id=args.repo_id, max_episodes=args.max_episodes)
    if args.group_by_task:
        episodes = sorted(episodes, key=lambda x: (x["task_index"], x["episode_index"]))
    n_tasks = len({item["task_index"] for item in episodes})
    logger.info(f"Eval on {len(episodes)} training episodes")
    logger.info(
        "Eval settings: no_video=%s, max_steps=%d, num_ode_steps=%d, replan_steps=%d, resolution=%d, tasks=%d, action_horizon=%d, log_loss=%s, loss_scope=%s",
        args.no_video,
        args.max_steps,
        args.num_ode_steps,
        args.replan_steps,
        args.resolution,
        n_tasks,
        cfg.model.action_horizon,
        args.log_loss,
        args.loss_scope,
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    model_name = args.checkpoint.parent.name if args.checkpoint.name in ("best", "latest") else args.checkpoint.name
    out_dir = args.video_out_path / f"{model_name}_{run_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    meta_path = out_dir / "metadata.json"

    results = []
    total_success = 0
    total_loss = 0.0
    n_loss = 0
    last_task_idx = -1
    env = None
    episode_row_index_map = {}
    if args.log_loss and args.loss_scope == "all_frames":
        target_eps = {item["episode_index"] for item in episodes}
        logger.info("Building episode->rows map for %d episodes (all_frames loss mode)...", len(target_eps))
        episode_row_index_map = build_episode_row_index_map(ds, target_eps)

    for item in tqdm(episodes, desc="Episodes"):
        ep_idx = item["episode_index"]
        task_idx = item["task_index"]
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

        success, replay_images, rollout_stats = run_episode(
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

        row_indices = [item["row_index"]]
        if args.log_loss and args.loss_scope == "all_frames":
            row_indices = episode_row_index_map.get(ep_idx, []) or [item["row_index"]]
        loss_value = None
        loss_details = None
        if args.log_loss:
            row_losses = []
            for ridx in row_indices:
                row = ds[ridx]
                row_loss = compute_supervised_loss_for_row(
                    row,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    norm_stats=norm_stats,
                    state_dim=cfg.model.state_dim,
                    action_horizon=cfg.model.action_horizon,
                    action_dim=cfg.model.action_dim,
                )
                if row_loss is not None:
                    row_losses.append(row_loss)
            loss_details = aggregate_row_losses(row_losses)
            if loss_details is not None:
                loss_value = float(loss_details["loss"])
                total_loss += loss_value
                n_loss += 1

        total_success += int(success)
        record = {
            "row_index": item["row_index"],
            "episode_index": ep_idx,
            "frame_index": item["frame_index"],
            "task_index": task_idx,
            "task_suite": suite_name,
            "task_id_in_suite": task_id_in_suite,
            "task_description": task_description,
            "prompt": item["prompt"],
            "init_state_idx": init_state_idx,
            "success": success,
            "rollout": rollout_stats,
            "supervised_train_loss": loss_value,
            "loss_details": loss_details,
        }
        results.append(record)
        logger.info(
            "Episode done: ep=%d task=%d success=%s loss=%s elapsed=%.2fs",
            ep_idx,
            task_idx,
            success,
            f"{loss_value:.6f}" if loss_value is not None else "N/A",
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

    task_summary = {}
    for rec in results:
        key = f"{rec['task_suite']}#{rec['task_id_in_suite']}"
        if key not in task_summary:
            task_summary[key] = {
                "task_suite": rec["task_suite"],
                "task_id_in_suite": rec["task_id_in_suite"],
                "task_description": rec["task_description"],
                "episodes": 0,
                "success": 0,
                "loss_sum": 0.0,
                "loss_count": 0,
                "elapsed_sec_sum": 0.0,
                "steps_sum": 0,
                "predict_calls_sum": 0,
            }
        ts = task_summary[key]
        ts["episodes"] += 1
        ts["success"] += int(rec["success"])
        ts["elapsed_sec_sum"] += rec["rollout"]["elapsed_sec"]
        ts["steps_sum"] += int(rec["rollout"]["steps_executed"])
        ts["predict_calls_sum"] += int(rec["rollout"]["predict_calls"])
        if rec["supervised_train_loss"] is not None:
            ts["loss_sum"] += float(rec["supervised_train_loss"])
            ts["loss_count"] += 1

    task_metrics = []
    for ts in task_summary.values():
        task_metrics.append(
            {
                "task_suite": ts["task_suite"],
                "task_id_in_suite": ts["task_id_in_suite"],
                "task_description": ts["task_description"],
                "episodes": ts["episodes"],
                "success_rate": ts["success"] / ts["episodes"] if ts["episodes"] else 0.0,
                "avg_supervised_train_loss": (ts["loss_sum"] / ts["loss_count"]) if ts["loss_count"] else None,
                "avg_elapsed_sec": ts["elapsed_sec_sum"] / ts["episodes"] if ts["episodes"] else 0.0,
                "avg_steps": ts["steps_sum"] / ts["episodes"] if ts["episodes"] else 0.0,
                "avg_predict_calls": ts["predict_calls_sum"] / ts["episodes"] if ts["episodes"] else 0.0,
            }
        )
    task_metrics.sort(key=lambda x: (x["task_suite"], x["task_id_in_suite"]))

    total_elapsed = sum(float(rec["rollout"]["elapsed_sec"]) for rec in results)
    total_steps = sum(int(rec["rollout"]["steps_executed"]) for rec in results)
    total_predict_calls = sum(int(rec["rollout"]["predict_calls"]) for rec in results)
    total_predict_sec = sum(float(rec["rollout"]["predict_sec_total"]) for rec in results)
    total_env_step_sec = sum(float(rec["rollout"]["env_step_sec_total"]) for rec in results)
    gripper_weighted_losses = [
        float(rec["loss_details"]["weighted_gripper_dim_loss"])
        for rec in results
        if rec.get("loss_details") is not None
    ]
    gripper_unweighted_losses = [
        float(rec["loss_details"]["unweighted_gripper_dim_loss"])
        for rec in results
        if rec.get("loss_details") is not None
    ]

    metadata = {
        "run_ts": run_ts,
        "checkpoint": str(args.checkpoint),
        "max_episodes": args.max_episodes,
        "total_episodes": len(results),
        "total_success": total_success,
        "success_rate": total_success / len(results) if results else 0,
        "log_loss_enabled": args.log_loss,
        "loss_scope": args.loss_scope if args.log_loss else None,
        "action_horizon": cfg.model.action_horizon,
        "action_dim": cfg.model.action_dim,
        "state_dim": cfg.model.state_dim,
        "gripper_loss_weight": cfg.model.gripper_loss_weight,
        "avg_supervised_train_loss": (total_loss / n_loss) if n_loss > 0 else None,
        "loss_count": n_loss,
        "avg_weighted_gripper_dim_loss": (
            float(np.mean(gripper_weighted_losses)) if gripper_weighted_losses else None
        ),
        "avg_unweighted_gripper_dim_loss": (
            float(np.mean(gripper_unweighted_losses)) if gripper_unweighted_losses else None
        ),
        "avg_episode_elapsed_sec": (total_elapsed / len(results)) if results else 0.0,
        "avg_steps_per_episode": (total_steps / len(results)) if results else 0.0,
        "avg_predict_calls_per_episode": (total_predict_calls / len(results)) if results else 0.0,
        "predict_sec_total": total_predict_sec,
        "env_step_sec_total": total_env_step_sec,
        "task_metrics": task_metrics,
        "video_out_path": str(out_dir),
        "results_path": str(results_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    sr = total_success / len(results) if results else 0
    logger.info(f"Success rate: {total_success}/{len(results)} = {sr:.2%}")
    if n_loss > 0:
        logger.info("Avg supervised train loss: %.6f over %d samples", total_loss / n_loss, n_loss)
    logger.info(f"Results: {results_path}")
    logger.info(f"Metadata: {meta_path}")
    logger.info(f"Videos: {out_dir}")


if __name__ == "__main__":
    main()
