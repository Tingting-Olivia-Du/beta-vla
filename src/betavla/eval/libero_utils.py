"""Utils for LIBERO benchmark evaluation."""

import math
import os
import time
from pathlib import Path

import numpy as np

try:
    import imageio
except ImportError:
    imageio = None
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_libero_env(task, resolution: int = 256):
    """Create LIBERO env and task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)
    return env, task_description


LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def get_libero_image(obs: dict) -> np.ndarray:
    """Third-person image, 180° rotate to match train."""
    img = obs["agentview_image"]
    return img[::-1, ::-1]


def get_libero_wrist_image(obs: dict) -> np.ndarray:
    """Wrist camera image, 180° rotate."""
    img = obs["robot0_eye_in_hand_image"]
    return img[::-1, ::-1]


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) to axis-angle. From robosuite."""
    q = np.asarray(quat)
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def prepare_state(obs: dict) -> np.ndarray:
    """8D state: eef_pos(3) + quat2axisangle(3) + gripper_qpos(2)."""
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )


def save_rollout_video(
    rollout_images: list[np.ndarray],
    episode_idx: int,
    success: bool,
    task_description: str,
    video_out_path: str | Path,
    log_file=None,
) -> str | None:
    """Save rollout as MP4. Returns path or None if imageio not installed."""
    if imageio is None:
        return None
    out_dir = Path(video_out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_desc = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = out_dir / f"{DATE_TIME}--ep{episode_idx}--success={success}--task={safe_desc}.mp4"
    writer = imageio.get_writer(str(mp4_path), fps=30)
    for img in rollout_images:
        writer.append_data(img)
    writer.close()
    msg = f"Saved rollout: {mp4_path}"
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()
    return str(mp4_path)
