"""LIBERO dataset loader for Beta-VLA training.

Loads physical-intelligence/libero from HuggingFace (LeRobot format).
Builds a correct action chunk by reading consecutive frames per episode.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from betavla.data.normalize import NormStats, load_norm_stats, normalize_quantile
from betavla.data.types import ObservationBatch

logger = logging.getLogger(__name__)


def _load_task_descriptions(repo_id: str) -> dict[int, str]:
    """Load task_index → task description mapping.

    The physical-intelligence/libero parquet files only contain a numeric
    ``task_index`` column — no text descriptions.  We download the
    ``meta/tasks.jsonl`` file from the HuggingFace Hub (works without
    the ``lerobot`` package).
    """
    import json

    # Primary: download tasks.jsonl directly from HF Hub
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename="meta/tasks.jsonl", repo_type="dataset")
        tasks: dict[int, str] = {}
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                tasks[int(entry["task_index"])] = entry["task"]
        if tasks:
            logger.info("Loaded %d task descriptions from %s/meta/tasks.jsonl", len(tasks), repo_id)
            return tasks
    except Exception as exc:
        logger.warning("Could not load tasks.jsonl from HF Hub: %s", exc)

    # Fallback: try lerobot metadata
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
        meta = LeRobotDatasetMetadata(repo_id)
        if meta.tasks:
            logger.info("Loaded %d task descriptions from LeRobot metadata", len(meta.tasks))
            return meta.tasks
    except Exception as exc:
        logger.warning("Could not load task descriptions from LeRobot metadata: %s", exc)

    logger.warning("No task descriptions available — prompts will be empty")
    return {}


@dataclass(frozen=True)
class LiberoDatasetConfig:
    repo_id: str = "physical-intelligence/libero"
    split: str = "train"
    num_workers: int = 4
    max_token_len: int = 128
    max_samples: int | None = None
    state_dim: int = 8
    norm_stats_path: str | Path | None = None
    temporal_frames: int = 1   # 1=single-frame (legacy), >1=multi-frame temporal
    temporal_stride: int = 5   # frame stride for temporal history (matches replan_steps)


def _to_float32_image(x: Any) -> np.ndarray:
    """Return HWC float32 in [0, 1] from any image input."""
    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D image, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32) / 255.0
    elif arr.max() > 1.0:
        arr = arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)


def _pick(sample: dict, keys: list[str]) -> Any | None:
    for k in keys:
        if k in sample and sample[k] is not None:
            return sample[k]
    return None


class LiberoDataset(Dataset):
    """Frame-by-frame LIBERO dataset with correct consecutive action chunks."""

    IMAGE_KEYS = ["observation/image", "image", "observation.image"]
    WRIST_KEYS = ["observation/wrist_image", "wrist_image", "observation.wrist_image"]
    STATE_KEYS = ["observation/state", "state", "observation.state"]
    ACTION_KEYS = ["actions", "action"]
    PROMPT_KEYS = ["prompt", "task", "instruction", "language_instruction"]

    def __init__(
        self,
        cfg: LiberoDatasetConfig,
        *,
        action_horizon: int,
        action_dim: int,
        tokenizer_name: str,
    ):
        self.cfg = cfg
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        # Load dataset
        if cfg.max_samples is not None:
            ds_stream = load_dataset(cfg.repo_id, split=cfg.split, streaming=True)
            self.ds = type("DS", (), {
                "__len__": lambda s: cfg.max_samples,
                "__getitem__": None,
            })()  # placeholder; overwritten below
            self.ds = list(ds_stream.take(cfg.max_samples))
            self._get = lambda i: self.ds[i]
            self._len = cfg.max_samples
        else:
            self.ds = load_dataset(cfg.repo_id, split=cfg.split)
            self._get = lambda i: self.ds[i]
            self._len = len(self.ds)

        # Task descriptions (task_index → text)
        self._task_descriptions = _load_task_descriptions(cfg.repo_id)

        # Norm stats
        self._norm: dict[str, NormStats] | None = load_norm_stats(cfg.norm_stats_path)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Log a sample prompt to verify task descriptions are loaded
        if self._task_descriptions:
            sample_row = self._get(0)
            task_idx = sample_row.get("task_index")
            if task_idx is not None:
                task_idx_int = int(np.asarray(task_idx).flat[0])
                logger.info("Sample prompt (task_index=%d): %r",
                            task_idx_int, self._task_descriptions.get(task_idx_int, ""))

        # Build episode index in a single linear scan.
        # We store (episode_index, frame_index) per row to sort without re-reading.
        ep_frame: list[tuple[int, int, int]] = []  # (ep_idx, frame_idx, row_idx)
        for i in tqdm(range(self._len), desc="Building episode index", unit="frames"):
            row = self._get(i)
            ep = int(np.asarray(row.get("episode_index", i)).flat[0])
            fi = int(np.asarray(row.get("frame_index", i)).flat[0])
            ep_frame.append((ep, fi, i))

        # Group and sort by frame_index within each episode
        ep_rows: dict[int, list[tuple[int, int]]] = {}  # ep -> [(frame_idx, row_idx)]
        for ep, fi, row_i in ep_frame:
            ep_rows.setdefault(ep, []).append((fi, row_i))
        for ep in ep_rows:
            ep_rows[ep].sort(key=lambda t: t[0])

        # Build flat index: (row_idx, position_in_episode, episode_id)
        self._ep_rows: dict[int, list[int]] = {
            ep: [t[1] for t in frames] for ep, frames in ep_rows.items()
        }
        self._index: list[tuple[int, int, int]] = []
        for ep, frames in ep_rows.items():
            for pos, (_, row_i) in enumerate(frames):
                self._index.append((row_i, pos, ep))

    def __len__(self) -> int:
        return len(self._index)

    def _load_images_for_frame(self, sample: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract base and wrist images from a sample row."""
        base_arr = _pick(sample, self.IMAGE_KEYS)
        if base_arr is None:
            raise KeyError(f"No base image found in sample keys: {list(sample.keys())}")
        wrist_arr = _pick(sample, self.WRIST_KEYS)
        if wrist_arr is None:
            wrist_arr = base_arr  # fall back to base (masked later)
        return _to_float32_image(base_arr), _to_float32_image(wrist_arr)

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], np.ndarray]:
        row_idx, pos, ep = self._index[idx]
        sample = self._get(row_idx)
        ep_row_list = self._ep_rows[ep]
        ep_len = len(ep_row_list)

        T = self.cfg.temporal_frames
        stride = self.cfg.temporal_stride

        if T > 1:
            # Load T frames: current frame + (T-1) history frames
            # Frame positions: [pos - (T-1)*stride, ..., pos - stride, pos]
            # Clamped to episode start (pos=0)
            base_frames: list[np.ndarray] = []
            wrist_frames: list[np.ndarray] = []
            for k in range(T):
                # k=0 is the oldest, k=T-1 is the current frame
                hist_pos = max(pos - (T - 1 - k) * stride, 0)
                hist_sample = self._get(ep_row_list[hist_pos])
                b, w = self._load_images_for_frame(hist_sample)
                base_frames.append(b)
                wrist_frames.append(w)
            # Stack: (T, H, W, C)
            base_img = np.stack(base_frames, axis=0)
            wrist_img = np.stack(wrist_frames, axis=0)
        else:
            # Single-frame: (H, W, C)
            base_img, wrist_img = self._load_images_for_frame(sample)

        # State (always from current frame)
        state_raw = _pick(sample, self.STATE_KEYS)
        if state_raw is None:
            state = np.zeros(self.cfg.state_dim, dtype=np.float32)
        else:
            state = np.asarray(state_raw, dtype=np.float32).reshape(-1)
            state = state[: self.cfg.state_dim]
            if state.shape[0] < self.cfg.state_dim:
                state = np.pad(state, (0, self.cfg.state_dim - state.shape[0]))
        if self._norm is not None and "state" in self._norm:
            state = normalize_quantile(state, self._norm["state"])

        # Action chunk: read consecutive frames (fixed)
        chunk: list[np.ndarray] = []
        for k in range(self.action_horizon):
            frame_pos = min(pos + k, ep_len - 1)  # clamp to last frame
            frame_row = self._get(ep_row_list[frame_pos])
            a_raw = _pick(frame_row, self.ACTION_KEYS)
            if a_raw is None:
                raise KeyError(f"No action found in sample keys: {list(frame_row.keys())}")
            a = np.asarray(a_raw, dtype=np.float32).reshape(-1)
            a = a[: self.action_dim]
            if a.shape[0] < self.action_dim:
                a = np.pad(a, (0, self.action_dim - a.shape[0]))
            chunk.append(a)
        actions = np.stack(chunk, axis=0)  # (action_horizon, action_dim)

        if self._norm is not None and "action" in self._norm:
            actions = normalize_quantile(actions, self._norm["action"])

        # Prompt — prefer text columns; fall back to task_index → description map
        prompt_raw = _pick(sample, self.PROMPT_KEYS)
        if prompt_raw is None and self._task_descriptions:
            task_idx = sample.get("task_index")
            if task_idx is not None:
                task_idx_int = int(np.asarray(task_idx).flat[0])
                prompt_raw = self._task_descriptions.get(task_idx_int)
        prompt = str(prompt_raw) if prompt_raw else ""

        return {
            "base_img": base_img,     # (H,W,C) or (T,H,W,C)
            "wrist_img": wrist_img,   # (H,W,C) or (T,H,W,C)
            "state": state,
            "prompt": prompt,
            "episode_index": ep,
            "frame_index": pos,
            "task_index": int(np.asarray(sample.get("task_index", -1)).flat[0]),
        }, actions

    def collate_fn(
        self, batch: list[tuple[dict[str, Any], np.ndarray]]
    ) -> tuple[ObservationBatch, torch.Tensor]:
        inputs, actions_list = zip(*batch, strict=True)

        # Images: (B, H, W, C) or (B, T, H, W, C) float32 in [0, 1]
        base = torch.from_numpy(np.stack([x["base_img"] for x in inputs], axis=0))
        wrist = torch.from_numpy(np.stack([x["wrist_img"] for x in inputs], axis=0))
        state = torch.from_numpy(np.stack([x["state"] for x in inputs], axis=0))
        prompts = [x["prompt"] for x in inputs]

        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=self.cfg.max_token_len,
            return_tensors="pt",
        )

        B = base.shape[0]
        obs = ObservationBatch(
            images={"base_0_rgb": base, "left_wrist_0_rgb": wrist},
            image_masks={
                "base_0_rgb": torch.ones(B, dtype=torch.bool),
                "left_wrist_0_rgb": torch.ones(B, dtype=torch.bool),
            },
            state=state,
            tokenized_prompt=tokenized["input_ids"],
            tokenized_prompt_mask=tokenized["attention_mask"].bool(),
        )
        actions = torch.from_numpy(np.stack(actions_list, axis=0)).float()
        return obs, actions


def create_dataloader(
    cfg: LiberoDatasetConfig,
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    tokenizer_name: str,
    dataset: LiberoDataset | None = None,
    sampler=None,
) -> DataLoader:
    if dataset is None:
        dataset = LiberoDataset(
            cfg,
            action_horizon=action_horizon,
            action_dim=action_dim,
            tokenizer_name=tokenizer_name,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )
