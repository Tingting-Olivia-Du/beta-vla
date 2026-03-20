"""LIBERO dataset loader for Beta-VLA training.

Loads physical-intelligence/libero from HuggingFace (LeRobot format).
Builds a correct action chunk by reading consecutive frames per episode.
"""
from __future__ import annotations

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


@dataclass(frozen=True)
class LiberoDatasetConfig:
    repo_id: str = "physical-intelligence/libero"
    split: str = "train"
    num_workers: int = 4
    max_token_len: int = 128
    max_samples: int | None = None
    state_dim: int = 8
    norm_stats_path: str | Path | None = None


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

        # Norm stats
        self._norm: dict[str, NormStats] | None = load_norm_stats(cfg.norm_stats_path)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], np.ndarray]:
        row_idx, pos, ep = self._index[idx]
        sample = self._get(row_idx)

        # Images
        base_arr = _pick(sample, self.IMAGE_KEYS)
        if base_arr is None:
            raise KeyError(f"No base image found in sample keys: {list(sample.keys())}")
        wrist_arr = _pick(sample, self.WRIST_KEYS)
        if wrist_arr is None:
            wrist_arr = base_arr  # fall back to base (masked later)

        # State
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

        # Action chunk: read consecutive frames (no zero-padding)
        ep_row_list = self._ep_rows[ep]
        ep_len = len(ep_row_list)
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

        # Prompt
        prompt_raw = _pick(sample, self.PROMPT_KEYS)
        prompt = str(prompt_raw) if prompt_raw is not None else ""

        return {
            "base_img": _to_float32_image(base_arr),
            "wrist_img": _to_float32_image(wrist_arr),
            "state": state,
            "prompt": prompt,
        }, actions

    def collate_fn(
        self, batch: list[tuple[dict[str, Any], np.ndarray]]
    ) -> tuple[ObservationBatch, torch.Tensor]:
        inputs, actions_list = zip(*batch, strict=True)

        # Images: (B, H, W, C) float32 in [0, 1]
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
