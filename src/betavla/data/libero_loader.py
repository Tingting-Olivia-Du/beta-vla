from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset, load_dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer

from betavla.data.types import ObservationBatch


def _pick_first(sample: dict[str, Any], keys: list[str]) -> Any | None:
    for k in keys:
        if k in sample and sample[k] is not None:
            return sample[k]
    return None


def _to_tensor_image(x: Any) -> torch.Tensor:
    """Convert image to [0,1] float tensor. Handles HF datasets: uint8 [0,255] or float [0,1]."""
    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() > 1.0:
            arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _quantile_normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """(x - q01) / (q99 - q01 + eps) * 2 - 1 -> [-1, 1]"""
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


@dataclass(frozen=True)
class LiberoLoaderConfig:
    repo_id: str = "physical-intelligence/libero"
    split: str = "train"
    num_workers: int = 2
    max_token_len: int = 128
    max_samples: int | None = None
    state_dim: int = 8  # eef_pos 3 + quat2axisangle 3 + gripper 2
    norm_stats_path: str | Path | None = None  # path to norm_stats.json for quantile norm


class LiberoTorchDataset(Dataset):
    def __init__(
        self,
        cfg: LiberoLoaderConfig,
        *,
        action_horizon: int,
        action_dim: int,
        state_dim: int,
        tokenizer_name: str,
    ):
        self.cfg = cfg
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        if cfg.max_samples is not None:
            # Use streaming + take to avoid downloading full dataset
            ds_stream = load_dataset(cfg.repo_id, split=cfg.split, streaming=True)
            ds_stream = ds_stream.take(cfg.max_samples)
            self.ds = HFDataset.from_list(list(ds_stream))
        else:
            self.ds = load_dataset(cfg.repo_id, split=cfg.split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._norm_stats: dict[str, dict[str, list[float]]] | None = None
        if cfg.norm_stats_path is not None:
            path = Path(cfg.norm_stats_path)
            if path.exists():
                with path.open() as f:
                    self._norm_stats = json.load(f)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], np.ndarray]:
        sample = self.ds[idx]

        base_img = _pick_first(sample, ["observation/image", "image", "observation.image"])
        wrist_img = _pick_first(sample, ["observation/wrist_image", "wrist_image", "observation.wrist_image"])
        if base_img is None:
            raise KeyError("Could not find base image key in LIBERO sample.")
        if wrist_img is None:
            wrist_img = base_img

        state = _pick_first(sample, ["observation/state", "state", "observation.state"])
        if state is None:
            state = np.zeros((self.state_dim,), dtype=np.float32)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] < self.state_dim:
            state = np.pad(state, (0, self.state_dim - state.shape[0]))
        state = state[: self.state_dim]
        if self._norm_stats is not None and "state" in self._norm_stats:
            ns = self._norm_stats["state"]
            state = _quantile_normalize(state, ns["q01"], ns["q99"])

        actions = _pick_first(sample, ["actions", "action"])
        if actions is None:
            raise KeyError("Could not find actions key in LIBERO sample.")
        actions = _ensure_2d_actions(np.asarray(actions, dtype=np.float32), self.action_horizon, self.action_dim)
        if self._norm_stats is not None and "action" in self._norm_stats: #normalize action
            ns = self._norm_stats["action"]
            actions = _quantile_normalize(actions, ns["q01"], ns["q99"])

        prompt = _pick_first(sample, ["prompt", "task", "instruction"])
        prompt = str(prompt) if prompt is not None else ""

        return {
            "base_img": _to_tensor_image(base_img),
            "wrist_img": _to_tensor_image(wrist_img),
            "state": torch.from_numpy(state),
            "prompt": prompt,
        }, actions

    def collate_fn(self, batch: list[tuple[dict[str, Any], np.ndarray]]) -> tuple[ObservationBatch, torch.Tensor]:
        inputs, actions = zip(*batch, strict=True)

        base = torch.stack([x["base_img"] for x in inputs], dim=0)
        wrist = torch.stack([x["wrist_img"] for x in inputs], dim=0)
        state = torch.stack([x["state"] for x in inputs], dim=0)
        prompts = [x["prompt"] for x in inputs]

        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding="max_length",
            max_length=self.cfg.max_token_len,
            return_tensors="pt",
        )

        obs = ObservationBatch(
            images={"base_0_rgb": base, "left_wrist_0_rgb": wrist},
            image_masks={
                "base_0_rgb": torch.ones(base.shape[0], dtype=torch.bool),
                "left_wrist_0_rgb": torch.ones(wrist.shape[0], dtype=torch.bool),
            },
            state=state,
            tokenized_prompt=tokenized["input_ids"],
            tokenized_prompt_mask=tokenized["attention_mask"].to(torch.bool),
        )
        return obs, torch.from_numpy(np.stack(actions, axis=0)).to(torch.float32)


def create_libero_dataloader(
    cfg: LiberoLoaderConfig,
    *,
    batch_size: int,
    action_horizon: int,
    action_dim: int,
    state_dim: int,
    tokenizer_name: str,
    dataset: LiberoTorchDataset | None = None,
    sampler: DistributedSampler | None = None,
) -> DataLoader:
    if dataset is None:
        dataset = LiberoTorchDataset(
            cfg,
            action_horizon=action_horizon,
            action_dim=action_dim,
            state_dim=state_dim,
            tokenizer_name=tokenizer_name,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
    )
