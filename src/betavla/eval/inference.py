"""Inference utilities for Beta-VLA."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from betavla.data.types import ObservationBatch
from betavla.models.beta_vla_model import BetaVLAModel, BetaVLAConfig


def _img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 [0,255] or float [0,1] -> [0,1] float tensor (1,H,W,C)."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if np.issubdtype(arr.dtype, np.floating) and arr.max() > 1.0:
        arr = arr.astype(np.float32) / 255.0
    elif not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)


def _quantile_normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """(x - q01) / (q99 - q01 + eps) * 2 - 1 -> [-1, 1]"""
    span = np.asarray(q99, dtype=np.float32) - np.asarray(q01, dtype=np.float32) + 1e-6
    return (np.asarray(x, dtype=np.float32) - np.asarray(q01, dtype=np.float32)) / span * 2.0 - 1.0


def _quantile_unnormalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """[-1,1] -> original scale: (x + 1) / 2 * (q99 - q01) + q01"""
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    return (np.asarray(x, dtype=np.float32) + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def load_model(
    checkpoint_dir: Path, device: torch.device, model_config: BetaVLAConfig | None = None
) -> tuple[BetaVLAModel, BetaVLAConfig]:
    """Load model from checkpoint."""
    import safetensors.torch

    ckpt_dir = Path(checkpoint_dir)
    if (ckpt_dir / "best").exists():
        ckpt_dir = ckpt_dir / "best"
    elif (ckpt_dir / "6000").exists():
        ckpt_dir = ckpt_dir / "6000"

    cfg = model_config or BetaVLAConfig(action_dim=7, action_horizon=10, state_dim=8)
    model = BetaVLAModel(cfg)
    safetensors.torch.load_model(model, ckpt_dir / "model.safetensors", device=str(device))
    model = model.to(device)  # 确保所有参数/buffer 在 device 上（load_state_dict 后部分可能仍在 CPU）
    model.eval()
    return model, cfg


def get_tokenizer(tokenizer_name: str):
    """Load tokenizer once (cached per process). Avoid loading on every predict()."""
    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def predict(
    model: BetaVLAModel,
    base_img: np.ndarray,
    wrist_img: np.ndarray,
    prompt: str,
    tokenizer_name: str,
    state: np.ndarray,
    device: torch.device,
    norm_stats: dict | None = None,
    replan_steps: int = 5,
    tokenizer=None,
    num_ode_steps: int = 10,
) -> np.ndarray:
    """Predict action chunk. Returns (replan_steps, 7) array.
    Pass tokenizer to avoid loading on every call (major speedup)."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_t = _img_to_tensor(base_img)
    wrist_t = _img_to_tensor(wrist_img)
    state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if norm_stats is not None and "state" in norm_stats:
        ns = norm_stats["state"]
        state_arr = _quantile_normalize(state_arr, ns["q01"], ns["q99"])
    state_t = torch.from_numpy(state_arr.reshape(1, -1))
    tokenized = tokenizer(
        [prompt],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    obs = ObservationBatch(
        images={"base_0_rgb": base_t, "left_wrist_0_rgb": wrist_t},
        image_masks={
            "base_0_rgb": torch.ones(1, dtype=torch.bool),
            "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool),
        },
        state=state_t,
        tokenized_prompt=tokenized["input_ids"],
        tokenized_prompt_mask=tokenized["attention_mask"].to(torch.bool),
    )
    obs = obs.to(device)

    with torch.no_grad():
        out = model(obs, actions=None, num_ode_steps=num_ode_steps)
    actions = out["actions"][0].cpu().numpy()

    if norm_stats is not None and "action" in norm_stats:
        ns = norm_stats["action"]
        actions = _quantile_unnormalize(actions, ns["q01"], ns["q99"])

    return actions[:replan_steps]


def process_action_for_env(action: np.ndarray, invert_gripper: bool = True) -> np.ndarray:
    """Prepare action for env.step: gripper [0,1] -> [-1,+1], optional invert."""
    a = action.copy()
    a[-1] = 2.0 * a[-1] - 1.0
    if a[-1] >= 0:
        a[-1] = 1.0
    else:
        a[-1] = -1.0
    if invert_gripper:
        a[-1] = -a[-1]
    return a


def load_norm_stats(path: Path | str | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)
