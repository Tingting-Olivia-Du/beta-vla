"""Inference utilities for Beta-VLA.

Key optimizations:
  - Tokenization cached per prompt string (CPU)
  - Language encoding cached per (prompt, device) (GPU)
  - Vision + VGGT prefix computed once per replan (images change each step)
  - action_head.sample() reuses prefix_tokens across all ODE denoising steps
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from betavla.data.normalize import NormStats, load_norm_stats, normalize_quantile, unnormalize_quantile
from betavla.models.model import BetaVLAConfig, BetaVLAModel


# ---------------------------------------------------------------------------
# Module-level caches (one per process, cleared on restart)
# ---------------------------------------------------------------------------

_token_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_lang_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}


def clear_caches() -> None:
    _token_cache.clear()
    _lang_cache.clear()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint: Path | str,
    device: torch.device,
    model_config: BetaVLAConfig | None = None,
) -> tuple[BetaVLAModel, BetaVLAConfig]:
    """Load model weights from a checkpoint directory."""
    import safetensors.torch

    ckpt = Path(checkpoint)
    # Accept either a step directory or a parent directory (auto-resolve best/latest)
    model_file = ckpt / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"No model.safetensors found at {ckpt}")

    cfg = model_config or BetaVLAConfig()
    model = BetaVLAModel(cfg)
    safetensors.torch.load_model(model, model_file, device=str(device))
    model.to(device)
    model.eval()
    return model, cfg


def get_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8/float → float32 HWC [0, 1], batched as (1, H, W, C)."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32) / 255.0
    elif arr.max() > 1.0:
        arr = arr.astype(np.float32) / 255.0
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: BetaVLAModel,
    base_img: np.ndarray,
    wrist_img: np.ndarray,
    prompt: str,
    unused_compat: str,         # legacy positional arg (tokenizer_name), kept for API compat
    state: np.ndarray,
    device: torch.device,
    norm_stats: dict[str, NormStats] | None = None,
    replan_steps: int = 5,
    tokenizer: AutoTokenizer | None = None,
    num_ode_steps: int = 5,
    max_token_len: int = 128,
    log_chunk: bool = False,
) -> np.ndarray:
    """Predict action chunk. Returns (replan_steps, action_dim) array in robot space."""
    if tokenizer is None:
        raise ValueError("tokenizer must be provided to predict()")

    # --- State normalisation ---
    state_np = np.asarray(state, dtype=np.float32).reshape(-1)
    if norm_stats is not None and "state" in norm_stats:
        state_np = normalize_quantile(state_np, norm_stats["state"])

    # --- Tokenise prompt (cached per prompt string) ---
    if prompt not in _token_cache:
        enc = tokenizer(
            [prompt],
            truncation=True,
            padding="max_length",
            max_length=max_token_len,
            return_tensors="pt",
        )
        _token_cache[prompt] = (enc["input_ids"], enc["attention_mask"].bool())
    token_ids, token_mask = _token_cache[prompt]

    # --- Language encoding (cached per prompt × device) ---
    lang_key = (prompt, str(device))
    if lang_key not in _lang_cache:
        ids = token_ids.to(device)
        mask = token_mask.to(device)
        raw_lang = model.language_encoder(ids, mask)         # (1, L, Dq)
        proj_lang = model.language_projector(raw_lang)       # (1, L, H)
        _lang_cache[lang_key] = (proj_lang, mask)            # keep on GPU
    proj_lang, lang_mask_gpu = _lang_cache[lang_key]

    # --- Vision (per step: images change) ---
    use_amp = device.type == "cuda"
    base_t = _img_to_tensor(base_img).to(device)
    wrist_t = _img_to_tensor(wrist_img).to(device)
    images = {"base_0_rgb": base_t, "left_wrist_0_rgb": wrist_t}
    image_masks = {
        "base_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones(1, device=device, dtype=torch.bool),
    }

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        vision_tokens = model.vision_tower(images, image_masks)   # (1, N_vis, Dv)
        vision_tokens = model.vision_projector(vision_tokens)     # (1, N_vis, H)

        fused = torch.cat([vision_tokens, proj_lang], dim=1)      # (1, N_vis+L, H)

        # VGGT attention mask (all-ones vision + text padding mask)
        B, V, _ = vision_tokens.shape
        vmask_long = torch.ones(B, V, device=device, dtype=torch.long)
        attn_mask = torch.cat([vmask_long, lang_mask_gpu.long()], dim=1)

        prefix_tokens = model.vggt_backbone(fused, attn_mask)     # (1, N_vis+L, H)

        # Prefix pad mask for action head (True = real token)
        vmask_bool = vmask_long.bool()
        prefix_pad_mask = torch.cat([vmask_bool, lang_mask_gpu], dim=1)

        # State
        state_t = torch.from_numpy(state_np).unsqueeze(0).to(device=device, dtype=torch.float32)

        # ODE solve: prefix_tokens reused across all denoising steps
        actions = model.action_head.sample(
            prefix_tokens, state_t,
            num_steps=num_ode_steps,
            prefix_pad_mask=prefix_pad_mask,
        )  # (1, action_horizon, action_dim)

    actions_np = actions[0].float().cpu().numpy()  # (action_horizon, action_dim)

    if log_chunk:
        import logging
        log = logging.getLogger("betavla.inference")
        log.info("[chunk_debug] raw normalized output:\n%s",
                 np.array2string(actions_np, precision=4, suppress_small=True))

    # --- Un-normalise actions ---
    if norm_stats is not None and "action" in norm_stats:
        actions_np = unnormalize_quantile(actions_np, norm_stats["action"])

    if log_chunk:
        import logging
        log = logging.getLogger("betavla.inference")
        norms = np.linalg.norm(actions_np, axis=-1)
        log.info("[chunk_debug] after unnorm per-step L2: %s",
                 np.array2string(norms, precision=4))

    return actions_np[:replan_steps]


def process_action_for_env(action: np.ndarray) -> np.ndarray:
    """Convert unnormalized model action to LIBERO env action.

    HuggingFace LIBERO data gripper: {-1.0=open, +1.0=close}
    After quantile unnorm, gripper is still in {-1, +1} (near-identity transform).
    LIBERO env expects: -1.0=open, +1.0=close  -- same convention.

    Binarize at 0.0 threshold (model output is continuous, snap to nearest integer).
    No inversion needed.
    """
    a = action.copy()
    a[-1] = -1.0 if a[-1] < 0.0 else 1.0
    return a
