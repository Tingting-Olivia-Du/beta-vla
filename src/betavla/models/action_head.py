"""Flow-matching action head: OpenPI-style with Gemma 300M action expert.

Fixes vs original:
  1. _make_prefix_lm_mask now accepts and applies a token padding mask so that
     padding positions cannot attend to or be attended by real tokens.
  2. position_ids are computed from the attention mask cumsum so padding tokens
     get position 0 (same slot) and don't shift real token positions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GemmaConfig, GemmaModel


# Gemma 300M config (matches OpenPI action_expert_config_hf)
_G300_WIDTH = 1024
_G300_DEPTH = 18
_G300_MLP_DIM = 4096
_G300_NUM_HEADS = 8
_G300_NUM_KV_HEADS = 1
_G300_HEAD_DIM = 256
_PALIGEMMA_VOCAB = 257152


@dataclass(frozen=True)
class FlowMatchingActionHeadConfig:
    action_dim: int = 7
    action_horizon: int = 10
    hidden_size: int = 2048          # VGGT output dim → projected to Gemma width
    state_dim: int = 8
    gripper_loss_weight: float = 5.0  # extra weight on gripper dim (index -1)


def _posemb_sincos(
    pos: torch.Tensor,
    dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Sine-cosine positional embedding for flow timestep (OpenPI posemb_sincos)."""
    if dim % 2 != 0:
        raise ValueError(f"dim ({dim}) must be even")
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=pos.device, dtype=pos.dtype)
    period = min_period * (max_period / min_period) ** fraction
    phase = pos.unsqueeze(-1) / period.unsqueeze(0) * 2 * math.pi
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


def _make_prefix_lm_mask(
    prefix_len: int,
    suffix_len: int,
    device: torch.device,
    dtype: torch.dtype,
    prefix_pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prefix-LM attention bias (additive, 0 = attend, -inf = block).

    prefix_pad_mask: bool tensor (B, prefix_len), True = real token.
    Padding positions in the prefix cannot attend to or be attended by anything.
    """
    total = prefix_len + suffix_len
    # Start with standard prefix-LM structure: prefix bidi, suffix causal
    mask = torch.zeros(total, total, device=device, dtype=dtype)
    # suffix tokens cannot attend to later suffix tokens
    for i in range(prefix_len, total):
        for j in range(i + 1, total):
            mask[i, j] = float("-inf")
    # prefix tokens cannot attend to suffix tokens
    mask[:prefix_len, prefix_len:] = float("-inf")

    # Apply padding mask: padded prefix positions are fully masked out
    if prefix_pad_mask is not None:
        # prefix_pad_mask: (B, prefix_len), True = real
        # Build per-batch mask; caller will add batch dim if needed
        # Here we return a shared mask suitable for non-padded case;
        # callers that have padding should call with prefix_pad_mask
        # Shape: (B, 1, total, total) — batched
        B = prefix_pad_mask.shape[0]
        batched = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, total, total).clone()
        pad_col = ~prefix_pad_mask  # (B, prefix_len) — True = pad
        # Rows corresponding to padded prefix positions → block everything
        batched[:, 0, :prefix_len, :][pad_col.unsqueeze(-1).expand(B, prefix_len, total)] = float("-inf")
        # Columns corresponding to padded prefix positions → block from suffix
        pad_col_suffix = pad_col.unsqueeze(1).unsqueeze(2).expand(B, 1, suffix_len, prefix_len)
        batched[:, 0, prefix_len:, :prefix_len][pad_col_suffix.view(B, suffix_len, prefix_len)] = float("-inf")
        return batched
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, total, total)


def _position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Compute position ids skipping padding: cumsum of real tokens, 0-indexed."""
    # mask: (B, L) bool, True = real token
    return (mask.long().cumsum(dim=1) - 1).clamp(min=0)


class OpenPIFlowMatchingActionHead(nn.Module):
    """OpenPI-style action head with full Gemma 300M action expert."""

    def __init__(self, cfg: FlowMatchingActionHeadConfig):
        super().__init__()
        self.cfg = cfg

        self.action_in_proj = nn.Linear(cfg.action_dim, _G300_WIDTH)
        self.state_proj = nn.Linear(cfg.state_dim, _G300_WIDTH)
        self.action_time_mlp_in = nn.Linear(2 * _G300_WIDTH, _G300_WIDTH)
        self.action_time_mlp_out = nn.Linear(_G300_WIDTH, _G300_WIDTH)
        self.action_out_proj = nn.Linear(_G300_WIDTH, cfg.action_dim)
        self.prefix_proj = nn.Linear(cfg.hidden_size, _G300_WIDTH)

        gemma_cfg = GemmaConfig(
            hidden_size=_G300_WIDTH,
            num_hidden_layers=_G300_DEPTH,
            intermediate_size=_G300_MLP_DIM,
            num_attention_heads=_G300_NUM_HEADS,
            num_key_value_heads=_G300_NUM_KV_HEADS,
            head_dim=_G300_HEAD_DIM,
            vocab_size=_PALIGEMMA_VOCAB,
            max_position_embeddings=8192,
            hidden_activation="gelu_pytorch_tanh",
        )
        self.gemma_expert = GemmaModel(gemma_cfg)
        self.gemma_expert.embed_tokens = None  # inputs_embeds only (OpenPI style)

    def _embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Embed [state_token, action_tokens] with timestep conditioning."""
        state_token = self.state_proj(state).unsqueeze(1)         # (B, 1, W)
        action_tokens = self.action_in_proj(noisy_actions)         # (B, H, W)
        time_emb = _posemb_sincos(timestep, _G300_WIDTH)           # (B, W)
        time_tokens = time_emb.unsqueeze(1).expand(-1, self.cfg.action_horizon, -1)
        action_time = torch.cat([action_tokens, time_tokens], dim=-1)  # (B, H, 2W)
        action_time = F.silu(self.action_time_mlp_in(action_time))
        action_time = F.silu(self.action_time_mlp_out(action_time))
        return torch.cat([state_token, action_time], dim=1)        # (B, 1+H, W)

    def _predict_velocity(
        self,
        prefix_tokens: torch.Tensor,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        prefix_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict flow-matching velocity u_t given prefix, state, x_t, t."""
        suffix = self._embed_suffix(state, noisy_actions, timestep)
        prefix_proj = self.prefix_proj(prefix_tokens)
        prefix_len = prefix_proj.shape[1]
        suffix_len = suffix.shape[1]

        combined = torch.cat([prefix_proj, suffix], dim=1)   # (B, P+S, W)
        B, total, _ = combined.shape

        # Attention mask: prefix-LM style with padding awareness
        attn_mask = _make_prefix_lm_mask(
            prefix_len, suffix_len,
            combined.device, combined.dtype,
            prefix_pad_mask=prefix_pad_mask,
        )

        # Position ids: skip padding slots
        if prefix_pad_mask is not None:
            suffix_mask = torch.ones(B, suffix_len, device=combined.device, dtype=torch.bool)
            full_mask = torch.cat([prefix_pad_mask, suffix_mask], dim=1)
        else:
            full_mask = torch.ones(B, total, device=combined.device, dtype=torch.bool)
        position_ids = _position_ids_from_mask(full_mask)

        out = self.gemma_expert(
            inputs_embeds=combined,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )
        suffix_out = out.last_hidden_state[:, -suffix_len:, :]    # (B, S, W)
        action_out = suffix_out[:, -self.cfg.action_horizon:, :]  # (B, H, W)
        return self.action_out_proj(action_out)                    # (B, H, action_dim)

    def compute_loss(
        self,
        prefix_tokens: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        prefix_pad_mask: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | dict:
        """Flow-matching loss: Beta(1.5,1) time sampling, MSE on velocity."""
        B = actions.shape[0]
        noise = torch.randn_like(actions)
        timestep = (
            torch.distributions.Beta(1.5, 1.0).sample((B,)).to(actions.device) * 0.999 + 0.001
        )
        t = timestep[:, None, None]
        x_t = t * noise + (1.0 - t) * actions
        u_t = noise - actions  # target velocity

        pred_v = self._predict_velocity(prefix_tokens, state, x_t, timestep, prefix_pad_mask)
        sq_err = (pred_v - u_t) ** 2

        # Optionally upweight gripper dimension (last dim)
        if self.cfg.gripper_loss_weight != 1.0 and self.cfg.action_dim >= 2:
            w = torch.ones_like(sq_err)
            w[..., -1] = self.cfg.gripper_loss_weight
            sq_err_weighted = sq_err * w
        else:
            sq_err_weighted = sq_err

        loss = sq_err_weighted.mean()
        if not return_details:
            return loss
        return {
            "loss": loss,
            "unweighted_loss": sq_err.mean(),
            "gripper_loss": sq_err[..., -1].mean(),
        }

    @torch.no_grad()
    def sample(
        self,
        prefix_tokens: torch.Tensor,
        state: torch.Tensor,
        num_steps: int = 10,
        prefix_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ODE solve: pure noise → actions (Euler, num_steps steps)."""
        B = prefix_tokens.shape[0]
        device = prefix_tokens.device
        dtype = prefix_tokens.dtype
        x_t = torch.randn(B, self.cfg.action_horizon, self.cfg.action_dim, device=device, dtype=dtype)
        dt = -1.0 / float(num_steps)
        time = torch.tensor(1.0, device=device, dtype=dtype)
        while time >= -dt / 2:
            timestep = time.expand(B)
            v_t = self._predict_velocity(prefix_tokens, state, x_t, timestep, prefix_pad_mask)
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t  # (B, action_horizon, action_dim)
