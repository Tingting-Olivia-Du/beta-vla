"""Action head: OpenPI-style with full Gemma 300M action expert."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from transformers import GemmaConfig, GemmaModel


# OpenPI gemma_300m config (gemma_pytorch.py, models/gemma.py get_config)
GEMMA_300M_WIDTH = 1024
GEMMA_300M_DEPTH = 18
GEMMA_300M_MLP_DIM = 4096
GEMMA_300M_NUM_HEADS = 8
GEMMA_300M_NUM_KV_HEADS = 1
GEMMA_300M_HEAD_DIM = 256
PALIGEMMA_VOCAB_SIZE = 257152  # gemma_pytorch.py action_expert_config_hf


@dataclass(frozen=True)
class FlowMatchingActionHeadConfig:
    action_dim: int = 32
    action_horizon: int = 10
    hidden_size: int = 2048  # VGGT output dim; will be projected to GEMMA_300M_WIDTH
    state_dim: int = 8
    use_gemma_expert: bool = True  # True = full Gemma 300M (OpenPI), False = 2-layer decoder


def _posemb_sincos(
    pos: torch.Tensor,
    dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Sine-cosine positional embedding for timestep (OpenPI posemb_sincos)."""
    if dim % 2 != 0:
        raise ValueError(f"dim ({dim}) must be even")
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=pos.device, dtype=pos.dtype)
    period = min_period * (max_period / min_period) ** fraction
    phase = pos.unsqueeze(-1) / period.unsqueeze(0) * 2 * math.pi
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


def _make_prefix_lm_mask(prefix_len: int, suffix_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Prefix-LM: prefix bidirectional, suffix causal and can attend to prefix."""
    total = prefix_len + suffix_len
    mask = torch.zeros(total, total, device=device, dtype=dtype)
    for i in range(total):
        for j in range(total):
            if i < prefix_len:
                if j < prefix_len:
                    mask[i, j] = 0.0
                else:
                    mask[i, j] = float("-inf")
            else:
                if j <= i:
                    mask[i, j] = 0.0
                else:
                    mask[i, j] = float("-inf")
    return mask.unsqueeze(0).unsqueeze(0)


class OpenPIFlowMatchingActionHead(nn.Module):
    """OpenPI-style action head with full Gemma 300M action expert."""

    def __init__(self, cfg: FlowMatchingActionHeadConfig):
        super().__init__()
        self.cfg = cfg
        expert_dim = GEMMA_300M_WIDTH

        self.action_in_proj = nn.Linear(cfg.action_dim, expert_dim)
        self.state_proj = nn.Linear(cfg.state_dim, expert_dim)
        self.action_time_mlp_in = nn.Linear(2 * expert_dim, expert_dim)
        self.action_time_mlp_out = nn.Linear(expert_dim, expert_dim)
        self.action_out_proj = nn.Linear(expert_dim, cfg.action_dim)

        self.prefix_proj = nn.Linear(cfg.hidden_size, expert_dim)

        if cfg.use_gemma_expert:
            gemma_config = GemmaConfig(
                hidden_size=expert_dim,
                num_hidden_layers=GEMMA_300M_DEPTH,
                intermediate_size=GEMMA_300M_MLP_DIM,
                num_attention_heads=GEMMA_300M_NUM_HEADS,
                num_key_value_heads=GEMMA_300M_NUM_KV_HEADS,
                head_dim=GEMMA_300M_HEAD_DIM,
                vocab_size=PALIGEMMA_VOCAB_SIZE,
                max_position_embeddings=8192,
                hidden_activation="gelu_pytorch_tanh",  # gemma_pytorch.py action_expert_config_hf
            )
            self.gemma_expert = GemmaModel(gemma_config)
            self.gemma_expert.embed_tokens = None  # use inputs_embeds only (OpenPI gemma_pytorch.py)
            self._use_gemma = True
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=expert_dim,
                nhead=8,
                dim_feedforward=expert_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
            self._use_gemma = False

    def _embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Embed state + action tokens with timestep (OpenPI embed_suffix)."""
        state_token = self.state_proj(state).unsqueeze(1)
        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = _posemb_sincos(timestep, GEMMA_300M_WIDTH)
        time_tokens = time_emb.unsqueeze(1).expand(-1, self.cfg.action_horizon, -1)
        action_time = torch.cat([action_tokens, time_tokens], dim=-1)
        action_time = F.silu(self.action_time_mlp_in(action_time))
        action_time = F.silu(self.action_time_mlp_out(action_time))
        suffix = torch.cat([state_token, action_time], dim=1)
        return suffix

    def _predict_velocity(
        self,
        prefix_tokens: torch.Tensor,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Predict u_t given prefix, state, x_t, t."""
        suffix = self._embed_suffix(state, noisy_actions, timestep)
        prefix_proj = self.prefix_proj(prefix_tokens)
        prefix_len = prefix_proj.shape[1]
        suffix_len = suffix.shape[1]

        combined = torch.cat([prefix_proj, suffix], dim=1)
        position_ids = torch.arange(
            combined.shape[1],
            device=combined.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(combined.shape[0], -1)

        if self._use_gemma:
            attn_mask = _make_prefix_lm_mask(prefix_len, suffix_len, combined.device, combined.dtype)
            out = self.gemma_expert(
                inputs_embeds=combined,
                attention_mask=attn_mask,
                position_ids=position_ids,
            )
            suffix_out = out.last_hidden_state[:, -suffix_len:, :]
        else:
            causal_mask = torch.triu(
                torch.full((suffix_len, suffix_len), float("-inf"), device=suffix.device, dtype=suffix.dtype),
                diagonal=1,
            )
            suffix_out = self.decoder(
                tgt=suffix,
                memory=prefix_proj,
                tgt_mask=causal_mask,
            )

        action_out = suffix_out[:, -self.cfg.action_horizon :, :]
        return self.action_out_proj(action_out)

    def compute_loss(
        self,
        prefix_tokens: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Flow-matching loss (OpenPI: Beta(1.5,1) time, u_t target)."""
        batch = actions.shape[0]
        noise = torch.randn_like(actions, device=actions.device, dtype=actions.dtype)
        timestep = torch.distributions.Beta(1.5, 1.0).sample((batch,)).to(actions.device) * 0.999 + 0.001
        t = timestep[:, None, None]
        x_t = t * noise + (1.0 - t) * actions
        u_t = noise - actions
        pred_v = self._predict_velocity(prefix_tokens, state, x_t, timestep)
        return F.mse_loss(pred_v, u_t)

    @torch.no_grad()
    def sample(
        self,
        prefix_tokens: torch.Tensor,
        state: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """ODE solve: noise -> actions (OpenPI sample_actions)."""
        batch = prefix_tokens.shape[0]
        device = prefix_tokens.device
        dtype = prefix_tokens.dtype
        x_t = torch.randn(
            batch,
            self.cfg.action_horizon,
            self.cfg.action_dim,
            device=device,
            dtype=dtype,
        )
        dt = -1.0 / float(num_steps)
        time = torch.tensor(1.0, device=device, dtype=dtype)
        while time >= -dt / 2:
            timestep = time.expand(batch)
            v_t = self._predict_velocity(prefix_tokens, state, x_t, timestep)
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t
