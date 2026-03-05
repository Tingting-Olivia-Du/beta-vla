from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FlowMatchingActionHeadConfig:
    action_dim: int = 32
    action_horizon: int = 10
    hidden_size: int = 2048


class OpenPIFlowMatchingActionHead(nn.Module):
    """Flow-matching action head compatible with continuous action chunks."""

    def __init__(self, cfg: FlowMatchingActionHeadConfig):
        super().__init__()
        self.cfg = cfg
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.SiLU(),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.action_horizon * cfg.action_dim),
        )

    def _sinusoidal_time_embed(self, timestep: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freq = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=timestep.device) / max(half - 1, 1))
        phase = timestep[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if emb.shape[-1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[-1]))
        return emb

    def _predict_velocity(self, features: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        pooled = features.mean(dim=1)
        time_cond = self.time_mlp(self._sinusoidal_time_embed(timestep, pooled.shape[-1]))
        conditioned = pooled + time_cond
        out = self.action_mlp(conditioned)
        return out.view(-1, self.cfg.action_horizon, self.cfg.action_dim)

    def compute_loss(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch = actions.shape[0]
        noise = torch.randn_like(actions)
        timestep = torch.rand(batch, device=actions.device, dtype=actions.dtype) * 0.999 + 0.001
        t = timestep[:, None, None]
        x_t = t * noise + (1.0 - t) * actions
        u_t = noise - actions
        _ = x_t  # this head follows openpi objective with u_t target.
        pred_v = self._predict_velocity(features, timestep)
        return F.mse_loss(pred_v, u_t)

    @torch.no_grad() # 从噪声到动作的 ODE 求解
    def sample(self, features: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        batch = features.shape[0]
        x_t = torch.randn(batch, self.cfg.action_horizon, self.cfg.action_dim, device=features.device, dtype=features.dtype)
        dt = -1.0 / float(num_steps)
        time = torch.tensor(1.0, device=features.device, dtype=features.dtype)
        while time >= -dt / 2:
            timestep = time.expand(batch)
            v_t = self._predict_velocity(features, timestep)
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t
