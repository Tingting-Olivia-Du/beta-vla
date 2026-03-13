"""Beta-VLA model.

Pipeline:
  images  → PaliGemmaVisionTower → vision_projector  ─┐
                                                        ├─► VGGT-1B ──► Gemma-300M action head ──► actions
  prompt  → Qwen3-0.6B ──────────► language_projector ─┘
                                                                  ▲
                                                         state + noisy_actions + t
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from betavla.models.action_head import FlowMatchingActionHeadConfig, OpenPIFlowMatchingActionHead
from betavla.models.language_encoder import LanguageEncoderConfig, QwenLanguageEncoder
from betavla.models.vggt_backbone import VGGTBackbone, VGGTBackboneConfig
from betavla.models.vision_tower import PaliGemmaVisionTower, PaliGemmaVisionTowerConfig
from betavla.data.types import ObservationBatch


@dataclass(frozen=True)
class BetaVLAConfig:
    action_dim: int = 7
    action_horizon: int = 10
    state_dim: int = 8
    gripper_loss_weight: float = 5.0
    freeze_vision: bool = True
    freeze_language: bool = False
    freeze_vggt: bool = False
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    )
    lora_target_modules_vggt: list[str] = field(
        default_factory=lambda: ["qkv", "proj", "fc1", "fc2"]
    )
    lora_on_language: bool = True
    lora_on_vggt: bool = True
    vision: PaliGemmaVisionTowerConfig = field(default_factory=PaliGemmaVisionTowerConfig)
    language: LanguageEncoderConfig = field(default_factory=LanguageEncoderConfig)
    vggt: VGGTBackboneConfig = field(default_factory=VGGTBackboneConfig)


def _gelu_mlp(in_dim: int, out_dim: int) -> nn.Module:
    """2-layer GELU MLP projector."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=True),
        nn.GELU(),
        nn.Linear(out_dim, out_dim, bias=True),
    )


def _set_frozen(module: nn.Module, freeze: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(not freeze)


def _inject_lora(
    module: nn.Module,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
) -> nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError(
            "LoRA requires `peft`. Install it: pip install peft"
        ) from e
    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(module, cfg)


class BetaVLAModel(nn.Module):
    """Full Beta-VLA model: vision + language → VGGT → flow-matching action head."""

    def __init__(self, cfg: BetaVLAConfig):
        super().__init__()
        self.cfg = cfg

        # Vision
        self.vision_tower = PaliGemmaVisionTower(cfg.vision)

        # Language
        self.language_encoder = QwenLanguageEncoder(cfg.language)

        # Backbone
        self.vggt_backbone = VGGTBackbone(cfg.vggt)

        # Projectors: align vision/language dims → VGGT hidden size
        self.vision_projector = _gelu_mlp(
            self.vision_tower.embed_dim, self.vggt_backbone.hidden_size
        )
        self.language_projector = _gelu_mlp(
            self.language_encoder.hidden_size, self.vggt_backbone.hidden_size
        )

        # Action head
        self.action_head = OpenPIFlowMatchingActionHead(
            FlowMatchingActionHeadConfig(
                action_dim=cfg.action_dim,
                action_horizon=cfg.action_horizon,
                hidden_size=self.vggt_backbone.hidden_size,
                state_dim=cfg.state_dim,
                gripper_loss_weight=cfg.gripper_loss_weight,
            )
        )

        # Freeze before LoRA so frozen weights don't get adapters
        _set_frozen(self.vision_tower, cfg.freeze_vision)
        _set_frozen(self.language_encoder, cfg.freeze_language)
        _set_frozen(self.vggt_backbone, cfg.freeze_vggt)

        # LoRA injection
        if cfg.use_lora and cfg.lora_on_language and not cfg.freeze_language:
            self.language_encoder.model = _inject_lora(
                self.language_encoder.model,
                r=cfg.lora_r,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
            )
        if cfg.use_lora and cfg.lora_on_vggt and not cfg.freeze_vggt:
            self.vggt_backbone.model = _inject_lora(
                self.vggt_backbone.model,
                r=cfg.lora_r,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules_vggt,
            )

    # ------------------------------------------------------------------
    # Attention mask helpers
    # ------------------------------------------------------------------

    def _build_attn_mask(
        self,
        vision_tokens: torch.Tensor,
        text_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Concatenate all-ones vision mask with text padding mask."""
        B, V, _ = vision_tokens.shape
        vmask = torch.ones(B, V, device=vision_tokens.device, dtype=torch.long)
        if text_mask is None:
            return vmask
        return torch.cat([vmask, text_mask.to(vmask.dtype)], dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, observation: ObservationBatch) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (prefix_tokens, prefix_pad_mask) for the action head."""
        vision_tokens = self.vision_tower(observation.images, observation.image_masks)
        text_tokens = self.language_encoder(
            observation.tokenized_prompt, observation.tokenized_prompt_mask
        )

        vision_tokens = self.vision_projector(vision_tokens)
        text_tokens = self.language_projector(text_tokens)

        fused = torch.cat([vision_tokens, text_tokens], dim=1)
        attn_mask = self._build_attn_mask(vision_tokens, observation.tokenized_prompt_mask)

        prefix_tokens = self.vggt_backbone(fused, attn_mask)

        # Build pad mask for action head (True = real token)
        if observation.tokenized_prompt_mask is not None:
            B, V, _ = vision_tokens.shape
            vmask = torch.ones(B, V, device=vision_tokens.device, dtype=torch.bool)
            prefix_pad_mask = torch.cat(
                [vmask, observation.tokenized_prompt_mask.bool().to(vmask.device)], dim=1
            )
        else:
            prefix_pad_mask = None

        return prefix_tokens, prefix_pad_mask

    def forward(
        self,
        observation: ObservationBatch,
        actions: torch.Tensor | None = None,
        num_ode_steps: int = 10,
        return_loss_details: bool = False,
    ) -> dict[str, Any]:
        prefix_tokens, prefix_pad_mask = self.encode(observation)
        state = observation.state

        if actions is None:
            sampled = self.action_head.sample(
                prefix_tokens, state,
                num_steps=num_ode_steps,
                prefix_pad_mask=prefix_pad_mask,
            )
            return {"actions": sampled}

        loss_out = self.action_head.compute_loss(
            prefix_tokens, state, actions,
            prefix_pad_mask=prefix_pad_mask,
            return_details=return_loss_details,
        )
        if return_loss_details:
            return loss_out
        return {"loss": loss_out}
