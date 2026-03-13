from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Union

import torch
from torch import nn

from betavla.models.action_head import FlowMatchingActionHeadConfig, OpenPIFlowMatchingActionHead
from betavla.models.language_encoder import LanguageEncoderConfig, QwenLanguageEncoder
from betavla.models.vggt_backbone import VGGTBackbone, VGGTBackboneConfig
from betavla.models.vision_tower import (
    OpenVLAVisionTower,
    PaliGemmaVisionTower,
    PaliGemmaVisionTowerConfig,
    VisionTowerConfig,
)


@dataclass(frozen=True)
class BetaVLAConfig:
    action_dim: int = 7  # libero, openpi use 32 but output 7
    action_horizon: int = 10
    state_dim: int = 8  # for action head state_proj
    gripper_loss_weight: float = 5.0  # Weight for gripper dim in flow-matching loss (1.0 = no weighting)
    freeze_vision: bool = False
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
    vision_tower_type: str = "openvla"  # "openvla" | "paligemma"
    vision: Union[VisionTowerConfig, PaliGemmaVisionTowerConfig] = field(default_factory=VisionTowerConfig)
    language: LanguageEncoderConfig = LanguageEncoderConfig()
    vggt: VGGTBackboneConfig = VGGTBackboneConfig()


def _fused_mlp_projector(vision_dim: int, target_dim: int) -> nn.Module:
    """OpenVLA-style fused projector: Linear -> GELU -> Linear -> GELU -> Linear (4x expansion)."""
    if vision_dim == target_dim:
        return nn.Identity()
    mid_dim = vision_dim * 4
    return nn.Sequential(
        nn.Linear(vision_dim, mid_dim, bias=True),
        nn.GELU(),
        nn.Linear(mid_dim, target_dim, bias=True),
        nn.GELU(),
        nn.Linear(target_dim, target_dim, bias=True),
    )


def _gelu_mlp_projector(in_dim: int, target_dim: int) -> nn.Module:
    """OpenVLA-style gelu-mlp: Linear -> GELU -> Linear."""
    return nn.Sequential(
        nn.Linear(in_dim, target_dim, bias=True),
        nn.GELU(),
        nn.Linear(target_dim, target_dim, bias=True),
    )


def _set_frozen(module: nn.Module, freeze: bool) -> None:
    for p in module.parameters():
        p.requires_grad = not freeze


def _inject_lora(
    module: nn.Module,
    *,
    enabled: bool,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
) -> nn.Module:
    if not enabled:
        return module

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError("LoRA is enabled but `peft` is not installed. Install it first (e.g. `uv add peft`).") from e

    peft_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(module, peft_cfg)


def _create_vision_tower(cfg: BetaVLAConfig) -> nn.Module:
    """Create vision tower based on vision_tower_type."""
    if cfg.vision_tower_type == "paligemma":
        if not isinstance(cfg.vision, PaliGemmaVisionTowerConfig):
            raise TypeError("vision must be PaliGemmaVisionTowerConfig when vision_tower_type='paligemma'")
        return PaliGemmaVisionTower(cfg.vision)
    if cfg.vision_tower_type == "openvla":
        if not isinstance(cfg.vision, VisionTowerConfig):
            raise TypeError("vision must be VisionTowerConfig when vision_tower_type='openvla'")
        return OpenVLAVisionTower(cfg.vision)
    raise ValueError(f"Unknown vision_tower_type: {cfg.vision_tower_type}")


class BetaVLAModel(nn.Module):
    """image->vision tower, language->encoder, fuse->VGGT, then flow-matching action head."""

    def __init__(self, cfg: BetaVLAConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_tower = _create_vision_tower(cfg)
        self.language_encoder = QwenLanguageEncoder(cfg.language)
        self.vggt_backbone = VGGTBackbone(cfg.vggt)
        # Gradient checkpointing to reduce activation memory (~50% savings)
        if cfg.vision_tower_type == "openvla":
            if hasattr(self.vision_tower.dino, "set_grad_checkpointing"):
                self.vision_tower.dino.set_grad_checkpointing(True)
            if hasattr(self.vision_tower.siglip, "set_grad_checkpointing"):
                self.vision_tower.siglip.set_grad_checkpointing(True)
        elif cfg.vision_tower_type == "paligemma":
            if hasattr(self.vision_tower.vision_tower, "gradient_checkpointing_enable"):
                self.vision_tower.vision_tower.gradient_checkpointing_enable()
        # Disable language grad checkpoint: DDP+LoRA+checkpointing causes "unused params" or "marked ready twice"
        # if hasattr(self.language_encoder.model, "gradient_checkpointing_enable"):
        #     self.language_encoder.model.gradient_checkpointing_enable()

        if cfg.use_lora and cfg.lora_on_language:
            self.language_encoder.model = _inject_lora(
                self.language_encoder.model,
                enabled=True,
                r=cfg.lora_r,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
            )
        if cfg.use_lora and cfg.lora_on_vggt:
            self.vggt_backbone.model = _inject_lora(
                self.vggt_backbone.model,
                enabled=True,
                r=cfg.lora_r,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules_vggt,
            )

        # Vision: OpenVLA-style fused-gelu-mlp (DINO+SigLIP fused tower)
        self.vision_projector = _fused_mlp_projector(
            self.vision_tower.embed_dim, self.vggt_backbone.hidden_size
        )
        # Language: gelu-mlp (OpenVLA-style 2-layer)
        self.language_projector = _gelu_mlp_projector(
            self.language_encoder.hidden_size, self.vggt_backbone.hidden_size
        )

        self.action_head = OpenPIFlowMatchingActionHead(
            FlowMatchingActionHeadConfig(
                action_dim=cfg.action_dim,
                action_horizon=cfg.action_horizon,
                hidden_size=self.vggt_backbone.hidden_size,
                state_dim=cfg.state_dim,
                gripper_loss_weight=cfg.gripper_loss_weight,
            )
        )

        _set_frozen(self.vision_tower, cfg.freeze_vision)
        _set_frozen(self.language_encoder, cfg.freeze_language)
        _set_frozen(self.vggt_backbone, cfg.freeze_vggt)

    def _build_attention_mask(self, vision_tokens: torch.Tensor, text_mask: torch.Tensor | None) -> torch.Tensor | None:
        batch, vision_len, _ = vision_tokens.shape
        vision_mask = torch.ones(batch, vision_len, device=vision_tokens.device, dtype=torch.long)
        if text_mask is None:
            return vision_mask
        return torch.cat([vision_mask, text_mask.to(vision_mask.dtype)], dim=1)

    def encode(self, observation) -> torch.Tensor:
        vision_tokens = self.vision_tower(observation.images, observation.image_masks)
        text_tokens = self.language_encoder(observation.tokenized_prompt, observation.tokenized_prompt_mask)

        vision_tokens = self.vision_projector(vision_tokens)
        text_tokens = self.language_projector(text_tokens)
        fused = torch.cat([vision_tokens, text_tokens], dim=1)
        attn_mask = self._build_attention_mask(vision_tokens, observation.tokenized_prompt_mask)
        return self.vggt_backbone(fused, attn_mask)

    def forward(
        self,
        observation,
        actions: torch.Tensor | None = None,
        num_ode_steps: int = 10,
        return_loss_details: bool = False,
    ):
        prefix_tokens = self.encode(observation)
        state = observation.state
        if actions is None:
            return {"actions": self.action_head.sample(prefix_tokens, state, num_steps=num_ode_steps)}
        loss_out = self.action_head.compute_loss(prefix_tokens, state, actions, return_details=return_loss_details)
        if return_loss_details:
            return loss_out
        return {"loss": loss_out}
