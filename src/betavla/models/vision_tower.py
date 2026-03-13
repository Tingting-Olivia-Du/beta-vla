"""Vision tower: PaliGemma SigLIP encoder + multi_modal_projector.

Same design as OpenPI: use PaliGemma's pretrained SigLIP vision encoder
and its built-in multi_modal_projector for image-text aligned features.
Output dim = projector output dim (2048 for paligemma2-3b).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _to_nchw(image: torch.Tensor) -> torch.Tensor:
    """Ensure image tensor is (B, C, H, W)."""
    if image.ndim != 4:
        raise ValueError(f"Expected 4-D image tensor, got shape {image.shape}")
    if image.shape[1] in (1, 3):
        return image  # already NCHW
    if image.shape[-1] in (1, 3):
        return image.permute(0, 3, 1, 2)  # NHWC -> NCHW
    raise ValueError(f"Cannot infer channel dim for shape {image.shape}")


@dataclass(frozen=True)
class PaliGemmaVisionTowerConfig:
    model_name: str = "google/paligemma2-3b-pt-224"
    image_size: int = 224


class PaliGemmaVisionTower(nn.Module):
    """OpenPI-style vision tower: PaliGemma SigLIP + multi_modal_projector.

    Loads only the vision components of PaliGemma2 (no language model).
    Output shape: (B, num_patches, projection_dim)  e.g. (B, 256, 2048)
    """

    def __init__(self, cfg: PaliGemmaVisionTowerConfig):
        super().__init__()
        self.cfg = cfg
        from transformers import PaliGemmaForConditionalGeneration

        pg = PaliGemmaForConditionalGeneration.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.vision_tower = pg.model.vision_tower
        self.multi_modal_projector = pg.model.multi_modal_projector
        del pg.model.language_model
        del pg

    @property
    def embed_dim(self) -> int:
        return self.multi_modal_projector.linear.out_features

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        image = _to_nchw(image).float()
        # Normalise from [0, 1] to [-1, 1] if needed
        if image.min() >= 0.0:
            image = image * 2.0 - 1.0
        return F.interpolate(
            image,
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        images: dict[str, torch.Tensor],
        image_masks: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not images:
            raise ValueError("No images provided")

        camera_tokens: list[torch.Tensor] = []
        for key, image in images.items():
            pixel_values = self._preprocess(image)
            vision_out = self.vision_tower(pixel_values)
            features = self.multi_modal_projector(vision_out.last_hidden_state)

            if key in image_masks:
                mask = (
                    image_masks[key]
                    .to(features.device, non_blocking=True)
                    .to(features.dtype)[:, None, None]
                )
                features = features * mask
            camera_tokens.append(features)

        return torch.cat(camera_tokens, dim=1)  # (B, N_cameras * N_patches, D)
