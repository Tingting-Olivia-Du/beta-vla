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
        """Encode images from all cameras.

        Supports both single-frame and temporal inputs:
          - Single-frame: images[key] shape (B, H, W, C)  → output (B, N_cam * 256, D)
          - Temporal:     images[key] shape (B, T, H, W, C) → output (B, T * N_cam * 256, D)
        """
        if not images:
            raise ValueError("No images provided")

        camera_tokens: list[torch.Tensor] = []
        for key, image in images.items():
            temporal = image.ndim == 5  # (B, T, H, W, C)
            if temporal:
                B, T = image.shape[0], image.shape[1]
                # Flatten batch and time: (B*T, H, W, C)
                image_flat = image.reshape(B * T, *image.shape[2:])
            else:
                B = image.shape[0]
                T = 1
                image_flat = image

            pixel_values = self._preprocess(image_flat)
            vision_out = self.vision_tower(pixel_values)
            features = self.multi_modal_projector(vision_out.last_hidden_state)
            # features: (B*T, N_patches, D)

            if key in image_masks:
                mask = (
                    image_masks[key]
                    .to(features.device, non_blocking=True)
                    .to(features.dtype)[:, None, None]
                )
                if temporal:
                    # mask shape (B,) → expand to (B*T, 1, 1)
                    mask = mask.squeeze(-1).squeeze(-1)  # (B,)
                    mask = mask.unsqueeze(1).expand(B, T).reshape(B * T)[:, None, None]
                features = features * mask

            if temporal:
                # Reshape back: (B, T * N_patches, D)
                N_patches = features.shape[1]
                features = features.view(B, T * N_patches, features.shape[2])

            camera_tokens.append(features)

        # Interleave: for temporal, we want [t0_cam0, t0_cam1, t1_cam0, t1_cam1, ...]
        # Current order from dict iteration: all T patches of cam0, then all T patches of cam1
        # We need to reorder to interleave by timestep
        first_img = next(iter(images.values()))
        is_temporal = first_img.ndim == 5
        if is_temporal and len(camera_tokens) > 1:
            B = first_img.shape[0]
            T = first_img.shape[1]
            N_patches = camera_tokens[0].shape[1] // T  # patches per frame per camera
            N_cam = len(camera_tokens)
            # Each camera_tokens[c]: (B, T * N_patches, D)
            # Reshape to (B, T, N_patches, D), stack cameras, then interleave
            per_cam = [ct.view(B, T, N_patches, -1) for ct in camera_tokens]
            # Stack: (B, N_cam, T, N_patches, D)
            stacked = torch.stack(per_cam, dim=1)
            # Transpose to (B, T, N_cam, N_patches, D) then flatten
            stacked = stacked.permute(0, 2, 1, 3, 4)  # (B, T, N_cam, N_patches, D)
            result = stacked.reshape(B, T * N_cam * N_patches, -1)
            return result

        return torch.cat(camera_tokens, dim=1)  # (B, N_cameras * N_patches, D)
