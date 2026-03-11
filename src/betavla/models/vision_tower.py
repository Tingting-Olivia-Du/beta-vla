from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import timm
import torch
from torch import nn
import torch.nn.functional as F


def _to_nchw(image: torch.Tensor) -> torch.Tensor:
    """Ensure image is (B, C, H, W)."""
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor with 4 dims, got {image.shape}")
    if image.shape[1] in (1, 3):
        return image
    if image.shape[-1] in (1, 3):
        return image.permute(0, 3, 1, 2)
    raise ValueError(f"Cannot infer channel dim for shape {image.shape}")


def _unpack_tuple(fn):
    """OpenVLA-style: get_intermediate_layers returns list/tuple, unpack to single tensor."""

    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, (list, tuple)) and len(result) >= 1:
            return result[0]
        return result

    return wrapper


@dataclass(frozen=True)
class VisionTowerConfig:
    dino_model_name: str = "vit_large_patch14_reg4_dinov2.lvd142m"
    siglip_model_name: str = "vit_so400m_patch14_siglip_224"
    image_size: int = 224


class OpenVLAVisionTower(nn.Module):
    """OpenVLA-style dual encoder vision tower (DINOv2 + SigLIP)."""

    def __init__(self, cfg: VisionTowerConfig):
        super().__init__()
        self.cfg = cfg
        self.dino = timm.create_model(cfg.dino_model_name, pretrained=True, num_classes=0, img_size=cfg.image_size)
        self.siglip = timm.create_model(cfg.siglip_model_name, pretrained=True, num_classes=0, img_size=cfg.image_size)
        # OpenVLA-style monkey-patch: forward -> get_intermediate_layers(penultimate) returning single tensor
        self.dino.forward = _unpack_tuple(
            partial(self.dino.get_intermediate_layers, n=[len(self.dino.blocks) - 2])
        )
        self.siglip.forward = _unpack_tuple(
            partial(self.siglip.get_intermediate_layers, n=[len(self.siglip.blocks) - 2])
        )

    @property
    def embed_dim(self) -> int:
        return int(self.dino.embed_dim + self.siglip.embed_dim)

    def _to_nchw(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor with 4 dims, got {image.shape}")
        if image.shape[1] in (1, 3):
            return image
        if image.shape[-1] in (1, 3):
            return image.permute(0, 3, 1, 2)
        raise ValueError(f"Cannot infer channel dim for shape {image.shape}")

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        image = self._to_nchw(image).to(torch.float32)
        if image.min() < 0:
            image = (image + 1.0) / 2.0
        return F.interpolate(image, size=(self.cfg.image_size, self.cfg.image_size), mode="bilinear", align_corners=False)

    def forward(self, images: dict[str, torch.Tensor], image_masks: dict[str, torch.Tensor]) -> torch.Tensor:
        if not images:
            raise ValueError("No images were provided in observation.images")

        camera_tokens: list[torch.Tensor] = []
        for key, image in images.items():
            image = self._preprocess(image)
            dino_tokens = self.dino(image)
            siglip_tokens = self.siglip(image)
            features = torch.cat([dino_tokens, siglip_tokens], dim=-1)

            if key in image_masks:
                mask = image_masks[key].to(features.device, non_blocking=True).to(features.dtype)[:, None, None]
                features = features * mask
            camera_tokens.append(features)

        return torch.cat(camera_tokens, dim=1)


@dataclass(frozen=True)
class PaliGemmaVisionTowerConfig:
    """Config for OpenPI-style PaliGemma vision encoder (SigLIP + projector)."""

    model_name: str = "google/paligemma2-3b-pt-224"
    image_size: int = 224


class PaliGemmaVisionTower(nn.Module):
    """OpenPI-style vision tower: PaliGemma's SigLIP + multi_modal_projector.

    Uses the same vision encoder as OpenPI/PaliGemma for image-text grounding.
    Output dim = projection_dim (2048), matches VGGT hidden_size.
    """

    def __init__(self, cfg: PaliGemmaVisionTowerConfig):
        super().__init__()
        self.cfg = cfg
        from transformers import PaliGemmaForConditionalGeneration

        paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.vision_tower = paligemma.model.vision_tower
        self.multi_modal_projector = paligemma.model.multi_modal_projector
        # Free references to language model to save memory
        del paligemma.model.language_model
        del paligemma

    @property
    def embed_dim(self) -> int:
        return self.multi_modal_projector.linear.out_features

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        image = _to_nchw(image).to(torch.float32)
        if image.min() < 0:
            image = (image + 1.0) / 2.0
        return F.interpolate(
            image,
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, images: dict[str, torch.Tensor], image_masks: dict[str, torch.Tensor]) -> torch.Tensor:
        if not images:
            raise ValueError("No images were provided in observation.images")

        camera_tokens: list[torch.Tensor] = []
        for key, image in images.items():
            pixel_values = self._preprocess(image)
            image_outputs = self.vision_tower(pixel_values)
            features = self.multi_modal_projector(image_outputs.last_hidden_state)

            if key in image_masks:
                mask = image_masks[key].to(features.device, non_blocking=True).to(features.dtype)[:, None, None]
                features = features * mask
            camera_tokens.append(features)

        return torch.cat(camera_tokens, dim=1)
