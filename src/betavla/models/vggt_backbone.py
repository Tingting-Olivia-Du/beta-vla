from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel


@dataclass(frozen=True)
class VGGTBackboneConfig:
    model_name: str = "facebook/VGGT-1B"
    trust_remote_code: bool = True
    # Vision layout for proper 2D positional encoding.
    # PaliGemma 224px / patch_size 14 → 16×16 = 256 patches per camera.
    vision_patch_hw: tuple[int, int] = (16, 16)
    num_cameras: int = 2


def _is_vggt_model(model_name: str) -> bool:
    return "VGGT" in model_name or "facebook/VGGT" in model_name


class _VGGTAggregatorBackbone(nn.Module):
    """Use VGGT aggregator's transformer blocks with inputs_embeds.
    Loads from vggt package via VGGT.from_pretrained (HuggingFace).
    """

    def __init__(self, model_name: str, vision_patch_hw: tuple[int, int] = (16, 16), num_cameras: int = 2):
        super().__init__()
        from vggt.models.vggt import VGGT

        vggt_full = VGGT.from_pretrained(model_name)
        agg = vggt_full.aggregator
        self.frame_blocks = agg.frame_blocks
        self.global_blocks = agg.global_blocks
        self.depth = agg.depth
        self.aa_order = agg.aa_order
        self.aa_block_size = agg.aa_block_size
        self.aa_block_num = agg.aa_block_num
        self.patch_start_idx = agg.patch_start_idx
        # Use camera + register tokens from aggregator (for S=1)
        self.camera_token = agg.camera_token[:, 0:1, :, :]  # (1, 1, 1, C)
        self.register_token = agg.register_token[:, 0:1, :, :]  # (1, 1, 4, C)
        self.embed_dim = agg.frame_blocks[0].norm1.normalized_shape[0]
        self.rope = agg.rope
        self.position_getter = agg.position_getter
        self.patch_size = agg.patch_size
        # Vision layout for 2D positional encoding
        self.vision_patch_hw = vision_patch_hw
        self.num_cameras = num_cameras

    @property
    def hidden_size(self) -> int:
        return self.embed_dim

    def forward(
        self,
        tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # PEFT passes first positional arg as input_ids; we use embeddings, so accept both
        x = tokens if tokens is not None else inputs_embeds if inputs_embeds is not None else input_ids
        if x is None:
            raise ValueError("Must provide tokens, input_ids, or inputs_embeds")
        B, T, C = x.shape
        S = 1  # single sequence
        # Prepend camera + register: (1,1,1,C) + (1,1,4,C) -> expand to (B, 1, 5, C)
        # Ensure same device/dtype as input (camera/register may stay on CPU after PEFT wrap)
        cam = self.camera_token.expand(B, 1, 1, C).to(device=x.device, dtype=x.dtype)
        reg = self.register_token.expand(B, 1, 4, C).to(device=x.device, dtype=x.dtype)
        special = torch.cat([cam, reg], dim=2).view(B, 5, C)
        tokens = torch.cat([special, x], dim=1)  # (B, 5+T, C)
        P = tokens.shape[1]

        # 2D positional encoding for RoPE.
        # Build proper 2D grid positions for vision patches (instead of 1D).
        # Token layout after special prepend: [special(5), cam1(H*W), cam2(H*W), ..., language(L)]
        pos = None
        if self.rope is not None and self.position_getter is not None:
            ph, pw = self.vision_patch_hw
            n_cam_patches = ph * pw
            n_vision = self.num_cameras * n_cam_patches
            n_special = self.patch_start_idx  # 5
            n_other = P - n_special - n_vision  # language tokens

            pos_parts: list[torch.Tensor] = []

            # Special tokens (camera + register): position (0, 0)
            pos_parts.append(torch.zeros(B, n_special, 2, device=tokens.device, dtype=torch.long))

            # Per-camera 2D grid positions with y-offset between cameras
            for cam_i in range(self.num_cameras):
                cam_pos = self.position_getter(B, ph, pw, device=tokens.device)  # (B, H*W, 2)
                cam_pos = cam_pos.clone()
                cam_pos[..., 0] += cam_i * ph  # y-offset to separate cameras
                pos_parts.append(cam_pos)

            # Language tokens: 1D row below all camera grids
            if n_other > 0:
                lang_pos = self.position_getter(B, 1, n_other, device=tokens.device)  # (B, L, 2)
                lang_pos = lang_pos.clone()
                lang_pos[..., 0] += self.num_cameras * ph  # y-offset below cameras
                pos_parts.append(lang_pos)

            pos = torch.cat(pos_parts, dim=1)  # (B, P, 2)
            assert pos.shape[1] == P, f"pos shape {pos.shape[1]} != P {P}"

        frame_idx = global_idx = 0
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens = tokens.view(B * S, P, C)
                    if pos is not None:
                        pos_f = pos.view(B * S, P, 2)
                    else:
                        pos_f = None
                    for _ in range(self.aa_block_size):
                        tokens = torch.utils.checkpoint.checkpoint(
                            self.frame_blocks[frame_idx], tokens, pos_f, use_reentrant=False
                        )
                        frame_idx += 1
                    tokens = tokens.view(B, P, C)
                else:
                    tokens = tokens.view(B, S * P, C)
                    if pos is not None:
                        pos_g = pos.view(B, S * P, 2)
                    else:
                        pos_g = None
                    for _ in range(self.aa_block_size):
                        tokens = torch.utils.checkpoint.checkpoint(
                            self.global_blocks[global_idx], tokens, pos_g, use_reentrant=False
                        )
                        global_idx += 1
                    tokens = tokens.view(B, P, C)

        return tokens[:, 5:, :]  # drop special tokens, return (B, T, C)


class VGGTBackbone(nn.Module):
    """Backbone wrapper that consumes fused token embeddings.
    For facebook/VGGT-1B: uses vggt package (VGGT.from_pretrained from HuggingFace).
    For other models: uses transformers AutoModel.
    """

    def __init__(self, cfg: VGGTBackboneConfig):
        super().__init__()
        self.cfg = cfg
        if _is_vggt_model(cfg.model_name):
            self.model = _VGGTAggregatorBackbone(
                cfg.model_name,
                vision_patch_hw=cfg.vision_patch_hw,
                num_cameras=cfg.num_cameras,
            )
            self.hidden_size = self.model.hidden_size
        else:
            self.model = AutoModel.from_pretrained(
                cfg.model_name, trust_remote_code=cfg.trust_remote_code
            )
            self.hidden_size = int(self.model.config.hidden_size)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if _is_vggt_model(self.cfg.model_name):
            return self.model(tokens, attention_mask)
        outputs = self.model(inputs_embeds=tokens, attention_mask=attention_mask, use_cache=False)
        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("Backbone did not return last_hidden_state")
        return outputs.last_hidden_state
