from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel


@dataclass(frozen=True)
class VGGTBackboneConfig:
    model_name: str = "facebook/VGGT-1B"
    trust_remote_code: bool = True


def _is_vggt_model(model_name: str) -> bool:
    return "VGGT" in model_name or "facebook/VGGT" in model_name


class _VGGTAggregatorBackbone(nn.Module):
    """Use VGGT aggregator's transformer blocks with inputs_embeds.
    Loads from vggt package via VGGT.from_pretrained (HuggingFace).
    """

    def __init__(self, model_name: str):
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

        # 2D pos for RoPE: use 1xP grid for 1D sequence (h=1, w=P)
        pos = None
        if self.rope is not None and self.position_getter is not None:
            pos_full = self.position_getter(B * S, 1, P, device=tokens.device)
            if pos_full.shape[1] < P:
                pos_full = torch.nn.functional.pad(pos_full, (0, 0, 0, P - pos_full.shape[1]), value=0)
            # Special tokens (first 5) get pos=0; rest use 2D grid
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=tokens.device, dtype=pos_full.dtype)
            pos = torch.cat([pos_special, pos_full[:, self.patch_start_idx : self.patch_start_idx + P - 5]], dim=1)
            if pos.shape[1] != P:
                pos = torch.nn.functional.pad(pos, (0, 0, 0, max(0, P - pos.shape[1])), value=0)

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
            self.model = _VGGTAggregatorBackbone(cfg.model_name)
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
