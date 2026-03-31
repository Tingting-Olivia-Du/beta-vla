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
    temporal_frames: int = 1  # 1=single-frame (legacy), >1=multi-frame temporal


def _is_vggt_model(model_name: str) -> bool:
    return "VGGT" in model_name or "facebook/VGGT" in model_name


def _slice_expand_tokens(
    token_param: torch.Tensor, B: int, S: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Expand camera/register tokens for S frames using VGGT convention.

    token_param: (1, 2, X, C)  — index 0 for first frame, index 1 for rest.
    Returns: (B, S, X, C)
    """
    first = token_param[:, 0:1, :, :].expand(B, 1, -1, -1)   # (B, 1, X, C)
    rest = token_param[:, 1:2, :, :].expand(B, S - 1, -1, -1) if S > 1 else first[:, :0]  # (B, S-1, X, C)
    out = torch.cat([first, rest], dim=1)  # (B, S, X, C)
    return out.to(device=device, dtype=dtype)


class _VGGTAggregatorBackbone(nn.Module):
    """VGGT aggregator with support for single-frame and multi-frame modes.

    Single-frame (temporal_frames=1): backward-compatible with original behavior.
    Multi-frame (temporal_frames>1): S = temporal_frames * num_cameras views,
      with proper frame/global alternating attention and per-view camera/register tokens.
    """

    def __init__(
        self,
        model_name: str,
        vision_patch_hw: tuple[int, int] = (16, 16),
        num_cameras: int = 2,
        temporal_frames: int = 1,
    ):
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
        self.patch_start_idx = agg.patch_start_idx  # 5 (1 camera + 4 register)
        # Keep full camera/register tokens: (1, 2, X, C) — needed for multi-frame
        self.camera_token = agg.camera_token     # (1, 2, 1, C)
        self.register_token = agg.register_token  # (1, 2, 4, C)
        self.embed_dim = agg.frame_blocks[0].norm1.normalized_shape[0]
        self.rope = agg.rope
        self.position_getter = agg.position_getter
        self.patch_size = agg.patch_size
        self.vision_patch_hw = vision_patch_hw
        self.num_cameras = num_cameras
        self.temporal_frames = temporal_frames

    @property
    def hidden_size(self) -> int:
        return self.embed_dim

    # ------------------------------------------------------------------
    # Single-frame forward (backward-compatible, temporal_frames=1)
    # ------------------------------------------------------------------

    def _forward_single_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Original single-frame path. Input: (B, N_vis+N_lang, C)."""
        B, T, C = x.shape
        S = 1
        # Prepend camera + register using only index 0
        cam = self.camera_token[:, 0:1, :, :].expand(B, 1, 1, C).to(device=x.device, dtype=x.dtype)
        reg = self.register_token[:, 0:1, :, :].expand(B, 1, 4, C).to(device=x.device, dtype=x.dtype)
        special = torch.cat([cam, reg], dim=2).view(B, 5, C)
        tokens = torch.cat([special, x], dim=1)  # (B, 5+T, C)
        P = tokens.shape[1]

        # 2D positional encoding for RoPE
        pos = None
        if self.rope is not None and self.position_getter is not None:
            ph, pw = self.vision_patch_hw
            n_cam_patches = ph * pw
            n_vision = self.num_cameras * n_cam_patches
            n_special = self.patch_start_idx  # 5
            n_other = P - n_special - n_vision

            pos_parts: list[torch.Tensor] = []
            pos_parts.append(torch.zeros(B, n_special, 2, device=tokens.device, dtype=torch.long))
            for cam_i in range(self.num_cameras):
                cam_pos = self.position_getter(B, ph, pw, device=tokens.device)
                cam_pos = cam_pos.clone()
                cam_pos[..., 0] += cam_i * ph
                pos_parts.append(cam_pos)
            if n_other > 0:
                lang_pos = self.position_getter(B, 1, n_other, device=tokens.device)
                lang_pos = lang_pos.clone()
                lang_pos[..., 0] += self.num_cameras * ph
                pos_parts.append(lang_pos)

            pos = torch.cat(pos_parts, dim=1)

        frame_idx = global_idx = 0
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens = tokens.view(B * S, P, C)
                    pos_f = pos.view(B * S, P, 2) if pos is not None else None
                    for _ in range(self.aa_block_size):
                        tokens = torch.utils.checkpoint.checkpoint(
                            self.frame_blocks[frame_idx], tokens, pos_f, use_reentrant=False
                        )
                        frame_idx += 1
                    tokens = tokens.view(B, P, C)
                else:
                    tokens = tokens.view(B, S * P, C)
                    pos_g = pos.view(B, S * P, 2) if pos is not None else None
                    for _ in range(self.aa_block_size):
                        tokens = torch.utils.checkpoint.checkpoint(
                            self.global_blocks[global_idx], tokens, pos_g, use_reentrant=False
                        )
                        global_idx += 1
                    tokens = tokens.view(B, P, C)

        return tokens[:, 5:, :]  # drop special tokens

    # ------------------------------------------------------------------
    # Multi-frame forward (temporal_frames > 1)
    # ------------------------------------------------------------------

    def _forward_multi_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-frame path with proper per-view frame/global attention.

        Input x: (B, S*P_patches + N_lang, C)
          where S = temporal_frames * num_cameras,
                P_patches = ph * pw (256 for 16x16)
        Vision tokens are ordered: [t0_cam0, t0_cam1, t1_cam0, t1_cam1, ...]
        followed by language tokens.

        Output: (B, num_cameras * P_patches + N_lang, C)
          — only the current (latest) timestep's vision + language tokens.
        """
        B = x.shape[0]
        C = x.shape[2]
        ph, pw = self.vision_patch_hw
        P_patches = ph * pw  # 256
        S = self.temporal_frames * self.num_cameras  # total views
        N_vis = S * P_patches
        N_lang = x.shape[1] - N_vis

        # 1. Split vision and language tokens
        vis_tokens = x[:, :N_vis, :]   # (B, S * P_patches, C)
        lang_tokens = x[:, N_vis:, :]  # (B, N_lang, C)

        # 2. Reshape vision to per-view: (B, S, P_patches, C)
        vis_tokens = vis_tokens.view(B, S, P_patches, C)

        # 3. Prepend camera + register tokens per view
        # camera_token: (1, 2, 1, C), register_token: (1, 2, 4, C)
        cam = _slice_expand_tokens(self.camera_token, B, S, x.device, x.dtype)   # (B, S, 1, C)
        reg = _slice_expand_tokens(self.register_token, B, S, x.device, x.dtype)  # (B, S, 4, C)

        # per_frame: (B, S, P_per_view, C) where P_per_view = 5 + P_patches = 261
        per_frame = torch.cat([cam, reg, vis_tokens], dim=2)
        P = per_frame.shape[2]  # tokens per view (261)
        n_special = self.patch_start_idx  # 5

        # 4. Build RoPE positions
        #    Frame attention pos: per-view, (B*S, P, 2)
        #    Global attention pos: all views + language, (B, S*P + N_lang, 2)
        pos_frame = None
        pos_global = None
        if self.rope is not None and self.position_getter is not None:
            # Per-view positions (same grid for each view in frame attention)
            special_pos = torch.zeros(B * S, n_special, 2, device=x.device, dtype=torch.long)
            patch_pos = self.position_getter(B * S, ph, pw, device=x.device)  # (B*S, P_patches, 2)
            # Offset patch positions by +1 so special tokens at (0,0) don't collide
            patch_pos = patch_pos + 1
            pos_frame = torch.cat([special_pos, patch_pos], dim=1)  # (B*S, P, 2)

            # Global positions: each view gets a unique y-offset
            global_parts: list[torch.Tensor] = []
            for view_i in range(S):
                view_special = torch.zeros(B, n_special, 2, device=x.device, dtype=torch.long)
                view_patches = self.position_getter(B, ph, pw, device=x.device).clone()
                view_patches = view_patches + 1  # match frame attention offset
                # y-offset to separate views
                y_offset = view_i * (ph + 1)
                view_special[..., 0] += y_offset
                view_patches[..., 0] += y_offset
                global_parts.append(torch.cat([view_special, view_patches], dim=1))  # (B, P, 2)

            # Language tokens below all views
            if N_lang > 0:
                lang_pos = self.position_getter(B, 1, N_lang, device=x.device).clone()
                lang_pos[..., 0] += S * (ph + 1)
                global_parts.append(lang_pos)

            pos_global = torch.cat(global_parts, dim=1)  # (B, S*P + N_lang, 2)

        # 5. Alternating attention loop
        frame_idx = global_idx = 0
        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    # Frame attention: within each view independently
                    tokens_flat = per_frame.reshape(B * S, P, C)
                    for _ in range(self.aa_block_size):
                        tokens_flat = torch.utils.checkpoint.checkpoint(
                            self.frame_blocks[frame_idx], tokens_flat, pos_frame,
                            use_reentrant=False,
                        )
                        frame_idx += 1
                    per_frame = tokens_flat.view(B, S, P, C)
                else:
                    # Global attention: across all views + language
                    vis_flat = per_frame.reshape(B, S * P, C)
                    global_tokens = torch.cat([vis_flat, lang_tokens], dim=1)  # (B, S*P+N_lang, C)
                    for _ in range(self.aa_block_size):
                        global_tokens = torch.utils.checkpoint.checkpoint(
                            self.global_blocks[global_idx], global_tokens, pos_global,
                            use_reentrant=False,
                        )
                        global_idx += 1
                    # Split back
                    per_frame = global_tokens[:, :S * P, :].view(B, S, P, C)
                    lang_tokens = global_tokens[:, S * P:, :]

        # 6. Output: only current timestep (last T) vision patches + language
        # Views are ordered [t0_cam0, t0_cam1, ..., t(T-1)_cam0, t(T-1)_cam1]
        # Current timestep starts at view index (temporal_frames - 1) * num_cameras
        cur_start = (self.temporal_frames - 1) * self.num_cameras
        cur_views = per_frame[:, cur_start:, :, :]  # (B, num_cameras, P, C)
        # Drop special tokens, keep only patches
        cur_patches = cur_views[:, :, n_special:, :]  # (B, num_cameras, P_patches, C)
        cur_patches = cur_patches.reshape(B, self.num_cameras * P_patches, C)

        return torch.cat([cur_patches, lang_tokens], dim=1)  # (B, 512+N_lang, C)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        x = tokens if tokens is not None else inputs_embeds if inputs_embeds is not None else input_ids
        if x is None:
            raise ValueError("Must provide tokens, input_ids, or inputs_embeds")

        if self.temporal_frames <= 1:
            return self._forward_single_frame(x)
        return self._forward_multi_frame(x)


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
                temporal_frames=cfg.temporal_frames,
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
