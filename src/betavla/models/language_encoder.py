from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel


@dataclass(frozen=True)
class LanguageEncoderConfig:
    model_name: str = "Qwen/Qwen3-0.6B-Base"
    trust_remote_code: bool = True


class QwenLanguageEncoder(nn.Module):
    """Thin wrapper around Qwen encoder used for token embeddings."""

    def __init__(self, cfg: LanguageEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.model = AutoModel.from_pretrained(cfg.model_name, trust_remote_code=cfg.trust_remote_code)
        self.hidden_size = int(self.model.config.hidden_size)

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        if token_ids is None:
            raise ValueError("observation.tokenized_prompt is required for QwenLanguageEncoder")

        outputs = self.model(input_ids=token_ids, attention_mask=token_mask, use_cache=False)
        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("Qwen encoder did not return last_hidden_state")
        return outputs.last_hidden_state
