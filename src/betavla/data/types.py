from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ObservationBatch:
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor

    def to(self, device: torch.device) -> "ObservationBatch":
        return ObservationBatch(
            images={k: v.to(device) for k, v in self.images.items()},
            image_masks={k: v.to(device) for k, v in self.image_masks.items()},
            state=self.state.to(device),
            tokenized_prompt=self.tokenized_prompt.to(device),
            tokenized_prompt_mask=self.tokenized_prompt_mask.to(device),
        )
