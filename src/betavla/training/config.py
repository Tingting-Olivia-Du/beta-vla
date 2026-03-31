"""Training configuration for Beta-VLA."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from betavla.data.libero_dataset import LiberoDatasetConfig
from betavla.models.language_encoder import LanguageEncoderConfig
from betavla.models.model import BetaVLAConfig
from betavla.models.vggt_backbone import VGGTBackboneConfig
from betavla.models.vision_tower import PaliGemmaVisionTowerConfig


@dataclasses.dataclass(frozen=True)
class TrainRuntimeConfig:
    exp_name: str = "libero_vggt"
    checkpoint_base_dir: str = "./checkpoints"
    wandb_project: str = "beta-vla"
    wandb_enabled: bool = True
    seed: int = 42
    batch_size: int = 16          # per GPU
    grad_accumulation_steps: int = 1
    use_amp: bool = True
    num_train_steps: int = 30000
    log_interval: int = 20
    save_interval: int = 1000
    learning_rate: float = 2.5e-5
    weight_decay: float = 1e-10   # negligible, matches openpi
    num_workers: int = 4
    resume: bool = False
    resume_from_best: bool = False
    overwrite: bool = False
    grad_clip_norm: float = 1.0
    warmup_steps: int = 1000
    lr_schedule: str = "cosine_warmup"  # "constant" | "cosine_warmup"
    end_lr: float | None = None


@dataclasses.dataclass(frozen=True)
class BetaVLATrainConfig:
    runtime: TrainRuntimeConfig = dataclasses.field(default_factory=TrainRuntimeConfig)
    data: LiberoDatasetConfig = dataclasses.field(default_factory=LiberoDatasetConfig)
    model: BetaVLAConfig = dataclasses.field(default_factory=BetaVLAConfig)

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.runtime.checkpoint_base_dir).resolve() / self.runtime.exp_name


def _as_float(val: Any, default: float) -> float:
    if val is None:
        return default
    return float(val)


def _as_int(val: Any, default: int) -> int:
    if val is None:
        return default
    return int(val)


def load_config(path: str | Path) -> BetaVLATrainConfig:
    """Load YAML config and construct BetaVLATrainConfig."""
    with Path(path).expanduser().resolve().open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    rt = raw.get("runtime", {})
    runtime = TrainRuntimeConfig(
        exp_name=str(rt.get("exp_name", "libero_vggt")),
        checkpoint_base_dir=str(rt.get("checkpoint_base_dir", "./checkpoints")),
        wandb_project=str(rt.get("wandb_project", "beta-vla")),
        wandb_enabled=bool(rt.get("wandb_enabled", True)),
        seed=_as_int(rt.get("seed"), 42),
        batch_size=_as_int(rt.get("batch_size"), 16),
        grad_accumulation_steps=_as_int(rt.get("grad_accumulation_steps"), 1),
        use_amp=bool(rt.get("use_amp", True)),
        num_train_steps=_as_int(rt.get("num_train_steps"), 30000),
        log_interval=_as_int(rt.get("log_interval"), 20),
        save_interval=_as_int(rt.get("save_interval"), 1000),
        learning_rate=_as_float(rt.get("learning_rate"), 2.5e-5),
        weight_decay=_as_float(rt.get("weight_decay"), 1e-10),
        num_workers=_as_int(rt.get("num_workers"), 4),
        resume=bool(rt.get("resume", False)),
        resume_from_best=bool(rt.get("resume_from_best", False)),
        overwrite=bool(rt.get("overwrite", False)),
        grad_clip_norm=_as_float(rt.get("grad_clip_norm"), 1.0),
        warmup_steps=_as_int(rt.get("warmup_steps"), 1000),
        lr_schedule=str(rt.get("lr_schedule", "cosine_warmup")),
        end_lr=_as_float(rt.get("end_lr"), 0.0) if rt.get("end_lr") is not None else None,
    )

    d = raw.get("data", {})
    data = LiberoDatasetConfig(
        repo_id=str(d.get("repo_id", "physical-intelligence/libero")),
        split=str(d.get("split", "train")),
        num_workers=_as_int(d.get("num_workers"), runtime.num_workers),
        max_token_len=_as_int(d.get("max_token_len"), 128),
        max_samples=_as_int(d.get("max_samples"), 0) or None,
        state_dim=_as_int(d.get("state_dim"), 8),
        norm_stats_path=d.get("norm_stats_path") or None,
        temporal_frames=_as_int(d.get("temporal_frames"), 1),
        temporal_stride=_as_int(d.get("temporal_stride"), 5),
    )

    m = raw.get("model", {})
    v = m.get("vision", {})
    lang = m.get("language", {})
    vggt = m.get("vggt", {})

    default_vision = PaliGemmaVisionTowerConfig()
    default_lang = LanguageEncoderConfig()
    default_vggt = VGGTBackboneConfig()

    vision_cfg = PaliGemmaVisionTowerConfig(
        model_name=str(v.get("model_name", default_vision.model_name)),
        image_size=_as_int(v.get("image_size"), default_vision.image_size),
    )
    lang_cfg = LanguageEncoderConfig(
        model_name=str(lang.get("model_name", default_lang.model_name)),
        trust_remote_code=bool(lang.get("trust_remote_code", default_lang.trust_remote_code)),
    )
    vggt_cfg = VGGTBackboneConfig(
        model_name=str(vggt.get("model_name", default_vggt.model_name)),
        trust_remote_code=bool(vggt.get("trust_remote_code", default_vggt.trust_remote_code)),
        temporal_frames=_as_int(vggt.get("temporal_frames"), data.temporal_frames),
    )

    model = BetaVLAConfig(
        action_dim=_as_int(m.get("action_dim"), 7),
        action_horizon=_as_int(m.get("action_horizon"), 10),
        state_dim=_as_int(m.get("state_dim"), data.state_dim),
        gripper_loss_weight=_as_float(m.get("gripper_loss_weight"), 5.0),
        freeze_vision=bool(m.get("freeze_vision", True)),
        freeze_language=bool(m.get("freeze_language", False)),
        freeze_vggt=bool(m.get("freeze_vggt", False)),
        use_lora=bool(m.get("use_lora", True)),
        lora_r=_as_int(m.get("lora_r"), 16),
        lora_alpha=_as_int(m.get("lora_alpha"), 32),
        lora_dropout=_as_float(m.get("lora_dropout"), 0.05),
        lora_target_modules=list(m.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"])),
        lora_target_modules_vggt=list(m.get("lora_target_modules_vggt", ["qkv", "proj", "fc1", "fc2"])),
        lora_on_language=bool(m.get("lora_on_language", True)),
        lora_on_vggt=bool(m.get("lora_on_vggt", True)),
        lora_on_vision=bool(m.get("lora_on_vision", False)),
        lora_target_modules_vision=list(m.get("lora_target_modules_vision", ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])),
        vision=vision_cfg,
        language=lang_cfg,
        vggt=vggt_cfg,
    )

    return BetaVLATrainConfig(runtime=runtime, data=data, model=model)
