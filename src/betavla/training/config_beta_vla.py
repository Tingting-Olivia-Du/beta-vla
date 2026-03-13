from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from betavla.data.libero_loader import LiberoLoaderConfig
from betavla.models.beta_vla_model import BetaVLAConfig
from betavla.models.language_encoder import LanguageEncoderConfig
from betavla.models.vggt_backbone import VGGTBackboneConfig
from betavla.models.vision_tower import PaliGemmaVisionTowerConfig, VisionTowerConfig


@dataclasses.dataclass(frozen=True)
class TrainRuntimeConfig:
    exp_name: str = "beta_vla_run"
    checkpoint_base_dir: str = "./checkpoints"
    wandb_project: str = "beta-vla"
    wandb_enabled: bool = True
    seed: int = 42
    batch_size: int = 32
    grad_accumulation_steps: int = 1
    use_amp: bool = True
    num_train_steps: int = 1000
    log_interval: int = 20
    save_interval: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    num_workers: int = 2
    resume: bool = False
    resume_from_best: bool = False  # if True, load best/ instead of latest step dir
    overwrite: bool = False
    grad_clip_norm: float = 1.0
    # LR schedule (OpenPI-style): warmup + cosine decay
    warmup_steps: int = 1000
    lr_schedule: str = "cosine_warmup"  # "constant" | "cosine_warmup"
    end_lr: float | None = None  # cosine end LR; if None, use peak_lr / 10


@dataclasses.dataclass(frozen=True)
class BetaVLATrainConfig:
    runtime: TrainRuntimeConfig = dataclasses.field(default_factory=TrainRuntimeConfig)
    data: LiberoLoaderConfig = dataclasses.field(default_factory=LiberoLoaderConfig)
    model: BetaVLAConfig = dataclasses.field(default_factory=BetaVLAConfig)

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.runtime.checkpoint_base_dir).resolve() / self.runtime.exp_name


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_float(val: Any, default: float) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        return float(val)
    return default


def load_beta_vla_config(path: str | Path) -> BetaVLATrainConfig:
    data = _read_yaml(path)
    runtime_data = data.get("runtime", {})
    runtime_data["learning_rate"] = _ensure_float(runtime_data.get("learning_rate"), 1e-4)
    runtime_data["weight_decay"] = _ensure_float(runtime_data.get("weight_decay"), 0.0)
    runtime_data["grad_clip_norm"] = _ensure_float(runtime_data.get("grad_clip_norm"), 1.0)
    runtime_data["grad_accumulation_steps"] = int(runtime_data.get("grad_accumulation_steps", 1))
    runtime_data["resume_from_best"] = bool(runtime_data.get("resume_from_best", False))
    runtime_data["use_amp"] = bool(runtime_data.get("use_amp", True))
    runtime_data["warmup_steps"] = int(runtime_data.get("warmup_steps", 1000))
    runtime_data["lr_schedule"] = str(runtime_data.get("lr_schedule", "cosine_warmup"))
    end_lr = runtime_data.get("end_lr")
    runtime_data["end_lr"] = _ensure_float(end_lr, 0.0) if end_lr is not None else None
    runtime = TrainRuntimeConfig(**runtime_data)
    data_cfg = data.get("data", {})
    model_cfg = data.get("model", {})
    vision_cfg = model_cfg.get("vision", {})
    vision_tower_type = model_cfg.get("vision_tower_type", "openvla")
    language_cfg = model_cfg.get("language", {})
    vggt_cfg = model_cfg.get("vggt", {})
    default_vision = VisionTowerConfig()
    default_paligemma = PaliGemmaVisionTowerConfig()
    default_language = LanguageEncoderConfig()
    default_vggt = VGGTBackboneConfig()

    if vision_tower_type == "paligemma":
        vision_config = PaliGemmaVisionTowerConfig(
            model_name=vision_cfg.get("model_name", default_paligemma.model_name),
            image_size=vision_cfg.get("image_size", default_paligemma.image_size),
        )
    else:
        vision_config = VisionTowerConfig(
            dino_model_name=vision_cfg.get("dino_model_name", default_vision.dino_model_name),
            siglip_model_name=vision_cfg.get("siglip_model_name", default_vision.siglip_model_name),
            image_size=vision_cfg.get("image_size", default_vision.image_size),
        )

    model = BetaVLAConfig(
        action_dim=model_cfg.get("action_dim", 32),
        action_horizon=model_cfg.get("action_horizon", 10),
        state_dim=model_cfg.get("state_dim") or data_cfg.get("state_dim", 8),
        gripper_loss_weight=model_cfg.get("gripper_loss_weight", 5.0),
        freeze_vision=model_cfg.get("freeze_vision", False),
        freeze_language=model_cfg.get("freeze_language", False),
        freeze_vggt=model_cfg.get("freeze_vggt", False),
        use_lora=model_cfg.get("use_lora", False),
        lora_r=model_cfg.get("lora_r", 16),
        lora_alpha=model_cfg.get("lora_alpha", 32),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        lora_target_modules=model_cfg.get(
            "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        ),
        lora_target_modules_vggt=model_cfg.get(
            "lora_target_modules_vggt", ["qkv", "proj", "fc1", "fc2"]
        ),
        lora_on_language=model_cfg.get("lora_on_language", True),
        lora_on_vggt=model_cfg.get("lora_on_vggt", True),
        vision_tower_type=vision_tower_type,
        vision=vision_config,
        language=LanguageEncoderConfig(
            model_name=language_cfg.get("model_name", default_language.model_name),
            trust_remote_code=language_cfg.get("trust_remote_code", default_language.trust_remote_code),
        ),
        vggt=VGGTBackboneConfig(
            model_name=vggt_cfg.get("model_name", default_vggt.model_name),
            trust_remote_code=vggt_cfg.get("trust_remote_code", default_vggt.trust_remote_code),
        ),
    )
    default_loader = LiberoLoaderConfig()
    loader_cfg = LiberoLoaderConfig(
        repo_id=data_cfg.get("repo_id", default_loader.repo_id),
        split=data_cfg.get("split", default_loader.split),
        num_workers=data_cfg.get("num_workers", runtime.num_workers),
        max_token_len=data_cfg.get("max_token_len", default_loader.max_token_len),
        max_samples=data_cfg.get("max_samples", default_loader.max_samples),
        state_dim=data_cfg.get("state_dim", default_loader.state_dim),
        norm_stats_path=data_cfg.get("norm_stats_path") or default_loader.norm_stats_path,
    )
    return BetaVLATrainConfig(runtime=runtime, data=loader_cfg, model=model)
