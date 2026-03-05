from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from betavla.data.libero_loader import LiberoLoaderConfig
from betavla.models.beta_vla_model import BetaVLAConfig
from betavla.models.language_encoder import LanguageEncoderConfig
from betavla.models.vggt_backbone import VGGTBackboneConfig
from betavla.models.vision_tower import VisionTowerConfig


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
    overwrite: bool = False
    grad_clip_norm: float = 1.0


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
    runtime_data["use_amp"] = bool(runtime_data.get("use_amp", True))
    runtime = TrainRuntimeConfig(**runtime_data)
    data_cfg = data.get("data", {})
    model_cfg = data.get("model", {})
    vision_cfg = model_cfg.get("vision", {})
    language_cfg = model_cfg.get("language", {})
    vggt_cfg = model_cfg.get("vggt", {})
    default_vision = VisionTowerConfig()
    default_language = LanguageEncoderConfig()
    default_vggt = VGGTBackboneConfig()
    model = BetaVLAConfig(
        action_dim=model_cfg.get("action_dim", 32),
        action_horizon=model_cfg.get("action_horizon", 10),
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
        vision=VisionTowerConfig(
            dino_model_name=vision_cfg.get("dino_model_name", default_vision.dino_model_name),
            siglip_model_name=vision_cfg.get("siglip_model_name", default_vision.siglip_model_name),
            image_size=vision_cfg.get("image_size", default_vision.image_size),
        ),
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
    )
    return BetaVLATrainConfig(runtime=runtime, data=loader_cfg, model=model)
