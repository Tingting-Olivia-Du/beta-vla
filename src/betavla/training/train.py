"""Beta-VLA training script.

Usage (single GPU):
    python src/betavla/training/train.py --config configs/libero_vggt.yaml

Usage (multi-GPU via torchrun):
    torchrun --standalone --nproc_per_node=4 src/betavla/training/train.py --config configs/libero_vggt.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import wandb

from betavla.data.libero_dataset import LiberoDataset, create_dataloader
from betavla.models.model import BetaVLAModel
from betavla.training.config import BetaVLATrainConfig, load_config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Distributed
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[bool, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------

def maybe_init_wandb(cfg: BetaVLATrainConfig, is_main: bool) -> None:
    if not is_main or not cfg.runtime.wandb_enabled:
        wandb.init(mode="disabled")
        return
    wandb.init(
        project=cfg.runtime.wandb_project,
        name=cfg.runtime.exp_name,
        config=dataclasses.asdict(cfg),
    )


# ---------------------------------------------------------------------------
# LR schedule (OpenPI-style cosine with warmup)
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: BetaVLATrainConfig) -> float:
    peak = cfg.runtime.learning_rate
    if cfg.runtime.lr_schedule != "cosine_warmup":
        return peak
    warmup = cfg.runtime.warmup_steps
    total = cfg.runtime.num_train_steps
    end = cfg.runtime.end_lr if cfg.runtime.end_lr is not None else peak / 10.0
    if step < warmup:
        return peak * (step + 1) / (warmup + 1)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return end + (peak - end) * cos


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _checkpoint_dir(cfg: BetaVLATrainConfig) -> Path:
    return cfg.checkpoint_dir


def _copy_norm_stats(cfg: BetaVLATrainConfig, dest: Path) -> None:
    if cfg.data.norm_stats_path is None:
        return
    src = Path(cfg.data.norm_stats_path)
    if src.exists():
        shutil.copy2(src, dest / "norm_stats.json")


def save_checkpoint(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    step: int,
    cfg: BetaVLATrainConfig,
) -> None:
    ckpt_dir = _checkpoint_dir(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step_dir = ckpt_dir / str(step)
    tmp = ckpt_dir / f"tmp_{step}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    m = model.module if isinstance(model, DDP) else model
    safetensors.torch.save_model(m, tmp / "model.safetensors")
    torch.save(optim.state_dict(), tmp / "optimizer.pt")
    torch.save({"step": step, "timestamp": time.time()}, tmp / "metadata.pt")
    _copy_norm_stats(cfg, tmp)
    if step_dir.exists():
        shutil.rmtree(step_dir)
    tmp.rename(step_dir)


def _best_info_path(cfg: BetaVLATrainConfig) -> Path:
    return _checkpoint_dir(cfg) / "best.json"


def _load_best_loss(cfg: BetaVLATrainConfig) -> tuple[float, int]:
    p = _best_info_path(cfg)
    if not p.exists():
        return float("inf"), -1
    info = json.loads(p.read_text(encoding="utf-8"))
    return float(info.get("best_loss", float("inf"))), int(info.get("best_step", -1))


def save_best_checkpoint(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    step: int,
    cfg: BetaVLATrainConfig,
    best_loss: float,
) -> None:
    ckpt_dir = _checkpoint_dir(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir = ckpt_dir / "best"
    tmp = ckpt_dir / "tmp_best"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    m = model.module if isinstance(model, DDP) else model
    safetensors.torch.save_model(m, tmp / "model.safetensors")
    torch.save(optim.state_dict(), tmp / "optimizer.pt")
    torch.save({"step": step, "best_loss": best_loss, "timestamp": time.time()}, tmp / "metadata.pt")
    _copy_norm_stats(cfg, tmp)
    if best_dir.exists():
        shutil.rmtree(best_dir)
    tmp.rename(best_dir)
    _best_info_path(cfg).write_text(
        json.dumps({"best_step": step, "best_loss": best_loss, "updated_at": time.time()}, indent=2),
        encoding="utf-8",
    )


def _latest_step(ckpt_dir: Path) -> int | None:
    if not ckpt_dir.exists():
        return None
    steps = [int(d.name) for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    return max(steps) if steps else None


def load_checkpoint_if_needed(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    cfg: BetaVLATrainConfig,
    device: torch.device,
) -> int:
    ckpt_dir = _checkpoint_dir(cfg)
    if cfg.runtime.overwrite and ckpt_dir.exists() and not cfg.runtime.resume:
        shutil.rmtree(ckpt_dir)
    if not cfg.runtime.resume:
        return 0
    m = model.module if isinstance(model, DDP) else model
    if cfg.runtime.resume_from_best:
        best_dir = ckpt_dir / "best"
        if not best_dir.exists():
            raise FileNotFoundError(f"No best checkpoint at {best_dir}")
        best_loss, best_step = _load_best_loss(cfg)
        safetensors.torch.load_model(m, best_dir / "model.safetensors", device=str(device))
        optim.load_state_dict(torch.load(best_dir / "optimizer.pt", map_location=device, weights_only=False))
        logging.info("Resumed from best checkpoint: step=%d loss=%.6f", best_step, best_loss)
        return best_step
    last_step = _latest_step(ckpt_dir)
    if last_step is None:
        raise FileNotFoundError(f"No checkpoints under {ckpt_dir}")
    step_dir = ckpt_dir / str(last_step)
    safetensors.torch.load_model(m, step_dir / "model.safetensors", device=str(device))
    optim.load_state_dict(torch.load(step_dir / "optimizer.pt", map_location=device, weights_only=False))
    logging.info("Resumed from step %d", last_step)
    return last_step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: BetaVLATrainConfig) -> None:
    logger = logging.getLogger("betavla.train")
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or dist.get_rank() == 0
    if is_main:
        logger.info("Device: %s | DDP: %s | world_size: %s", device, use_ddp,
                    dist.get_world_size() if use_ddp else 1)
    set_seed(cfg.runtime.seed + local_rank)
    maybe_init_wandb(cfg, is_main)

    # Dataset
    dataset = LiberoDataset(
        cfg.data,
        action_horizon=cfg.model.action_horizon,
        action_dim=cfg.model.action_dim,
        tokenizer_name=cfg.model.language.model_name,
    )
    sampler = None
    if use_ddp and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = create_dataloader(
        cfg.data,
        batch_size=cfg.runtime.batch_size,
        action_horizon=cfg.model.action_horizon,
        action_dim=cfg.model.action_dim,
        tokenizer_name=cfg.model.language.model_name,
        dataset=dataset,
        sampler=sampler,
    )
    if is_main:
        logger.info("Dataset size: %d samples", len(dataset))

    # Model
    model = BetaVLAModel(cfg.model).to(device)
    if is_main:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Params: trainable=%s total=%s (%.1f%%)",
                    f"{trainable:,}", f"{total:,}", 100.0 * trainable / max(total, 1))

    if use_ddp:
        # find_unused_parameters=True when some components are frozen
        # (frozen params don't get gradients → "unused" in DDP sense)
        has_frozen = cfg.model.freeze_vision or cfg.model.freeze_language or cfg.model.freeze_vggt
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=has_frozen,
        )

    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.runtime.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=cfg.runtime.weight_decay,
    )

    start_step = load_checkpoint_if_needed(model, optim, cfg, device)
    best_loss, best_step = _load_best_loss(cfg) if is_main else (float("inf"), -1)

    model.train()
    use_amp = cfg.runtime.use_amp and device.type == "cuda"
    grad_accum = cfg.runtime.grad_accumulation_steps
    if is_main:
        logger.info("Starting training: steps=%d AMP=%s grad_accum=%s lr_schedule=%s",
                    cfg.runtime.num_train_steps, use_amp, grad_accum, cfg.runtime.lr_schedule)

    loader_iter = iter(loader)
    if use_ddp and sampler is not None:
        sampler.set_epoch(0)

    step = start_step
    accum_count = 0
    loss_window: list[float] = []  # smoothed over log_interval steps

    while step < cfg.runtime.num_train_steps:
        try:
            observation, actions = next(loader_iter)
        except StopIteration:
            if use_ddp and sampler is not None:
                sampler.set_epoch(step // max(1, len(loader)))
            loader_iter = iter(loader)
            observation, actions = next(loader_iter)

        observation = observation.to(device)
        actions = actions.to(device=device, dtype=torch.float32)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(observation, actions)
            loss = out["loss"] / grad_accum

        loss.backward()
        loss_window.append(float(loss.item() * grad_accum))
        accum_count += 1

        if accum_count < grad_accum:
            continue

        # LR update
        lr = get_lr(step, cfg)
        for pg in optim.param_groups:
            pg["lr"] = lr

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.runtime.grad_clip_norm)
        optim.step()
        optim.zero_grad(set_to_none=True)
        accum_count = 0
        step += 1

        if is_main and step % cfg.runtime.log_interval == 0 and loss_window:
            avg_loss = sum(loss_window) / len(loss_window)
            logger.info("step=%d loss=%.6f lr=%.2e", step, avg_loss, lr)
            wandb.log({"loss": avg_loss, "learning_rate": lr, "step": step}, step=step)

            # Track best checkpoint using smoothed loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_step = step
                save_best_checkpoint(model, optim, step, cfg, best_loss)
                logger.info("New best model: step=%d loss=%.6f", step, best_loss)
                wandb.log({"best_loss": best_loss, "best_step": step}, step=step)

            loss_window = []

        if is_main and step % cfg.runtime.save_interval == 0:
            save_checkpoint(model, optim, step, cfg)
            logger.info("Checkpoint saved at step=%d", step)

    if is_main:
        save_checkpoint(model, optim, step, cfg)
        logger.info("Training done. Best: step=%d loss=%.6f", best_step, best_loss)

    wandb.finish()
    cleanup_ddp()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Beta-VLA")
    p.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    return p.parse_args()


def main() -> None:
    init_logging()
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
