from __future__ import annotations

import sys
print("[beta-vla] train_beta_vla module loading...", flush=True)
sys.stdout.flush()
sys.stderr.flush()

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import wandb

from betavla.data.libero_loader import create_libero_dataloader
from betavla.models.beta_vla_model import BetaVLAModel
from betavla.training.config_beta_vla import BetaVLATrainConfig, load_beta_vla_config


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_ddp() -> tuple[bool, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://")
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


def maybe_init_wandb(cfg: BetaVLATrainConfig, is_main: bool) -> None:
    if not is_main or not cfg.runtime.wandb_enabled:
        wandb.init(mode="disabled")
        return
    wandb.init(
        project=cfg.runtime.wandb_project,
        name=cfg.runtime.exp_name,
        config=dataclasses.asdict(cfg),
    )


def _checkpoint_dir(cfg: BetaVLATrainConfig) -> Path:
    return cfg.checkpoint_dir


def save_checkpoint(model: torch.nn.Module, optim: torch.optim.Optimizer, step: int, cfg: BetaVLATrainConfig) -> None:
    ckpt_dir = _checkpoint_dir(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step_dir = ckpt_dir / f"{step}"
    tmp = ckpt_dir / f"tmp_{step}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, DDP) else model
    safetensors.torch.save_model(model_to_save, tmp / "model.safetensors")
    torch.save(optim.state_dict(), tmp / "optimizer.pt")
    torch.save({"step": step, "timestamp": time.time()}, tmp / "metadata.pt")
    if step_dir.exists():
        shutil.rmtree(step_dir)
    tmp.rename(step_dir)


def _best_info_path(cfg: BetaVLATrainConfig) -> Path:
    return _checkpoint_dir(cfg) / "best.json"


def _load_best_loss(cfg: BetaVLATrainConfig) -> tuple[float, int]:
    info_path = _best_info_path(cfg)
    if not info_path.exists():
        return float("inf"), -1
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return float(info.get("best_loss", float("inf"))), int(info.get("best_step", -1))


def save_best_checkpoint(
    model: torch.nn.Module, optim: torch.optim.Optimizer, step: int, cfg: BetaVLATrainConfig, best_loss: float
) -> None:
    ckpt_dir = _checkpoint_dir(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dir = ckpt_dir / "best"
    tmp_best_dir = ckpt_dir / "tmp_best"

    if tmp_best_dir.exists():
        shutil.rmtree(tmp_best_dir)
    tmp_best_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, DDP) else model
    safetensors.torch.save_model(model_to_save, tmp_best_dir / "model.safetensors")
    torch.save(optim.state_dict(), tmp_best_dir / "optimizer.pt")
    torch.save({"step": step, "best_loss": best_loss, "timestamp": time.time()}, tmp_best_dir / "metadata.pt")

    if best_dir.exists():
        shutil.rmtree(best_dir)
    tmp_best_dir.rename(best_dir)

    _best_info_path(cfg).write_text(
        json.dumps({"best_step": step, "best_loss": best_loss, "updated_at": time.time()}, indent=2),
        encoding="utf-8",
    )


def _latest_checkpoint_step(ckpt_dir: Path) -> int | None:
    if not ckpt_dir.exists():
        return None
    steps = [int(d.name) for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    return max(steps) if steps else None


def load_checkpoint_if_needed(
    model: torch.nn.Module, optim: torch.optim.Optimizer, cfg: BetaVLATrainConfig, device: torch.device
) -> int:
    ckpt_dir = _checkpoint_dir(cfg)
    if cfg.runtime.overwrite and ckpt_dir.exists() and not cfg.runtime.resume:
        shutil.rmtree(ckpt_dir)
    if not cfg.runtime.resume:
        return 0
    last_step = _latest_checkpoint_step(ckpt_dir)
    if last_step is None:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir} for resume")
    model_to_load = model.module if isinstance(model, DDP) else model
    safetensors.torch.load_model(model_to_load, ckpt_dir / f"{last_step}" / "model.safetensors", device=str(device))
    optim.load_state_dict(torch.load(ckpt_dir / f"{last_step}" / "optimizer.pt", map_location=device, weights_only=False))
    return last_step


def build_loader(cfg: BetaVLATrainConfig, use_ddp: bool = False):
    from betavla.data.libero_loader import LiberoTorchDataset, create_libero_dataloader

    dataset = LiberoTorchDataset(
        cfg.data,
        action_horizon=cfg.model.action_horizon,
        action_dim=cfg.model.action_dim,
        tokenizer_name=cfg.model.language.model_name,
    )
    sampler = None
    if use_ddp and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
    return create_libero_dataloader(
        cfg.data,
        batch_size=cfg.runtime.batch_size,
        action_horizon=cfg.model.action_horizon,
        action_dim=cfg.model.action_dim,
        tokenizer_name=cfg.model.language.model_name,
        dataset=dataset,
        sampler=sampler,
    )


def train(cfg: BetaVLATrainConfig) -> None:
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or dist.get_rank() == 0
    if is_main:
        logging.info("[beta-vla] setup_ddp done, rank=%s device=%s", local_rank, device)
    set_seed(cfg.runtime.seed + local_rank)
    if is_main:
        logging.info("[beta-vla] initializing wandb...")
    maybe_init_wandb(cfg, is_main)
    if is_main:
        logging.info("[beta-vla] wandb done, building dataloader...")

    loader = build_loader(cfg, use_ddp=use_ddp)
    if is_main:
        logging.info("[beta-vla] dataloader ready, loading model...")
    model = BetaVLAModel(cfg.model).to(device)
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = 100.0 * trainable_params / max(total_params, 1)
        logging.info(
            "Model params: trainable=%s total=%s (%.2f%%)",
            f"{trainable_params:,}",
            f"{total_params:,}",
            ratio,
        )
    if use_ddp:
        if is_main:
            logging.info("[beta-vla] wrapping model with DDP (may take 1-2 min for 2B params)...")
        # find_unused_parameters=False avoids "marked as ready twice" with PEFT/LoRA
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=False)
        if is_main:
            logging.info("[beta-vla] DDP done")

    if is_main:
        logging.info("[beta-vla] creating optimizer...")
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.runtime.learning_rate,
        weight_decay=cfg.runtime.weight_decay,
    )
    start_step = load_checkpoint_if_needed(model, optim, cfg, device)
    best_loss, best_step = _load_best_loss(cfg) if is_main else (float("inf"), -1)
    model.train()
    if is_main:
        logging.info("[beta-vla] starting training loop (first batch may take 1-2 min)...")
    if use_ddp and hasattr(loader, "sampler") and loader.sampler is not None:
        loader.sampler.set_epoch(0)

    step = start_step
    log_cache: list[float] = []
    use_amp = cfg.runtime.use_amp and device.type == "cuda"
    grad_accum = cfg.runtime.grad_accumulation_steps
    if is_main:
        logging.info("[beta-vla] use_amp=%s grad_accumulation_steps=%s", use_amp, grad_accum)

    loader_iter = iter(loader)
    accum_count = 0
    while step < cfg.runtime.num_train_steps:
        try:
            observation, actions = next(loader_iter)
        except StopIteration:
            if use_ddp and hasattr(loader, "sampler") and loader.sampler is not None:
                loader.sampler.set_epoch(step // max(1, len(loader)))
            loader_iter = iter(loader)
            observation, actions = next(loader_iter)

        observation = observation.to(device)
        actions = actions.to(device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(observation, actions)
            loss = out["loss"] / grad_accum
        loss.backward()
        log_cache.append(float(loss.item() * grad_accum))
        accum_count += 1

        if accum_count < grad_accum:
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.runtime.grad_clip_norm)
        optim.step()
        optim.zero_grad(set_to_none=True)
        accum_count = 0

        if is_main:
            step_loss = sum(log_cache[-grad_accum:]) / grad_accum
            if step_loss < best_loss:
                best_loss = step_loss
                best_step = step
                save_best_checkpoint(model, optim, step, cfg, best_loss)
                logging.info("new best model at step=%s loss=%.6f", step, best_loss)
                wandb.log({"best_loss": best_loss, "best_step": step}, step=step)

        if is_main and step % cfg.runtime.log_interval == 0 and log_cache:
            avg_loss = sum(log_cache) / len(log_cache)
            logging.info("step=%s loss=%.6f", step, avg_loss)
            wandb.log({"loss": avg_loss, "step": step}, step=step)
            log_cache = []

        step += 1
        if is_main and step % cfg.runtime.save_interval == 0:
            save_checkpoint(model, optim, step, cfg)

    if is_main:
        save_checkpoint(model, optim, step, cfg)
        logging.info("best model summary: step=%s loss=%.6f", best_step, best_loss)
    wandb.finish()
    cleanup_ddp()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train self-contained Beta-VLA.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = parse_args()
    cfg = load_beta_vla_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
