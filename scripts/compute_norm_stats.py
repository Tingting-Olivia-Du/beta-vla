"""Compute normalization statistics (mean, std, q01, q99) for LIBERO dataset.

Uses full training pipeline + RunningStats (OpenPI-style) so stats match training data.
Saves norm_stats.json for action/state quantile normalization.

Usage:
  python scripts/compute_norm_stats.py --config configs/libero_vggt.yaml
  python scripts/compute_norm_stats.py --config configs/libero_vggt.yaml --max_samples 5000 --out_dir assets/physical-intelligence/libero
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from betavla.data.libero_dataset import LiberoDatasetConfig, LiberoDataset, create_dataloader
from betavla.data.running_stats import RunningStats
from betavla.training.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute norm stats via full pipeline (OpenPI-style)"
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/libero_vggt.yaml"),
        help="Path to train config (provides data/model params)",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit samples for faster run (default: full dataset)",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output dir for norm_stats.json (default: assets/physical-intelligence/libero)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for dataloader",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers",
    )
    args = p.parse_args()

    cfg = load_config(args.config)

    # Use same data config as training, but WITHOUT norm_stats (raw data for stats)
    loader_cfg = LiberoDatasetConfig(
        repo_id=cfg.data.repo_id,
        split=cfg.data.split,
        num_workers=args.num_workers,
        max_token_len=cfg.data.max_token_len,
        max_samples=args.max_samples or cfg.data.max_samples,
        state_dim=cfg.data.state_dim,
        norm_stats_path=None,  # critical: no normalization when computing stats
    )

    dataset = LiberoDataset(
        loader_cfg,
        action_horizon=cfg.model.action_horizon,
        action_dim=cfg.model.action_dim,
        tokenizer_name=cfg.model.language.model_name,
    )

    loader = create_dataloader(
        loader_cfg,
        batch_size=args.batch_size,
        action_horizon=cfg.model.action_horizon,
        action_dim=cfg.model.action_dim,
        tokenizer_name=cfg.model.language.model_name,
        dataset=dataset,
    )

    state_stats = RunningStats()
    action_stats = RunningStats()

    num_batches = len(loader)
    if args.max_samples is not None:
        num_batches = min(num_batches, (args.max_samples + args.batch_size - 1) // args.batch_size)

    for batch in tqdm(loader, total=num_batches, desc="Computing norm stats"):
        obs, actions = batch
        state = obs.state.numpy()           # (B, state_dim)
        actions_np = actions.numpy()        # (B, action_horizon, action_dim)

        state_stats.update(state)
        # flatten action horizon: (B, T, D) -> (B*T, D)
        actions_flat = actions_np.reshape(-1, actions_np.shape[-1])
        action_stats.update(actions_flat)

        if args.max_samples is not None and state_stats.count >= args.max_samples:
            break

    # get_statistics() returns dict with keys: mean, std, q01, q99 (each a list of floats)
    state_dict = state_stats.get_statistics()
    action_dict = action_stats.get_statistics()

    norm_stats = {
        "state": state_dict,
        "action": action_dict,
    }

    out_dir = args.out_dir or Path("assets/physical-intelligence/libero")
    out_path = out_dir / "norm_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\nSaved norm_stats to {out_path}")
    print(f"  state:  dim={len(state_dict['mean'])}, samples={state_stats.count}")
    print(f"  action: dim={len(action_dict['mean'])}, samples={action_stats.count}")
    print(f"\nState  q01: {[round(x, 4) for x in state_dict['q01']]}")
    print(f"State  q99: {[round(x, 4) for x in state_dict['q99']]}")
    print(f"Action q01: {[round(x, 4) for x in action_dict['q01']]}")
    print(f"Action q99: {[round(x, 4) for x in action_dict['q99']]}")


if __name__ == "__main__":
    main()
