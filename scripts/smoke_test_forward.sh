#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/train_beta_vla_libero.yaml}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-beta}"
HF_HOME_DEFAULT="${HF_HOME_DEFAULT:-/data/hf-home}"
TRANSFORMERS_CACHE_DEFAULT="${TRANSFORMERS_CACHE_DEFAULT:-/data/hf-home/transformers}"
HF_DATASETS_CACHE_DEFAULT="${HF_DATASETS_CACHE_DEFAULT:-/data/hf-home/datasets}"
export BETA_VLA_CONFIG_PATH="${CONFIG_PATH}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME_DEFAULT}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE_DEFAULT}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_DEFAULT}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

conda run -n "${CONDA_ENV_NAME}" python - <<'PY'
import torch

from betavla.data.libero_loader import create_libero_dataloader
from betavla.models.beta_vla_model import BetaVLAModel
from betavla.training.config_beta_vla import load_beta_vla_config

import os

cfg = load_beta_vla_config(os.environ["BETA_VLA_CONFIG_PATH"])
loader = create_libero_dataloader(
    cfg.data,
    batch_size=cfg.runtime.batch_size,
    action_horizon=cfg.model.action_horizon,
    action_dim=cfg.model.action_dim,
    tokenizer_name=cfg.model.language.model_name,
)
obs, act = next(iter(loader))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
obs = obs.to(device)
act = act.to(device=device, dtype=torch.float32)

model = BetaVLAModel(cfg.model).to(device)
out = model(obs, act)
print({"loss": float(out["loss"].item())})
PY
