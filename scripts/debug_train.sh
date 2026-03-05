#!/usr/bin/env bash
# Minimal debug script to isolate where training hangs.
# Run: cd beta-vla && conda activate beta && bash scripts/debug_train.sh
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/umd-datapool/tingting/hf-home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/umd-datapool/tingting/hf-home/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/umd-datapool/tingting/hf-home/datasets}"
export WANDB_DIR="${WANDB_DIR:-/umd-datapool/tingting/wandb}"
export TMPDIR="${TMPDIR:-/umd-datapool/tingting/.tmp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${WANDB_DIR}" "${TMPDIR}"

echo "[debug] Step 1: import torch"
python -u -c "import torch; print('[debug] torch OK, cuda:', torch.cuda.is_available(), flush=True)"

echo "[debug] Step 2: import wandb"
python -u -c "import wandb; print('[debug] wandb OK', flush=True)"

echo "[debug] Step 3: import betavla + load_dataset path"
python -u -c "
from datasets import load_dataset
print('[debug] datasets OK', flush=True)
"

echo "[debug] Step 4: load libero dataset (may download)"
python -u -c "
from datasets import load_dataset
ds = load_dataset('physical-intelligence/libero', split='train')
print('[debug] libero loaded, len=', len(ds), flush=True)
"

echo "[debug] Step 5: load tokenizer"
python -u -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)
print('[debug] tokenizer OK', flush=True)
"

echo "[debug] Step 6: create BetaVLAModel (loads DINO, SigLIP, Qwen, VGGT)"
python -u -c "
from betavla.training.config_beta_vla import load_beta_vla_config
from betavla.models.beta_vla_model import BetaVLAModel
cfg = load_beta_vla_config('configs/train_beta_vla_libero.yaml')
print('[debug] loading model...', flush=True)
model = BetaVLAModel(cfg.model)
print('[debug] model OK', flush=True)
"

echo "[debug] All steps passed."
