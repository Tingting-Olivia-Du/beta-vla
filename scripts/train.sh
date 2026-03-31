#!/bin/bash
# Train Beta-VLA with torchrun (single or multi-GPU).
# set up huggingface first! gated repo! 
# Usage:
#   bash scripts/train.sh                              # GPU 0, default config
#   bash scripts/train.sh 0,1,2,3                      # GPUs 0-3, default config
#   bash scripts/train.sh 1,2 configs/libero_vggt.yaml
#   bash scripts/train.sh 4,5,6,7 configs/libero_vggt.yaml
#
# To use a single GPU:
#   bash scripts/train.sh 0

set -euo pipefail

cd "$(dirname "$0")/.."
# 统一 HF/wandb 路径到 umd-datapool（避免 /root/.cache 爆满）
source "$(dirname "$0")/env_umd_datapool.sh"
# 若存在则加载 wandb 配置（WANDB_API_KEY 等，配置一次即可，见 env_wandb.local.sh.example）
[[ -f "$(dirname "$0")/env_wandb.local.sh" ]] && source "$(dirname "$0")/env_wandb.local.sh"


GPUS="${1:-0}"
CONFIG="${2:-configs/libero_vggt.yaml}"

# Count GPUs
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l | tr -d ' ')

echo "==> Training Beta-VLA"
echo "    GPUs:   $GPUS  (num=$NUM_GPUS)"
echo "    Config: $CONFIG"

export CUDA_VISIBLE_DEVICES="$GPUS"

if [ "$NUM_GPUS" -eq 1 ]; then
    python src/betavla/training/train.py --config "$CONFIG"
else
    torchrun \
        --standalone \
        --nproc_per_node="$NUM_GPUS" \
        src/betavla/training/train.py \
        --config "$CONFIG"
fi
