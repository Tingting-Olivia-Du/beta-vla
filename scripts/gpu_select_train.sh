#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <gpu_ids_comma_sep> <command...>"
  echo "Example: $0 0,1 uv run torchrun --nproc_per_node=2 -m betavla.training.train_beta_vla --config configs/train_beta_vla_libero.yaml"
  exit 1
fi

GPU_IDS="$1"
shift

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
echo "[beta-vla] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
exec "$@"
