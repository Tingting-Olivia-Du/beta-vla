#!/usr/bin/env bash
# ./scripts/train_beta_vla.sh
# Run from beta-vla dir: conda activate beta && GPU_IDS="0,1" bash scripts/train_beta_vla.sh
set -euo pipefail

cd "$(dirname "$0")/.."
# 统一 HF/wandb 路径到 umd-datapool（避免 /root/.cache 爆满）
source "$(dirname "$0")/env_umd_datapool.sh"
# 若存在则加载 wandb 配置（WANDB_API_KEY 等，配置一次即可，见 env_wandb.local.sh.example）
[[ -f "$(dirname "$0")/env_wandb.local.sh" ]] && source "$(dirname "$0")/env_wandb.local.sh"
CONFIG_PATH="configs/train_beta_vla_libero_paligemma.yaml"
# Respect GPU_IDS from env (e.g. GPU_IDS="0" for single GPU)
GPU_IDS="${GPU_IDS:-1,2}"
RUN_DDP=1
CONDA_ENV_NAME="${CONDA_ENV_NAME:-beta}"
# 路径由 env_umd_datapool.sh 统一设置；此处仅作兜底
HF_HOME_DEFAULT="${HF_HOME_DEFAULT:-/umd-datapool/tingting/hf-home}"
TRANSFORMERS_CACHE_DEFAULT="${TRANSFORMERS_CACHE_DEFAULT:-/umd-datapool/tingting/hf-home/transformers}"
HF_DATASETS_CACHE_DEFAULT="${HF_DATASETS_CACHE_DEFAULT:-/umd-datapool/tingting/hf-home/datasets}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --gpus)
      GPU_IDS="$2"
      shift 2
      ;;
    --ddp)
      RUN_DDP=1
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install/use conda first."
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
GPU_COUNT=0
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ -n "${gpu// /}" ]]; then
    GPU_COUNT=$((GPU_COUNT + 1))
  fi
done

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
# 确保 HF/wandb 在 umd-datapool（env 已 source，此处兜底）
export HF_HOME="${HF_HOME:-${HF_HOME_DEFAULT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME_DEFAULT}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${TRANSFORMERS_CACHE_DEFAULT}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_DATASETS_CACHE_DEFAULT}}"
export WANDB_DIR="${WANDB_DIR:-/umd-datapool/tingting/wandb}"
# If script hangs at wandb: set WANDB_MODE=offline or wandb_enabled: false in config
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${WANDB_DIR}" "${TMPDIR}"

# Use conda activate + direct python (conda run can drop env vars / buffer output)
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"
python - <<'PY'
import sys

checks = [
    ("torch", "import torch"),
    ("transformers", "import transformers"),
    ("timm", "import timm"),
    ("datasets", "import datasets"),
    ("betavla", "import betavla.training.train_beta_vla"),
]
errors = []
for name, stmt in checks:
    try:
        exec(stmt, {})
    except Exception as e:  # noqa: BLE001
        errors.append((name, str(e)))
if errors:
    print("[beta-vla] Environment preflight failed.")
    for name, err in errors:
        print(f"  - {name}: {err}")
    print("[beta-vla] Please install beta-vla dependencies in this conda env.")
    sys.exit(1)
PY

# Use PYTHONUNBUFFERED=1 so output appears immediately (avoids "hanging" appearance with conda run)
export PYTHONUNBUFFERED=1
# Reduce CUDA memory fragmentation (helps with OOM during optimizer step)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ "${RUN_DDP}" -eq 1 && "${GPU_COUNT}" -gt 1 ]]; then
  ./scripts/gpu_select_train.sh "${GPU_IDS}" \
    python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${GPU_COUNT}" \
    -m betavla.training.train_beta_vla --config "${CONFIG_PATH}"
else
  ./scripts/gpu_select_train.sh "${GPU_IDS}" \
    python -u -m betavla.training.train_beta_vla --config "${CONFIG_PATH}"
fi
