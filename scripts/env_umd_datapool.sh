#!/usr/bin/env bash
# 设置环境变量：HF 用 umd-datapool；uv/pip 的 temp 用本地（NFS 上锁会失败）
# Usage: source scripts/env_umd_datapool.sh

# HF 缓存放 umd-datapool（模型大，本地可能满）
export HF_HOME="${HF_HOME:-/workspace/tingting/hf-home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/tingting/hf-home/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/tingting/hf-home/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/workspace/tingting/hf-home/datasets}"
export WANDB_DIR="${WANDB_DIR:-/workspace/tingting/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/workspace/tingting/.wandb-config}"

# uv/pip 的 temp 和 cache 必须用本地磁盘：NFS 上会 Remote I/O error (121)
# 强制覆盖已有值，优先 /data，其次 /tmp，最后 /dev/shm
if [[ -d /data ]] && [[ -w /data ]]; then
  export TMPDIR="/data/tingting-tmp"
  export UV_CACHE_DIR="/data/tingting-uv-cache"
elif [[ -d /tmp ]] && [[ -w /tmp ]]; then
  export TMPDIR="/tmp/tingting-tmp"
  export UV_CACHE_DIR="/tmp/tingting-uv-cache"
else
  export TMPDIR="/dev/shm/tingting-tmp"
  export UV_CACHE_DIR="/dev/shm/tingting-uv-cache"
fi

mkdir -p "${UV_CACHE_DIR}" "${TMPDIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${WANDB_DIR}" "${WANDB_CONFIG_DIR}"
echo "[env] TMPDIR=$TMPDIR (local, for uv/pip)"
echo "[env] UV_CACHE_DIR=$UV_CACHE_DIR"
echo "[env] HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE"
