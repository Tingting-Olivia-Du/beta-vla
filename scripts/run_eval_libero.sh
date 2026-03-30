#!/bin/bash
# Run LIBERO eval with beta env. Default: save video.
# Usage: bash scripts/run_eval_libero.sh [extra args]
#
# GPU 设置:
#   单卡: --gpus 7  或  --gpu 7
#   多卡: --gpus 4,5,6,7  (并行 eval，每卡一个 worker)
#
# Example:
#   bash scripts/run_eval_libero.sh --checkpoint checkpoints/beta_vla_libero/best --max_tasks 1 --num_trials_per_task 10 --no_norm_stats --gpus 7
#   bash scripts/run_eval_libero.sh --checkpoint checkpoints/beta_vla_libero/5000 --num_trials_per_task 1  --task_suite all --gpus 7

# bash scripts/run_eval_libero.sh --checkpoint checkpoints/beta-0329/best --verbose  --num_trials_per_task 20 --task_suite all --gpus 7 --video_out_path data/libero/beta-fulltune-0329
# bash scripts/run_eval_libero.sh --checkpoint checkpoints/beta-0329/best --num_trials_per_task 20 --task_suite libero_goal --gpus 7 --video_out_path data/libero/beta-fulltune-libero-goal-0330
#   bash scripts/run_eval_libero.sh --checkpoint ... --gpus 4,5,6,7 --max_tasks 2
#   全部 6 个 suite: --task_suite all
#
# 若 EGL 报错: MUJOCO_GL=osmesa bash scripts/run_eval_libero.sh ...

set -e
cd "$(dirname "$0")/.."
# robosuite 1.4.1 EGL 与 PyTorch CUDA 同进程约 40-50 次推理后 SIGABRT
# 根因在 robosuite EGL context 实现，egl;  分卡也无法绕过，必须用 osmesa
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
# 避免 tokenizer fork 后死锁警告
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# LIBERO 路径由 eval_libero.py 自动设置
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate beta
pip install imageio -q 2>/dev/null || true
python scripts/eval_libero.py "$@"
