#!/bin/bash
# Eval on TRAINING data (verify eval pipeline correctness)
# Usage: bash scripts/run_eval_on_train.sh --checkpoint checkpoints/beta_vla_libero/best
#
# Example:
#   快速测试(推荐，3 个 episode，快速验证 pipeline): --max_episodes 3 --quick_debug
#   MUJOCO_GL=egl bash scripts/run_eval_on_train.sh --checkpoint checkpoints/beta_vla_libero/best --max_episodes 3 --quick_debug
#   bash scripts/run_eval_on_train.sh --checkpoint ... --max_episodes 20
#   bash scripts/run_eval_on_train.sh --checkpoint ... --max_episodes 50 --no_video  # 默认 50
#
# 渲染: egl=GPU 快很多; osmesa=CPU 极慢(每 episode 可能 2h+)
# 若 EGL 报错: MUJOCO_GL=osmesa bash scripts/run_eval_on_train.sh ...

set -e
cd "$(dirname "$0")/.."
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate beta
pip install imageio -q 2>/dev/null || true
python scripts/eval_on_train_data.py "$@"
