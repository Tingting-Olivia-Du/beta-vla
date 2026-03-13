#!/bin/bash
# Evaluate Beta-VLA on LIBERO benchmark.
#
# Usage:
#   bash scripts/eval_libero.sh                                       # GPU 0, best checkpoint
#   bash scripts/eval_libero.sh 0 checkpoints/libero_vggt/best
#   bash scripts/eval_libero.sh 4,5,6,7 checkpoints/libero_vggt/best libero_10
#   bash scripts/eval_libero.sh 0 checkpoints/libero_vggt/best all   # all 6 suites
#
# Args:
#   $1  GPU ids (comma-separated, default "0")
#   $2  Checkpoint path (default "checkpoints/libero_vggt/best")
#   $3  Task suite name or "all" (default "libero_spatial")

set -euo pipefail

GPUS="${1:-0}"
CHECKPOINT="${2:-checkpoints/libero_vggt/best}"
TASK_SUITE="${3:-libero_spatial}"

echo "==> Evaluating Beta-VLA on LIBERO"
echo "    GPUs:       $GPUS"
echo "    Checkpoint: $CHECKPOINT"
echo "    Suite:      $TASK_SUITE"

export CUDA_VISIBLE_DEVICES="$GPUS"

# Count GPUs for multi-GPU eval (each GPU runs a worker process)
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l | tr -d ' ')
GPU_ARG=""
if [ "$NUM_GPUS" -gt 1 ]; then
    GPU_ARG="--gpus $GPUS"
fi

python scripts/eval_libero.py \
    --checkpoint "$CHECKPOINT" \
    --task_suite "$TASK_SUITE" \
    --num_trials_per_task 20 \
    --replan_steps 5 \
    --num_ode_steps 5 \
    ${GPU_ARG}
