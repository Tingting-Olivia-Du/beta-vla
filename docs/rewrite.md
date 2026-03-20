# Single GPU
bash scripts/train.sh 0 configs/libero_vggt.yaml

# 4 GPUs
bash scripts/train.sh 0,1,2,3 configs/libero_vggt.yaml

# Eval
bash scripts/eval_libero.sh 0 checkpoints/libero_vggt/best libero_spatial