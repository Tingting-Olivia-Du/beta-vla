# beta-vla

This directory implements a VLA stack using:
- Vision tower: SigLIP + DINOv2 (OpenVLA-style dual tower)
- Language encoder: `Qwen/Qwen3-0.6B-Base`
- Backbone: `facebook/VGGT-1B`
- Action head: OpenPI-style flow matching head (continuous actions)

## Dependencies

- This repo is self-contained; dependencies are installed from `beta-vla/pyproject.toml`.
- LoRA requires `peft` (for example: `uv add peft` in this project env).

## Layout

- `src/betavla/models/vision_tower.py`: dual vision tower
- `src/betavla/models/language_encoder.py`: Qwen encoder wrapper
- `src/betavla/models/vggt_backbone.py`: VGGT wrapper
- `src/betavla/models/action_head.py`: flow-matching action head
- `src/betavla/models/beta_vla_model.py`: full end-to-end model
- `src/betavla/training/config_beta_vla.py`: beta-vla train config schema
- `src/betavla/training/train_beta_vla.py`: training entrypoint
- `configs/train_beta_vla_libero.yaml`: training YAML example
- `scripts/gpu_select_train.sh`: GPU selector helper
- `scripts/train_beta_vla.sh`: single/multi GPU launcher
- `scripts/smoke_test_forward.sh`: Milestone-1 forward smoke test
- `docs/milestones.md`: validation and acceptance checklist

## Milestone checks

1. Milestone 1 (forward):
   - `cd /umd-datapool/tingting/beta-vla`
   - `bash scripts/smoke_test_forward.sh`

2. Milestone 2 (100-500 step smoke train):
   - set `num_train_steps: 500` in `configs/train_beta_vla_libero.yaml`
   - `bash scripts/train_beta_vla.sh --gpus 0`
   - default env is conda `beta`; override with `CONDA_ENV_NAME=<env>`

3. Milestone 3 (LIBERO subset):
   - update `data.repo_id` / `data.split` if needed
   - launch multi-GPU:
     - `bash scripts/train_beta_vla.sh --gpus 0,1 --ddp`

4. Milestone 4 (action IO checks):
   - verify `action_dim` and `action_horizon` match target policy interface
   - inspect checkpoint output under `checkpoints/<exp_name>/`

## LoRA switch

In `configs/train_beta_vla_libero.yaml`, set:
- `model.use_lora: true|false`
- `model.lora_on_language: true|false`
- `model.lora_on_vggt: true|false`
- `model.lora_r`, `model.lora_alpha`, `model.lora_dropout`
- `model.lora_target_modules` (target linear names)
