# Beta-VLA Validation Milestones

## Milestone 1: Forward smoke test
- Command: `bash scripts/smoke_test_forward.sh configs/train_beta_vla_libero.yaml`
- Pass criteria:
  - one batch runs through `vision -> language -> VGGT -> flow head`
  - script prints a finite scalar `loss`

## Milestone 2: Training loop smoke test (100-500 steps)
- Config: set `runtime.num_train_steps` to `100` or `500`
- Command: `bash scripts/train_beta_vla.sh --config configs/train_beta_vla_libero.yaml --gpus 0`
- Pass criteria:
  - loss logs appear every `runtime.log_interval`
  - checkpoint folders appear every `runtime.save_interval`

## Milestone 3: LIBERO subset run
- Config:
  - set `data.repo_id` / `data.split` / `data.max_samples` as needed
  - set real train length and expected checkpoint cadence
- Command (multi-GPU): `bash scripts/train_beta_vla.sh --config configs/train_beta_vla_libero.yaml --gpus 0,1 --ddp`
- Pass criteria:
  - DDP initializes with expected world size
  - training and checkpoint save are stable for target duration

## Milestone 4: Action interface validation
- Verify that:
  - `model.action_dim` equals runtime consumer action dimension
  - `model.action_horizon` equals consumer action chunk size
  - inference output shape is `[batch, action_horizon, action_dim]`
- Pass criteria:
  - action tensor shape matches downstream policy/runtime contract
