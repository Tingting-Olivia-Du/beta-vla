#!/usr/bin/env bash
# Self-contained beta-vla environment checker.
# Usage: CONDA_ENV_NAME=beta bash scripts/check_env.sh

set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-beta}"
BETAVLA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Beta-VLA Environment Check ==="
echo "Conda env: ${CONDA_ENV_NAME}"
echo "beta-vla: ${BETAVLA_DIR}"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  echo "[FAIL] Conda env '${CONDA_ENV_NAME}' not found."
  exit 1
fi
echo "[OK] Conda env exists."

echo "--- Python module checks ---"
conda run -n "${CONDA_ENV_NAME}" python - <<'PY'
import importlib
import sys

required = [
    "torch",
    "transformers",
    "datasets",
    "timm",
    "peft",
    "wandb",
    "betavla.training.train_beta_vla",
]
failed = []
for mod in required:
    try:
        importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001
        failed.append((mod, str(e)))

if failed:
    print("[FAIL] Missing/broken modules:")
    for m, err in failed:
        print(f"  - {m}: {err}")
    sys.exit(1)
print("[OK] All required modules import successfully.")
PY

echo "--- Key versions ---"
conda run -n "${CONDA_ENV_NAME}" python - <<'PY'
import datasets
import timm
import torch
import transformers

print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"datasets: {datasets.__version__}")
print(f"timm: {timm.__version__}")
PY

echo "=== Check finished ==="
