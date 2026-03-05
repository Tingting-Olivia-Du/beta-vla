#!/usr/bin/env bash
# Self-contained beta-vla environment bootstrap.
set -euo pipefail

ENV_NAME="${ENV_NAME:-beta}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BETAVLA_DIR="${ROOT_DIR}/beta-vla"
# 全部放到 umd-datapool，避免 root / 和 /data 满
UV_CACHE_DIR_DEFAULT="${UV_CACHE_DIR_DEFAULT:-/umd-datapool/tingting/uv-cache}"
TMPDIR_DEFAULT="${TMPDIR_DEFAULT:-/umd-datapool/tingting/.tmp}"
# HF cache: use umd-datapool to avoid filling root / or /data
HF_HOME_DEFAULT="${HF_HOME_DEFAULT:-/umd-datapool/tingting/hf-home}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] conda not found. Please install conda first."
  exit 1
fi

if [[ ! -d "${BETAVLA_DIR}" ]]; then
  echo "[setup] Missing: ${BETAVLA_DIR}"
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] Conda env '${ENV_NAME}' exists, reusing."
else
  echo "[setup] Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION}) ..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

echo "[setup] Installing/upgrading uv ..."
conda run -n "${ENV_NAME}" pip install -U uv

export UV_LINK_MODE=copy
export UV_CACHE_DIR="${UV_CACHE_DIR_DEFAULT}"
export TMPDIR="${TMPDIR_DEFAULT}"
export HF_HOME="${HF_HOME_DEFAULT}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${UV_CACHE_DIR}" "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"
echo "[setup] UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "[setup] TMPDIR=${TMPDIR}"
echo "[setup] HF_HOME=${HF_HOME}"

echo "[setup] Installing beta-vla and all dependencies ..."
conda run -n "${ENV_NAME}" bash -lc "cd \"${BETAVLA_DIR}\" && uv pip install -e . --system"

echo "[setup] Done. Activate: conda activate ${ENV_NAME}"
echo "[setup] Verify: bash scripts/smoke_test_forward.sh"
