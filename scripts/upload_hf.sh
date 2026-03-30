#!/bin/bash
# Upload checkpoint to HuggingFace
# Usage:
#   bash scripts/upload_hf.sh <local_path> [remote_path]
#
# Examples:
#   bash scripts/upload_hf.sh checkpoints/beta-0329/best beta-0329/best
#   bash scripts/upload_hf.sh checkpoints/beta-0329/best              # remote defaults to same path
#
# Config: edit .env to set HF_TOKEN and HF_REPO

set -e
cd "$(dirname "$0")/.."

# Load .env
if [ ! -f .env ]; then
    echo "Error: .env not found. Fill in your token in .env"
    exit 1
fi
source .env

if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "hf_YOUR_TOKEN_HERE" ]; then
    echo "Error: Set your HF_TOKEN in .env"
    exit 1
fi

LOCAL_PATH="${1:?Usage: bash scripts/upload_hf.sh <local_path> [remote_path]}"
REMOTE_PATH="${2:-$1}"

echo "Repo:   $HF_REPO"
echo "Local:  $LOCAL_PATH"
echo "Remote: $REMOTE_PATH"
echo ""

# Create repo if not exists (ignore error if already exists)
hf repo create "$(basename "$HF_REPO")" --type model --token "$HF_TOKEN" 2>/dev/null || true

# Upload
hf upload "$HF_REPO" "$LOCAL_PATH" "$REMOTE_PATH" --token "$HF_TOKEN"

echo ""
echo "Done! https://huggingface.co/$HF_REPO"
