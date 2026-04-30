#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-pritamdeka/S-PubMedBert-MS-MARCO}"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"

echo "Preloading Hugging Face repos"
echo "  python: ${PYTHON_BIN}"
echo "  model: ${MODEL_NAME}"
echo "  embedding model: ${EMBEDDING_MODEL}"
echo "  HF_HOME: ${HF_HOME}"
echo "  HF_HUB_CACHE: ${HF_HUB_CACHE}"

"${PYTHON_BIN}" - <<'PY'
import os
from huggingface_hub import snapshot_download

repos = [
    os.environ["MODEL_NAME"],
    os.environ["EMBEDDING_MODEL"],
]
cache_dir = os.environ["HF_HUB_CACHE"]

for repo_id in repos:
    print(f"\nDownloading {repo_id} into {cache_dir}")
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_files_only=False,
    )
    print(f"Cached snapshot: {path}")

print("\nDone. Submit the batch job with the same HF_HOME/HF_HUB_CACHE values.")
PY
