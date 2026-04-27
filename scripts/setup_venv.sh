#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${ROOT_DIR}/.venv"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Required interpreter not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=... if you want to use a different Python." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Qwen3.6 requires the latest transformers according to the official model card.
python -m pip install --upgrade \
  "torch>=2.7" \
  "transformers[serving]" \
  "accelerate>=1.10" \
  "safetensors>=0.6" \
  "sentencepiece>=0.2" \
  "huggingface_hub>=0.34" \
  "torchvision" \
  "pillow"

mkdir -p \
  "${ROOT_DIR}/hf_cache" \
  "${ROOT_DIR}/hf_cache/hub" \
  "${ROOT_DIR}/hf_cache/transformers" \
  "${ROOT_DIR}/hf_cache/datasets"

python - <<'PY'
import importlib
modules = ["torch", "transformers", "accelerate", "huggingface_hub", "safetensors"]
for name in modules:
    importlib.import_module(name)
print("Environment is ready.")
PY

cat <<EOF

Created virtual environment: ${VENV_DIR}
Activate it with:
  source "${VENV_DIR}/bin/activate"

EOF
