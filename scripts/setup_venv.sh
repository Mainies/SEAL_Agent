#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${ROOT_DIR}/.venv"
REQ_FILE="${REQ_FILE:-requirements-hf.txt}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Required interpreter not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=... if you want to use a different Python." >&2
  exit 1
fi

if [[ ! -f "${ROOT_DIR}/${REQ_FILE}" ]]; then
  echo "Requirements file not found: ${ROOT_DIR}/${REQ_FILE}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

python -m pip install --upgrade -r "${ROOT_DIR}/${REQ_FILE}"

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

Installed requirements from:
  ${ROOT_DIR}/${REQ_FILE}

EOF
