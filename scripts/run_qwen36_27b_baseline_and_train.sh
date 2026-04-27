#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Missing virtual environment at ${VENV_DIR}. Run scripts/setup_venv.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "${ROOT_DIR}"

export HF_HOME="${ROOT_DIR}/hf_cache"
export HF_HUB_CACHE="${ROOT_DIR}/hf_cache/hub"
export TRANSFORMERS_CACHE="${ROOT_DIR}/hf_cache/transformers"
export HF_DATASETS_CACHE="${ROOT_DIR}/hf_cache/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1

MODEL_NAME="Qwen/Qwen3.6-27B"
MAX_TOKENS="${MAX_TOKENS:-260}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SUCCESS_CASES="${SUCCESS_CASES:-10000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-qwen36_27b_baseline_eval}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-qwen36_27b_10k_eval_every_1000}"

python run_evaluation.py \
  --eval_dataset evaluation/id_expert_hard_eval_200.json \
  --kb_path data/infectious_diseases_seal_expert_rcl_kb_v6_compact.json \
  --backend hf \
  --model "${MODEL_NAME}" \
  --run_name "${BASELINE_RUN_NAME}" \
  --eval_mode no_memory \
  --max_tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}"

python run_simulation.py \
  --backend hf \
  --model "${MODEL_NAME}" \
  --kb_path data/infectious_diseases_seal_expert_rcl_kb_v6_compact.json \
  --sampler_config data/expert_curriculum_sampler_config_v1.json \
  --n_successful_cases "${SUCCESS_CASES}" \
  --run_name "${TRAIN_RUN_NAME}" \
  --max_tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --log_every 10 \
  --run_evaluation \
  --eval_every "${EVAL_EVERY}" \
  --eval_dataset evaluation/id_expert_hard_eval_200.json \
  --eval_mode with_memory
