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

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
MAX_TOKENS="${MAX_TOKENS:-260}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SUCCESS_CASES="${SUCCESS_CASES:-20000}"
EVAL_EVERY="${EVAL_EVERY:-0}"
EVAL_SUCCESS_MILESTONES="${EVAL_SUCCESS_MILESTONES:-5000,10000,15000,20000}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-semantic}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-pritamdeka/S-PubMedBert-MS-MARCO}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-qwen36_35b_a3b_baseline_eval}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-qwen36_35b_a3b_20k_semantic_rag_eval_milestones}"

echo "Running disease-only SEAL/RCL benchmark"
echo "  model: ${MODEL_NAME}"
echo "  target successful cases: ${SUCCESS_CASES}"
echo "  held-out success milestones: ${EVAL_SUCCESS_MILESTONES}"
echo "  retrieval: ${RETRIEVAL_MODE}"
echo "  embedding model: ${EMBEDDING_MODEL}"
echo "  embedding device: ${EMBEDDING_DEVICE}"

python run_benchmark.py \
  --eval_dataset evaluation/id_expert_hard_eval_200.json \
  --kb_path data/infectious_diseases_seal_expert_rcl_kb_v6_compact.json \
  --sampler_config data/expert_curriculum_sampler_config_v1.json \
  --backend hf \
  --model "${MODEL_NAME}" \
  --baseline_run_name "${BASELINE_RUN_NAME}" \
  --train_run_name "${TRAIN_RUN_NAME}" \
  --n_successful_cases "${SUCCESS_CASES}" \
  --max_tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --log_every 10 \
  --retrieval_mode "${RETRIEVAL_MODE}" \
  --embedding_model "${EMBEDDING_MODEL}" \
  --embedding_device "${EMBEDDING_DEVICE}" \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
  --eval_every "${EVAL_EVERY}" \
  --eval_success_milestones "${EVAL_SUCCESS_MILESTONES}"
