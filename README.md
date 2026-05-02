# SEAL_AGENTS

Infectious-disease doctor-agent evolution experiment.

This repository implements a local, contained SEAL/RCL-style loop for infectious-disease learning. The main benchmark is disease diagnosis: a doctor agent sees synthetic patient cases, answers with a diagnosis, is judged against a hidden condition, and accumulates memory from successful cases and validated failure reflections. There is also a text-only examination-selection extension where the doctor chooses what to examine, ask, collect, or test next before definitive results.

This is a research scaffold only. It is not clinical software and should not be used for patient care.

## Current Experiment

The active benchmark remains disease diagnosis. Examination selection is implemented as a separate text-only task, with separate memory files to avoid mixing diagnosis and examination experience. Treatment planning is still out of scope.

Default benchmark:

- Base model: `Qwen/Qwen3.6-35B-A3B`
- Runtime: local Hugging Face Transformers inside `.venv`
- Memory retrieval: semantic RAG
- Embedding model: `pritamdeka/S-PubMedBert-MS-MARCO`
- Successful cases retrieved per prompt: top 3
- Reflections/experiences retrieved per prompt: top 4
- Target successful cases: 20,000
- Held-out eval checkpoints: baseline, 5k, 10k, 15k, 20k successful cases
- Held-out eval set: `evaluation/id_expert_hard_eval_200.json`

The benchmark is single-process via `run_benchmark.py`, so the Qwen model is loaded once and kept in memory for baseline evaluation, training/evolution, and milestone evaluations.

## Conceptual Design

The loop has two learned memory stores:

1. Successful case base
   - Saved when the doctor answer is judged correct.
   - Stores the patient case, doctor answer, condition metadata, retrieval text, and retrieval tags.

2. Experience/reflection base
   - Saved only when an initial wrong answer produces a reflection and the same-patient retry succeeds.
   - This acts as validated learning from failure.
   - Failed or unhelpful reflections are discarded.

At inference time, the doctor prompt includes:

- the current synthetic patient case
- top 3 semantically similar successful cases
- top 4 semantically similar validated reflections

Semantic retrieval is local. It embeds patient/question text with a Hugging Face encoder and retrieves by cosine similarity. The older tag-overlap retrieval path is still available with `--retrieval_mode tag`.

## Main Files

- `run_benchmark.py`
  - Recommended benchmark entry point.
  - Runs baseline evaluation and training in one process so the model stays loaded.
  - Supports `diagnosis` and `examination_selection`.

- `run_simulation.py`
  - Lower-level training/evolution entry point.
  - Useful for custom diagnosis or examination-selection training runs.

- `run_evaluation.py`
  - Standalone held-out evaluation runner for diagnosis or examination selection.

- `src/backends.py`
  - LLM backend implementations.
  - Supports Ollama and Hugging Face Transformers.
  - Handles Qwen3.6 dense and MoE conditional-generation classes.

- `src/loop.py`
  - Main training loop.
  - Generates patient cases, retrieves memory, asks doctor, judges answer, records successes/reflections/discards, triggers milestone evaluation.

- `src/memory.py`
  - Successful-case and reflection memory store.
  - Maintains semantic indexes and writes JSONL artifacts.

- `src/semantic_retrieval.py`
  - Local embedding model wrapper and cosine retrieval index.

- `src/evaluation.py`
  - Held-out diagnosis evaluation and memory-aware retrieval for eval prompts.

- `src/examination.py`
  - Text-only examination-selection task.
  - Loads the examination KB/eval set, builds exam prompts, stores separate exam memories, and runs held-out exam evaluation.

- `src/prompts.py`
  - Patient generation, doctor, judge, QC, and eval prompt builders.

- `data/infectious_diseases_seal_expert_rcl_kb_v6_compact.json`
  - Compact infectious-disease diagnosis KB.

- `data/expert_curriculum_sampler_config_v1.json`
  - Lane-weighted sampler config for training condition selection.

- `evaluation/id_expert_hard_eval_200.json`
  - Held-out 200-question diagnosis eval set.

- `data/id_examination_extension_kb_v1.json`
  - Examination-selection KB modules.

- `evaluation/id_expert_examination_eval_200.json`
  - Held-out 200-item examination-selection eval set.

## Setup

From the repo root:

```bash
bash scripts/setup_venv.sh
```

This creates `.venv`, installs `requirements-hf.txt`, and creates local Hugging Face cache directories under `hf_cache/`.

The default requirements pin a recent Transformers stack for Qwen3.6 support.

## Run The Default Benchmark

From the repo root:

```bash
bash scripts/run_qwen36_27b_baseline_and_train.sh
```

Despite the historical filename, the script now defaults to:

```bash
Qwen/Qwen3.6-35B-A3B
```

The script runs:

```bash
python run_benchmark.py ...
```

This keeps Qwen loaded for the full baseline + training + milestone-eval workflow.

## Useful Overrides

Run a shorter smoke experiment:

```bash
SUCCESS_CASES=100 EVAL_SUCCESS_MILESTONES=50,100 \
bash scripts/run_qwen36_27b_baseline_and_train.sh
```

Switch back to the dense 27B model:

```bash
MODEL_NAME=Qwen/Qwen3.6-27B \
bash scripts/run_qwen36_27b_baseline_and_train.sh
```

Use legacy tag retrieval instead of semantic RAG:

```bash
RETRIEVAL_MODE=tag \
bash scripts/run_qwen36_27b_baseline_and_train.sh
```

Change the embedding model:

```bash
EMBEDDING_MODEL=pritamdeka/S-PubMedBert-MS-MARCO \
bash scripts/run_qwen36_27b_baseline_and_train.sh
```

By default, embeddings run on CPU so they do not consume Qwen GPU memory:

```bash
EMBEDDING_DEVICE=cpu
```

## Persistent Current Run / Resume

By default, training writes to a new timestamped directory under `runs/`. For a server job with a wall-time limit, opt into a stable resumable directory:

```bash
CURRENT_RUN=1 bash scripts/run_qwen36_27b_baseline_and_train.sh
```

This writes training artifacts under:

```text
runs/current_run/
```

If the job is killed, rerun with the same target count. The loop reloads the existing JSONL memory files, rebuilds the in-memory semantic indexes, restores progress counters, and continues until the requested total number of successful cases is reached:

```bash
CURRENT_RUN=1 SKIP_BASELINE=1 bash scripts/run_qwen36_27b_baseline_and_train.sh
```

Use `SKIP_BASELINE=1` when resuming so the 200-question baseline evaluation is not repeated. The stable directory name can be changed with `CURRENT_RUN_DIR=my_run`.

The direct Python entry points expose the same option:

```bash
python run_simulation.py ... --current-run
python run_benchmark.py ... --current-run --skip-baseline
```

## Run Examination Selection

Use the same single-process benchmark path so Qwen is loaded once for baseline eval, training, and milestone evals:

```bash
python run_benchmark.py \
  --task-type examination_selection \
  --backend hf \
  --model Qwen/Qwen3.6-35B-A3B \
  --exam-kb-path data/id_examination_extension_kb_v1.json \
  --eval-file evaluation/id_expert_examination_eval_200.json \
  --baseline-run-name qwen36_35b_a3b_exam_baseline_eval \
  --train-run-name qwen36_35b_a3b_exam_20k_semantic_rag_eval_milestones \
  --target-successful-cases 20000 \
  --eval-success-milestones 5000,10000,15000,20000
```

Or through the shell wrapper:

```bash
TASK_TYPE=examination_selection SUCCESS_CASES=100 EVAL_SUCCESS_MILESTONES=50,100 \
bash scripts/run_qwen36_27b_baseline_and_train.sh
```

Examination memories are task-specific and are written as:

- `examination_successful_cases.jsonl`
- `examination_validated_reflections.jsonl`
- `examination_discard_summary.jsonl`

Diagnosis and examination artifacts both include `task_type`, so analysis can split curves, eval rows, memory growth, and reflection counts by task.

## Prompt Smoke Test

To save prompt/output pairs for a small local Ollama smoke:

```bash
python scripts/run_ollama_prompt_smoke.py --model qwen3.5:4b --per-task 3
```

This writes under the repo root at `smoke_tests/ollama_prompt_smoke_*/` and captures three diagnosis cases plus three examination-selection cases. For each usable case it saves generation, QC, doctor-answer, and hidden-judge prompt/output files. If QC marks a generated case unusable or malformed, downstream doctor/judge stages are skipped to match the real loop. Add `--continue-on-qc-fail` if you want to capture downstream prompts anyway, or `--include-eval` if you also want held-out eval prompt/output samples.

## HPC Notes

The repository script is not a Slurm submission script. The intended HPC pattern is:

1. Write a Slurm wrapper that requests the required GPUs, CPU, memory, and wall time.
2. Load the cluster modules/environment needed for Python/CUDA.
3. `cd` to the repo root.
4. Run:

```bash
bash scripts/run_qwen36_27b_baseline_and_train.sh
```

The run is local and contained:

- no OpenAI API
- no vLLM/SGLang server
- no external model-serving process
- Hugging Face Transformers loads the model directly
- model/cache paths are local to the repo unless overridden

## Outputs

Runs are written under `runs/`.

Baseline evaluation writes to:

```text
runs/qwen36_35b_a3b_baseline_eval/
```

Training creates a timestamped directory under `runs/`, for example:

```text
runs/YYYYMMDD_HHMMSS_qwen36_35b_a3b_20k_semantic_rag_eval_milestones/
```

Inside a training run:

- `successful_cases.jsonl`
  - saved successful diagnosis cases

- `validated_reflections.jsonl`
  - saved reflections that rescued a failed first attempt on retry

- `discard_summary.jsonl`
  - QC failures, malformed judge outputs, unrecovered retries, and empty generations

- `eval_summaries.jsonl`
  - periodic/milestone loop summaries

- `heldout_eval_summaries.jsonl`
  - held-out eval summaries across checkpoints

- `final_summary.json`
  - final training metrics

Milestone held-out evals are saved in subdirectories:

```text
eval_success_005000/
eval_success_010000/
eval_success_015000/
eval_success_020000/
```

Each contains:

- `eval_results.jsonl`
- `eval_summary.json`
- `eval_metadata.json`

## How To Explain This To Another GPT

Use this summary:

> We are building a local infectious-disease doctor-agent evolution benchmark. The main path is diagnosis: Qwen3.6 runs through Hugging Face Transformers, generates synthetic patient cases from a compact infectious-disease KB, answers with a diagnosis, and is judged by a hidden judge prompt. Correct cases are stored in successful-case memory. Failed first attempts can generate a reflection; the same case is retried, and the reflection is saved only if the retry succeeds. Semantic RAG retrieves the top 3 similar successful cases and top 4 validated reflections using a local PubMedBERT MS-MARCO embedding model. Diagnosis is benchmarked at baseline, 5k, 10k, 15k, and 20k successful cases. A separate text-only examination-selection task now exists for deciding what to examine, ask, collect, or test next, using its own KB, held-out 200-item eval set, and separate memory files.

## Current Limitations

- Examination selection is standalone only; it is not yet staged into diagnosis.
- No treatment task yet.
- Semantic RAG currently uses in-memory indexes, not persisted vector databases.
- There is no helpfulness judge for retrieved experiences.
- Patient generation, doctor answering, and judging all use the same backend model.
- The benchmark is optimized for a single long run rather than distributed training.

## Verification Commands

Basic syntax checks:

```bash
python3 -m compileall run_benchmark.py run_simulation.py run_evaluation.py src
bash -n scripts/run_qwen36_27b_baseline_and_train.sh
```

Show benchmark CLI:

```bash
python run_benchmark.py --help
```
