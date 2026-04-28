# Codex Spec: Add Text-Only Examination-Selection Task to SEAL_AGENTS

## Goal

Extend the current disease-diagnosis-only infectious-disease SEAL/RCL experiment with a minimal **text-only examination-selection task**.

Do **not** build a game-like backend, spatial simulator, hospital map, agent movement system, registration workflow, pharmacy, or treatment/follow-up simulator.

The new abstraction is:

```text
condition_id = hidden clinical anchor
condition_name = hidden clinical condition/syndrome
task_type = expert skill being trained/evaluated
```

Examples:

```text
condition_name = Malaria
task_type = examination_selection
```

```text
condition_name = Malaria
task_type = diagnosis
```

Later, `staged_case` can compose examination-selection followed by diagnosis, but this spec only requires `examination_selection` as a standalone text task.

---

## Files to Add

Copy the two new files into the repo:

```text
data/id_examination_extension_kb_v1.json
evaluation/id_expert_examination_eval_200.json
```

Source files supplied by the user:

```text
/mnt/data/id_examination_extension_kb_v1.json
/mnt/data/id_expert_examination_eval_200.json
```

The existing diagnosis KB remains the shared condition catalogue:

```text
data/infectious_diseases_seal_expert_rcl_kb_v6_compact.json
```

The examination KB extends it; do not duplicate or rewrite the diagnosis KB.

---

## High-Level Design

Keep the existing loop shape:

```text
sample hidden condition
produce visible case
QC visible case
retrieve memory
ask doctor model
judge answer
if correct: save successful case
if wrong: generate reflection, retry same case once, save reflection only if retry succeeds
```

Make `task_type` a first-class field. The loop should dispatch task-specific behaviour through a small task adapter/registry rather than hardcoding diagnosis-only logic.

Required task types:

```text
diagnosis
examination_selection
```

Optional future task type, not required now:

```text
staged_case
```

---

## Task Registry

Add a task registry, for example:

```python
TASK_REGISTRY = {
    "diagnosis": DiagnosisTask(...),
    "examination_selection": ExaminationSelectionTask(...),
}
```

A task adapter should own these behaviours:

```text
load task-specific KB/eval data
sample eligible condition/module
generate visible episode/case
build QC prompt
build doctor prompt
build judge prompt
build retry prompt
parse judge result
format successful memory record
format validated reflection record
run held-out eval item
```

Keep the existing diagnosis code working. If full refactor is risky, wrap the current diagnosis path as `DiagnosisTask` with minimal changes and implement `ExaminationSelectionTask` alongside it.

---

## Examination KB Format

The new KB file has this top-level shape:

```json
{
  "metadata": {...},
  "runtime_design": {...},
  "condition_modules": [
    {
      "condition_id": "inf_0018_sepsis",
      "condition_name": "Sepsis",
      "task_type": "examination_selection",
      "task_inclusion": "include",
      "difficulty_hint": "hard",
      "curriculum_lanes": [...],
      "exam_focus": [...],
      "visible_case_stage": "presenting_case_before_results",
      "case_generation_targets": [...],
      "patient_qc": {...},
      "core_exam_or_history": [...],
      "essential_examination_or_tests": [...],
      "accepted_alternatives": [...],
      "conditional_or_second_line_tests": [...],
      "avoid_or_low_value": [...],
      "dangerous_misses": [...],
      "judge": {...},
      "retrieval_tags": [...]
    }
  ]
}
```

Load only modules where:

```text
task_type == examination_selection
task_inclusion == include
```

Do not show `condition_name`, hidden condition metadata, gold tests, dangerous misses, or judge rules to the doctor model.

---

## Examination Evaluation File Format

The new held-out eval file has this top-level shape:

```json
{
  "metadata": {...},
  "items": [
    {
      "question_id": "id_exam_0001",
      "task_type": "examination_selection",
      "condition_id": "inf_0018_sepsis",
      "hidden_condition": "Sepsis",
      "difficulty": "hard",
      "curriculum_lanes": [...],
      "exam_focus": [...],
      "presenting_case": "...",
      "question": "What should an infectious diseases expert examine, ask, collect, or test next?...",
      "gold_essential": [...],
      "gold_focused_exam_or_history": [...],
      "accepted_alternatives_or_second_line": [...],
      "dangerous_misses": [...],
      "low_value_or_wrong": [...],
      "judge_rule": {...},
      "retrieval_tags": [...]
    }
  ]
}
```

There should be 200 eval items.

During evaluation, show the doctor model only:

```text
presenting_case
question
retrieved examination memories, if memory retrieval is enabled
```

Never show:

```text
hidden_condition
gold_essential
gold_focused_exam_or_history
accepted_alternatives_or_second_line
dangerous_misses
low_value_or_wrong
judge_rule
```

Those fields are for judging and analysis only.

---

## Episode Object

Create or approximate this internal episode shape:

```json
{
  "episode_id": "ep_000001",
  "task_type": "examination_selection",
  "condition_id": "inf_0018_sepsis",
  "condition_name": "Sepsis",
  "visible_case_stage": "presenting_case_before_results",
  "visible_case": "...",
  "hidden_gold": {
    "core_exam_or_history": [],
    "essential_examination_or_tests": [],
    "accepted_alternatives": [],
    "conditional_or_second_line_tests": [],
    "avoid_or_low_value": [],
    "dangerous_misses": []
  },
  "retrieval_tags": [],
  "doctor_answer": null,
  "judge_result": null
}
```

For generated training episodes, `hidden_gold` comes from the examination KB module.

For held-out evaluation episodes, `hidden_gold` comes from the eval item.

---

## Examination Patient/Case Generation

For training, generate a presenting case from an examination KB module.

The visible case must be **before definitive results**.

Prompt requirements:

```text
Generate a realistic infectious-disease presenting case.
The doctor should decide what to examine, ask, collect, or test next.
Do not reveal the condition name, pathogen name, condition ID, definitive diagnostic result, or gold test answer.
Include enough syndrome, exposure, host-context, severity, and time-course clues to justify the expected examination/testing strategy.
Do not include treatment decisions.
```

The generated case should usually be concise: 3-7 sentences.

The case should support the KB fields:

```text
core_exam_or_history
essential_examination_or_tests
conditional_or_second_line_tests
dangerous_misses
avoid_or_low_value
```

---

## Examination QC

Add a binary QC prompt for generated training cases.

QC output must be JSON:

```json
{
  "usable": 1,
  "reason": "short reason"
}
```

QC should fail if:

```text
definitive diagnosis, pathogen, or exact diagnostic result is leaked
case already contains the result of the expected test
case is too vague to choose a focused exam/testing strategy
case contradicts the hidden module's host, exposure, time-course, or severity clues
case requires treatment planning rather than examination/test selection
```

Discard failed cases. Do not save failed patient text to long-term memory.

---

## Examination Doctor Prompt

Add a new prompt builder for `task_type == examination_selection`.

The prompt should ask:

```text
You are an infectious diseases expert.
Given this presenting case before definitive results, answer:
What should you examine, ask, collect, or test next?
Give a focused examination/testing plan only. Do not provide treatment.
```

Required output headings:

```text
1. Immediate safety/context
2. Focused examination/history
3. Initial tests/specimens
4. Do not miss / avoid
```

Keep this concise. The model should not produce a long management plan.

The doctor prompt may include retrieved examination successful cases/reflections, but should not include diagnosis-task memory unless explicitly enabled in a later experiment.

---

## Examination Judge Prompt

Add a new judge prompt for `task_type == examination_selection`.

The judge sees:

```text
hidden condition/module metadata
visible presenting case
doctor answer
core_exam_or_history
essential_examination_or_tests
accepted_alternatives
conditional_or_second_line_tests
avoid_or_low_value
dangerous_misses
judge.correct_if / judge.incorrect_if
```

The judge returns JSON only:

```json
{
  "correct": 1,
  "essential_hits": ["..."],
  "dangerous_misses": [],
  "reflection": null
}
```

or:

```json
{
  "correct": 0,
  "essential_hits": ["..."],
  "dangerous_misses": ["..."],
  "reflection": "One concise examination-selection lesson that would help solve the same case on retry."
}
```

Judge as correct when the answer:

```text
captures the syndrome-defining examination/specimen/test family
includes urgent safety assessment where relevant: sepsis, shock, airway, CNS, ocular, pregnancy, immunocompromise, limb-threatening infection
uses exposure, travel, sexual, device, immune, pregnancy, or healthcare context when central
is specific to the visible case stage
avoids relying mainly on low-value broad panels or nonspecific inflammatory markers
```

Judge as incorrect when the answer:

```text
gives only diagnosis or treatment
omits the central specimen/test family
misses a dangerous safety or specimen-timing issue
uses low-value testing as the main next step
ignores host/exposure context that determines the correct test strategy
```

---

## Reflection Retry

Use the same retry policy as diagnosis:

```text
if initial answer correct: save successful examination case
if initial answer wrong: judge provides one reflection
retry same case once with the reflection
if retry correct: save successful case and validated reflection
if retry wrong: discard patient text and unvalidated reflection
```

Do not save unvalidated reflections.

---

## Memory

For the first implementation, prefer separate examination memory files because this avoids cross-task contamination:

```text
examination_successful_cases.jsonl
examination_validated_reflections.jsonl
examination_discard_summary.jsonl
```

Each examination successful case should include:

```json
{
  "successful_case_id": "...",
  "attempted_patient_id": "...",
  "task_type": "examination_selection",
  "condition_id": "...",
  "condition_name": "...",
  "training_lanes": [],
  "exam_focus": [],
  "patient_case": "...",
  "doctor_answer": "...",
  "judge_result": {...},
  "solved_on": "first_attempt|retry",
  "validated_reflection_used": null,
  "expert_lesson": "...",
  "retrieval_tags": []
}
```

Each validated reflection should include:

```json
{
  "reflection_id": "...",
  "successful_case_id": "...",
  "attempted_patient_id": "...",
  "task_type": "examination_selection",
  "condition_id": "...",
  "condition_name": "...",
  "failure_mode": "...",
  "reflection": "...",
  "validated_by": "same_patient_retry",
  "retrieval_tags": []
}
```

If using unified memory files instead, every record must include `task_type`, and retrieval must support filtering:

```python
memory.retrieve(query, task_type="examination_selection", top_successes=3, top_reflections=4)
```

For the initial experiment, examination prompts should retrieve only examination memories.

---

## CLI Changes

Add `--task-type` to training and evaluation entry points.

Required values:

```text
diagnosis
examination_selection
```

Add paths:

```text
--exam-kb-path data/id_examination_extension_kb_v1.json
--eval-file evaluation/id_expert_examination_eval_200.json
```

Examples, adjust exact existing argument names where necessary:

```bash
python run_evaluation.py \
  --task-type examination_selection \
  --exam-kb-path data/id_examination_extension_kb_v1.json \
  --eval-file evaluation/id_expert_examination_eval_200.json \
  --output-dir runs/exam_baseline_eval
```

```bash
python run_simulation.py \
  --task-type examination_selection \
  --exam-kb-path data/id_examination_extension_kb_v1.json \
  --target-successful-cases 100 \
  --output-dir runs/exam_smoke_100
```

Later, optional task mixing can be added:

```text
--task-mix diagnosis:0.7,examination_selection:0.3
```

Do not implement task mixing unless the single-task examination path works.

---

## Metrics

For examination-selection runs, log at minimum:

```text
attempted_patients
qc_discards
first_attempt_successes
retry_successes
discarded_unrecovered_cases
first_attempt_success_rate
retry_rescue_rate
coverage_by_condition
coverage_by_curriculum_lane
coverage_by_exam_focus
```

For held-out examination eval, log:

```text
accuracy
item_count
correct_count
incorrect_count
essential_hit_rate, if parsed from judge
dangerous_miss_rate, if parsed from judge
low_value_or_wrong_test_rate, if parsed from judge
```

Do not require these extra analysis rates for the loop to run; binary accuracy is enough for the first implementation.

---

## Integration Order

1. Copy the two new JSON files into `data/` and `evaluation/`.
2. Add a loader for `id_examination_extension_kb_v1.json`.
3. Add `task_type` support to records and run metadata.
4. Add `ExaminationSelectionTask` prompt builders.
5. Add examination QC prompt.
6. Add examination judge prompt.
7. Add task-filtered or separate examination memory stores.
8. Add `--task-type examination_selection` to `run_evaluation.py`.
9. Add `--task-type examination_selection` to `run_simulation.py`.
10. Run syntax checks.
11. Run a no-memory examination baseline eval.
12. Run a tiny examination training smoke test, e.g. 10-100 successful cases.
13. Run examination held-out eval with examination memory enabled.
14. Confirm diagnosis-only path still runs unchanged.

---

## Acceptance Criteria

The implementation is acceptable when:

```text
python3 -m compileall run_benchmark.py run_simulation.py run_evaluation.py src
```

passes, and:

```text
bash -n scripts/run_qwen36_27b_baseline_and_train.sh
```

passes.

The diagnosis-only benchmark path must still work without requiring the examination files.

The new evaluation path must load exactly 200 examination eval items from:

```text
evaluation/id_expert_examination_eval_200.json
```

The doctor model must never receive hidden fields from the evaluation file.

A smoke examination run must produce some or all of:

```text
examination_successful_cases.jsonl
examination_validated_reflections.jsonl
examination_discard_summary.jsonl
heldout_eval_summaries.jsonl
final_summary.json
```

Every saved examination memory record must include:

```text
task_type == examination_selection
condition_id
condition_name
patient_case
doctor_answer
retrieval_tags
```

Judge outputs must be robustly parsed. Malformed judge output should be recorded as a discard or evaluation failure, not crash the full run.

---

## Guardrails

Do not:

```text
build a game backend
add spatial movement or map simulation
add treatment planning to this task
show hidden condition or gold answers to the doctor model
save failed patient text or unvalidated reflections into long-term memory
mix diagnosis and examination memory by default
require external APIs
change Hugging Face cache assumptions
require commands to run outside repo root
```

Do:

```text
keep the task text-only
keep diagnosis path intact
make task_type explicit
keep first implementation single-task before mixed-task experiments
use binary correctness for the loop
save richer judge metadata only for analysis
```

---

## Future Extension: Staged Case

Do not implement yet, but design should not block it.

Future staged case shape:

```text
Stage 1: presenting case -> doctor selects examinations/tests
Stage 2: environment reveals appropriate selected test results
Stage 3: doctor gives diagnosis
Stage 4: judge examination selection and diagnosis
```

This is still text-only. It is a composed task:

```text
staged_case = examination_selection + result_reveal + diagnosis
```

No game backend is needed.
