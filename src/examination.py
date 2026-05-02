from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation import EVAL_MODES
from src.semantic_retrieval import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    RETRIEVAL_MODES,
    LocalTextEmbedder,
    SemanticRecordIndex,
    embedding_config_from_env,
)
from src.utils import (
    append_jsonl,
    coerce_binary_flag,
    extract_json_object,
    iso_now,
    load_json,
    safe_rate,
    slugify,
    unique_strings,
    write_json,
)

EXAMINATION_TASK_TYPE = "examination_selection"
MAX_ATTEMPT_MULTIPLIER = 50


@dataclass(frozen=True)
class ExaminationJudgeResult:
    correct: bool
    reflection: str | None
    raw_output: str
    essential_hits: list[str]
    dangerous_misses: list[str]
    parse_error: str | None = None


@dataclass(frozen=True)
class ExaminationQCResult:
    usable: bool
    reason: str | None
    raw_output: str | None = None
    parse_error: str | None = None


@dataclass(frozen=True)
class ExaminationLoopConfig:
    backend: str
    model: str
    exam_kb_path: str
    n_successful_cases: int
    run_name: str
    eval_every: int = 0
    run_evaluation: bool = False
    eval_file: str | None = None
    eval_mode: str | None = None
    eval_limit: int | None = None
    eval_success_milestones: tuple[int, ...] = ()
    log_every: int = 10
    verbose_events: bool = False
    quiet: bool = False
    temperature: float = 0.2
    max_tokens: int = 768
    seed: int = 17
    n_success_memory: int = 3
    n_reflection_memory: int = 4
    retrieval_mode: str = "semantic"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    runs_root: str = "runs"
    output_dir: str | None = None
    resume_existing: bool = False


@dataclass(frozen=True)
class ExaminationEvalConfig:
    eval_file: str
    exam_kb_path: str
    run_name: str
    backend: str
    model: str
    successful_cases: str | None = None
    validated_reflections: str | None = None
    eval_mode: str | None = None
    n_success_memory: int = 3
    n_reflection_memory: int = 4
    retrieval_mode: str = "semantic"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    temperature: float = 0.0
    max_tokens: int = 512
    eval_limit: int | None = None
    runs_root: str = "runs"
    output_dir: str | None = None
    quiet: bool = False


def load_examination_modules(path: str) -> list[dict[str, Any]]:
    payload = load_json(path)
    modules = payload.get("condition_modules")
    if not isinstance(modules, list):
        raise ValueError(f"Examination KB is missing condition_modules: {path}")

    filtered: list[dict[str, Any]] = []
    for module in modules:
        if not isinstance(module, dict):
            continue
        if module.get("task_type") != EXAMINATION_TASK_TYPE:
            continue
        if module.get("task_inclusion") != "include":
            continue
        normalized = dict(module)
        normalized["curriculum_lanes"] = unique_strings(module.get("curriculum_lanes", []))
        normalized["exam_focus"] = unique_strings(module.get("exam_focus", []))
        normalized["retrieval_tags"] = unique_strings(module.get("retrieval_tags", []))
        filtered.append(normalized)
    if not filtered:
        raise RuntimeError(f"No included examination modules found: {path}")
    return filtered


def load_examination_eval_items(path: str) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict):
        items = payload.get("items")
    else:
        items = payload
    if not isinstance(items, list):
        raise ValueError(f"Examination eval file is missing items: {path}")
    records = [
        dict(item)
        for item in items
        if isinstance(item, dict) and item.get("task_type") == EXAMINATION_TASK_TYPE
    ]
    if len(records) != 200:
        raise ValueError(
            f"Expected exactly 200 examination eval items, found {len(records)} in {path}"
        )
    return records


def exam_case_retrieval_text(record: dict[str, Any]) -> str:
    return _first_text(
        record.get("retrieval_text"),
        record.get("patient_case"),
        record.get("presenting_case"),
        record.get("question"),
        record.get("condition_name"),
    )


def exam_reflection_retrieval_text(record: dict[str, Any]) -> str:
    return _first_text(
        record.get("retrieval_text"),
        record.get("patient_case"),
        record.get("presenting_case"),
        "\n".join(
            str(value)
            for value in (
                record.get("condition_name"),
                record.get("failure_mode"),
                record.get("reflection"),
            )
            if value
        ),
    )


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _as_list(value: Any) -> list[str]:
    return unique_strings(value if isinstance(value, list) else [])


def _exam_hidden_gold(module: dict[str, Any]) -> dict[str, Any]:
    return {
        "core_exam_or_history": _as_list(module.get("core_exam_or_history")),
        "essential_examination_or_tests": _as_list(
            module.get("essential_examination_or_tests")
        ),
        "accepted_alternatives": _as_list(module.get("accepted_alternatives")),
        "conditional_or_second_line_tests": _as_list(
            module.get("conditional_or_second_line_tests")
        ),
        "avoid_or_low_value": _as_list(module.get("avoid_or_low_value")),
        "dangerous_misses": _as_list(module.get("dangerous_misses")),
    }


def _exam_eval_hidden_gold(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "core_exam_or_history": _as_list(item.get("gold_focused_exam_or_history")),
        "essential_examination_or_tests": _as_list(item.get("gold_essential")),
        "accepted_alternatives": _as_list(
            item.get("accepted_alternatives_or_second_line")
        ),
        "conditional_or_second_line_tests": [],
        "avoid_or_low_value": _as_list(item.get("low_value_or_wrong")),
        "dangerous_misses": _as_list(item.get("dangerous_misses")),
    }


def build_examination_case_generation_prompt(module: dict[str, Any]) -> str:
    visible_payload = {
        "task_type": EXAMINATION_TASK_TYPE,
        "condition_id": module.get("condition_id"),
        "condition_name": module.get("condition_name"),
        "difficulty_hint": module.get("difficulty_hint"),
        "curriculum_lanes": module.get("curriculum_lanes", []),
        "exam_focus": module.get("exam_focus", []),
        "visible_case_stage": module.get("visible_case_stage"),
        "case_generation_targets": module.get("case_generation_targets", []),
        "core_exam_or_history": module.get("core_exam_or_history", []),
        "essential_examination_or_tests": module.get(
            "essential_examination_or_tests",
            [],
        ),
        "conditional_or_second_line_tests": module.get(
            "conditional_or_second_line_tests",
            [],
        ),
        "dangerous_misses": module.get("dangerous_misses", []),
        "avoid_or_low_value": module.get("avoid_or_low_value", []),
    }
    return f"""You are generating a synthetic infectious-disease presenting case for an examination-selection simulation.

Hidden examination module:
{_pretty_json(visible_payload)}

Generate a realistic presenting case before definitive examination or test results.

Rules:
- Do not reveal the condition name, pathogen name, condition ID, definitive diagnostic result, or gold test answer.
- Include enough syndrome, exposure, host-context, severity, and time-course clues to justify the expected examination/testing strategy.
- Basic vital signs, physical findings, and nonspecific triage results may be included if useful.
- If including labs or imaging, keep them nonspecific and non-definitive. Do not include positive/negative results from the expected diagnostic specimen/test family.
- Write this as a patient/chart vignette only. Do not add meta-instructions such as "you must select tests" or "further diagnostic evaluation is required."
- Do not include treatment decisions.
- Do not include the result of the expected diagnostic test or specimen.
- Keep the case focused and information-dense; avoid unnecessary narrative filler.
- Return plain text only.
"""


def build_examination_qc_prompt(module: dict[str, Any], patient_case: str) -> str:
    qc_payload = {
        "condition_id": module.get("condition_id"),
        "condition_name": module.get("condition_name"),
        "exam_focus": module.get("exam_focus", []),
        "patient_qc": module.get("patient_qc", {}),
        "expected_exam_or_history": module.get("core_exam_or_history", []),
        "expected_tests": module.get("essential_examination_or_tests", []),
    }
    return f"""You are doing binary quality control for an infectious-disease examination-selection case.

Hidden examination module:
{_pretty_json(qc_payload)}

Generated presenting case:
{patient_case}

Mark unusable if:
- definitive diagnosis, pathogen, or exact diagnostic result is leaked
- the case already contains the result of the expected test
- the case is too vague to choose a focused examination/testing strategy
- the case contradicts host, exposure, time-course, or severity clues
- the case requires treatment planning rather than examination/test selection
- the case is written as instructions to the doctor rather than a patient/chart vignette

Do not mark unusable merely because the case includes basic vitals, physical exam,
or nonspecific triage labs/imaging. These are allowed if the focused examination,
specimen, or test decision still remains. Mark unusable only when the hidden
condition/pathogen is named, the gold diagnostic test/specimen result is already
given, or the next testing decision is no longer meaningful.

Return JSON only:
{{
  "usable": 1 or 0,
  "reason": null or "short reason"
}}
"""


def build_examination_doctor_prompt(
    memory_context: str,
    patient_case: str,
    question: str | None = None,
    previous_answer: str | None = None,
    provisional_reflection: str | None = None,
) -> str:
    retry_block = ""
    if previous_answer and provisional_reflection:
        retry_block = f"""
Previous answer for this same presenting case:
{previous_answer}

Validated direction to apply on retry:
{provisional_reflection}

Use this reflection to revise the focused examination/testing plan.
"""

    task_question = question or (
        "What should you examine, ask, collect, or test next?"
    )
    return f"""You are an infectious diseases expert.

Given this presenting case before definitive results, answer:
{task_question}

Retrieved successful examination cases and validated examination reflections:
{memory_context}

Presenting case:
{patient_case}
{retry_block}

Give a focused examination/testing plan only. Do not provide treatment.

Return using exactly these headings:

1. Immediate safety/context
2. Focused examination/history
3. Initial tests/specimens
4. Do not miss / avoid

Rules:
- Be concise.
- Do not provide treatment, drug doses, or a management plan.
- Do not give a final diagnosis as the main answer.
- Focus on what to examine, ask, collect, or test next from this visible case stage.
"""


def build_examination_judge_prompt(
    module_or_item: dict[str, Any],
    patient_case: str,
    doctor_answer: str,
    hidden_gold: dict[str, Any],
) -> str:
    judge_payload = {
        "task_type": EXAMINATION_TASK_TYPE,
        "condition_id": module_or_item.get("condition_id"),
        "condition_name": module_or_item.get("condition_name")
        or module_or_item.get("hidden_condition"),
        "difficulty": module_or_item.get("difficulty")
        or module_or_item.get("difficulty_hint"),
        "curriculum_lanes": module_or_item.get("curriculum_lanes", []),
        "exam_focus": module_or_item.get("exam_focus", []),
        "visible_case_stage": module_or_item.get(
            "visible_case_stage",
            "presenting_case_before_results",
        ),
        "core_exam_or_history": hidden_gold.get("core_exam_or_history", []),
        "essential_examination_or_tests": hidden_gold.get(
            "essential_examination_or_tests",
            [],
        ),
        "accepted_alternatives": hidden_gold.get("accepted_alternatives", []),
        "conditional_or_second_line_tests": hidden_gold.get(
            "conditional_or_second_line_tests",
            [],
        ),
        "avoid_or_low_value": hidden_gold.get("avoid_or_low_value", []),
        "dangerous_misses": hidden_gold.get("dangerous_misses", []),
        "judge": module_or_item.get("judge") or module_or_item.get("judge_rule", {}),
        "notes_for_hidden_judge": module_or_item.get("notes_for_hidden_judge"),
    }
    return f"""You are the hidden judge for an infectious-disease examination-selection simulation.

The judge is binary and evaluates examination/history/specimen/test selection only.

Hidden examination module:
{_pretty_json(judge_payload)}

Visible presenting case:
{patient_case}

Doctor answer:
{doctor_answer}

Mark correct when the answer:
- captures the syndrome-defining examination/specimen/test family
- includes urgent safety assessment when relevant: sepsis, shock, airway, CNS, ocular, pregnancy, immunocompromise, or limb-threatening infection
- uses exposure, travel, sexual, device, immune, pregnancy, or healthcare context when central
- is specific to the visible case stage
- avoids relying mainly on low-value broad panels or nonspecific inflammatory markers

Mark incorrect when the answer:
- gives only diagnosis or treatment
- omits the central specimen/test family
- misses a dangerous safety or specimen-timing issue
- uses low-value testing as the main next step
- ignores host/exposure context that determines the test strategy

Do not include hidden chain-of-thought.

Return JSON only:
{{
  "correct": 1 or 0,
  "essential_hits": ["matched essential exam/test concepts"],
  "dangerous_misses": ["missed dangerous concepts"],
  "reflection": null if correct, otherwise "one concise examination-selection lesson"
}}
"""


def _pretty_json(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True)


def parse_examination_judge_response(response: str) -> ExaminationJudgeResult:
    try:
        payload = extract_json_object(response)
        correct = bool(coerce_binary_flag(payload.get("correct"), "correct"))
    except (TypeError, ValueError) as exc:
        return ExaminationJudgeResult(
            correct=False,
            reflection=None,
            raw_output=response,
            essential_hits=[],
            dangerous_misses=[],
            parse_error=str(exc),
        )

    reflection = payload.get("reflection")
    if correct:
        reflection = None
    elif not isinstance(reflection, str) or not reflection.strip():
        reflection = None
    else:
        reflection = reflection.strip()
    return ExaminationJudgeResult(
        correct=correct,
        reflection=reflection,
        raw_output=response,
        essential_hits=_as_list(payload.get("essential_hits")),
        dangerous_misses=_as_list(payload.get("dangerous_misses")),
    )


def parse_examination_qc_response(response: str) -> ExaminationQCResult:
    try:
        payload = extract_json_object(response)
        usable = bool(coerce_binary_flag(payload.get("usable"), "usable"))
    except (TypeError, ValueError) as exc:
        return ExaminationQCResult(
            usable=False,
            reason="malformed_examination_qc_output",
            raw_output=response,
            parse_error=str(exc),
        )
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = None
    return ExaminationQCResult(usable=usable, reason=reason, raw_output=response)


class ExaminationMemoryStore:
    def __init__(
        self,
        runs_root: str,
        run_name: str,
        config_snapshot: dict[str, Any],
        retrieval_mode: str,
        embedding_model: str,
        embedding_device: str,
        embedding_batch_size: int,
        output_dir: str | None = None,
        resume_existing: bool = False,
    ) -> None:
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")
        self.retrieval_mode = retrieval_mode
        self.run_dir = (
            Path(output_dir)
            if output_dir
            else Path(runs_root) / f"{_now_timestamp()}_{slugify(run_name)}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=resume_existing or bool(output_dir))

        self.success_path = self.run_dir / "examination_successful_cases.jsonl"
        self.reflection_path = self.run_dir / "examination_validated_reflections.jsonl"
        self.discard_path = self.run_dir / "examination_discard_summary.jsonl"
        self.eval_summary_path = self.run_dir / "eval_summaries.jsonl"
        self.heldout_eval_summary_path = self.run_dir / "heldout_eval_summaries.jsonl"
        self.final_summary_path = self.run_dir / "final_summary.json"
        self.metadata_path = self.run_dir / "run_metadata.json"
        if output_dir and not resume_existing:
            existing_outputs = [
                path
                for path in (
                    self.success_path,
                    self.reflection_path,
                    self.discard_path,
                    self.eval_summary_path,
                    self.heldout_eval_summary_path,
                    self.final_summary_path,
                    self.metadata_path,
                )
                if path.exists()
            ]
            if existing_outputs:
                existing_list = ", ".join(str(path) for path in existing_outputs)
                raise FileExistsError(
                    f"Output directory already contains examination run files: {existing_list}"
                )

        self.successful_cases: list[dict[str, Any]] = []
        self.validated_reflections: list[dict[str, Any]] = []
        self.discards: list[dict[str, Any]] = []
        self.eval_summaries: list[dict[str, Any]] = []
        self.heldout_eval_summaries: list[dict[str, Any]] = []
        self._success_counter = 0
        self._reflection_counter = 0
        self._success_counts_by_condition: dict[str, int] = {}

        self._success_index: SemanticRecordIndex | None = None
        self._reflection_index: SemanticRecordIndex | None = None
        if resume_existing:
            self._load_existing_records()

        if retrieval_mode == "semantic":
            embedding_config = embedding_config_from_env(
                model=embedding_model,
                device=embedding_device,
                batch_size=embedding_batch_size,
            )
            embedder = LocalTextEmbedder(embedding_config)
            self._success_index = SemanticRecordIndex(
                records=self.successful_cases,
                embedder=embedder,
                text_builder=exam_case_retrieval_text,
            )
            self._reflection_index = SemanticRecordIndex(
                records=self.validated_reflections,
                embedder=embedder,
                text_builder=exam_reflection_retrieval_text,
            )

        write_json(
            self.metadata_path,
            {
                **config_snapshot,
                "resume_existing": resume_existing,
                "resumed_successful_cases": len(self.successful_cases),
                "resumed_validated_reflections": len(self.validated_reflections),
                "resumed_discards": len(self.discards),
            },
        )

    def _load_existing_records(self) -> None:
        self.successful_cases = _load_jsonl_if_exists(self.success_path)
        self.validated_reflections = _load_jsonl_if_exists(self.reflection_path)
        self.discards = _load_jsonl_if_exists(self.discard_path)
        self.eval_summaries = _load_jsonl_if_exists(self.eval_summary_path)
        self.heldout_eval_summaries = _load_jsonl_if_exists(
            self.heldout_eval_summary_path
        )
        self._success_counter = _max_numeric_suffix(
            self.successful_cases,
            field="successful_case_id",
            fallback=len(self.successful_cases),
        )
        self._reflection_counter = _max_numeric_suffix(
            self.validated_reflections,
            field="reflection_id",
            fallback=len(self.validated_reflections),
        )
        for record in self.successful_cases:
            condition_id = record.get("condition_id")
            if isinstance(condition_id, str) and condition_id:
                self._success_counts_by_condition[condition_id] = (
                    self._success_counts_by_condition.get(condition_id, 0) + 1
                )

    @property
    def success_index(self) -> SemanticRecordIndex | None:
        return self._success_index

    @property
    def reflection_index(self) -> SemanticRecordIndex | None:
        return self._reflection_index

    def success_counts_by_condition(self) -> dict[str, int]:
        return dict(self._success_counts_by_condition)

    def record_success(
        self,
        attempted_patient_id: str,
        module: dict[str, Any],
        patient_case: str,
        doctor_answer: str,
        judge_result: ExaminationJudgeResult,
        solved_on: str,
        validated_reflection_used: str | None,
    ) -> dict[str, Any]:
        self._success_counter += 1
        condition_id = str(module["condition_id"])
        record = {
            "successful_case_id": f"exam_success_{self._success_counter:06d}",
            "attempted_patient_id": attempted_patient_id,
            "task_type": EXAMINATION_TASK_TYPE,
            "condition_id": condition_id,
            "condition_name": module["condition_name"],
            "training_lanes": _as_list(module.get("curriculum_lanes")),
            "exam_focus": _as_list(module.get("exam_focus")),
            "patient_case": patient_case,
            "retrieval_text": patient_case,
            "doctor_answer": doctor_answer,
            "judge_result": _judge_result_payload(judge_result),
            "solved_on": solved_on,
            "validated_reflection_used": validated_reflection_used,
            "expert_lesson": _build_exam_expert_lesson(module),
            "retrieval_tags": _as_list(module.get("retrieval_tags")),
        }
        self._success_counts_by_condition[condition_id] = (
            self._success_counts_by_condition.get(condition_id, 0) + 1
        )
        self.successful_cases.append(record)
        if self._success_index is not None:
            self._success_index.add(record)
        append_jsonl(self.success_path, record)
        return record

    def record_validated_reflection(
        self,
        successful_case_id: str,
        attempted_patient_id: str,
        module: dict[str, Any],
        patient_case: str,
        doctor_answer: str,
        reflection: str,
    ) -> dict[str, Any]:
        self._reflection_counter += 1
        record = {
            "reflection_id": f"exam_reflection_{self._reflection_counter:06d}",
            "successful_case_id": successful_case_id,
            "attempted_patient_id": attempted_patient_id,
            "task_type": EXAMINATION_TASK_TYPE,
            "condition_id": module["condition_id"],
            "condition_name": module["condition_name"],
            "training_lanes": _as_list(module.get("curriculum_lanes")),
            "exam_focus": _as_list(module.get("exam_focus")),
            "patient_case": patient_case,
            "doctor_answer": doctor_answer,
            "retrieval_text": patient_case,
            "failure_mode": _classify_exam_failure_mode(reflection),
            "reflection": reflection,
            "validated_by": "same_patient_retry",
            "retrieval_tags": _as_list(module.get("retrieval_tags")),
        }
        self.validated_reflections.append(record)
        if self._reflection_index is not None:
            self._reflection_index.add(record)
        append_jsonl(self.reflection_path, record)
        return record

    def record_discard(
        self,
        attempted_patient_id: str,
        module: dict[str, Any],
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "attempted_patient_id": attempted_patient_id,
            "task_type": EXAMINATION_TASK_TYPE,
            "condition_id": module["condition_id"],
            "condition_name": module["condition_name"],
            "training_lanes": _as_list(module.get("curriculum_lanes")),
            "exam_focus": _as_list(module.get("exam_focus")),
            "reason": reason,
        }
        if details:
            record["details"] = details
        self.discards.append(record)
        append_jsonl(self.discard_path, record)

    def build_memory_context(
        self,
        patient_case: str,
        module: dict[str, Any],
        n_success_memory: int,
        n_reflection_memory: int,
    ) -> str:
        if (
            self.retrieval_mode == "semantic"
            and patient_case.strip()
            and (self.successful_cases or self.validated_reflections)
            and self._success_index is not None
        ):
            query_embedding = self._success_index.embedder.encode([patient_case])
            success_records = (
                self._success_index.search_embedding(query_embedding, n_success_memory)
                if self.successful_cases
                else []
            )
            reflection_records = (
                self._reflection_index.search_embedding(query_embedding, n_reflection_memory)
                if self.validated_reflections and self._reflection_index is not None
                else []
            )
        else:
            success_records = self._tag_records(
                self.successful_cases,
                _as_list(module.get("retrieval_tags")),
                n_success_memory,
            )
            reflection_records = self._tag_records(
                self.validated_reflections,
                _as_list(module.get("retrieval_tags")),
                n_reflection_memory,
            )
        return format_examination_memory_context(success_records, reflection_records)

    def _tag_records(
        self,
        records: list[dict[str, Any]],
        target_tags: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or not records:
            return []
        target_tag_set = set(target_tags)
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for index, record in enumerate(records):
            overlap = len(target_tag_set & set(_as_list(record.get("retrieval_tags"))))
            scored.append((overlap, index, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored[:limit]]

    def append_eval_summary(self, summary: dict[str, Any]) -> None:
        self.eval_summaries.append(summary)
        append_jsonl(self.eval_summary_path, summary)

    def append_heldout_eval_summary(self, summary: dict[str, Any]) -> None:
        self.heldout_eval_summaries.append(summary)
        append_jsonl(self.heldout_eval_summary_path, summary)

    def write_final_summary(self, summary: dict[str, Any]) -> None:
        write_json(self.final_summary_path, summary)


def format_examination_memory_context(
    success_records: list[dict[str, Any]],
    reflection_records: list[dict[str, Any]],
) -> str:
    if not success_records and not reflection_records:
        return "No previous successful examination cases or validated examination reflections available."

    sections: list[str] = []
    if success_records:
        lines = ["Successful examination cases:"]
        for record in success_records:
            lines.append(
                "\n".join(
                    [
                        f"- Case ID: {record.get('successful_case_id')}",
                        f"  Exam focus: {', '.join(_as_list(record.get('exam_focus')))}",
                        f"  General rule: {record.get('expert_lesson')}",
                        f"  Presenting case pattern: {record.get('patient_case')}",
                        f"  Successful plan: {record.get('doctor_answer')}",
                    ]
                )
            )
        sections.append("\n".join(lines))

    if reflection_records:
        lines = ["Validated examination reflections:"]
        for record in reflection_records:
            lines.append(
                "\n".join(
                    [
                        f"- Reflection ID: {record.get('reflection_id')}",
                        f"  Exam focus: {', '.join(_as_list(record.get('exam_focus')))}",
                        f"  Failure mode: {record.get('failure_mode')}",
                        f"  Reusable rule: {record.get('reflection')}",
                    ]
                )
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


class ExaminationLoop:
    def __init__(self, backend: Any, config: ExaminationLoopConfig) -> None:
        self.backend = backend
        self.config = config
        self.modules = load_examination_modules(config.exam_kb_path)
        self.rng = random.Random(config.seed)
        self.memory = ExaminationMemoryStore(
            runs_root=config.runs_root,
            run_name=config.run_name,
            config_snapshot={
                **asdict(config),
                "task_type": EXAMINATION_TASK_TYPE,
                "included_module_count": len(self.modules),
            },
            retrieval_mode=config.retrieval_mode,
            embedding_model=config.embedding_model,
            embedding_device=config.embedding_device,
            embedding_batch_size=config.embedding_batch_size,
            output_dir=config.output_dir,
            resume_existing=config.resume_existing,
        )
        self._restore_metrics_from_memory()
        self.eval_success_milestones = tuple(
            sorted(
                {
                    milestone
                    for milestone in config.eval_success_milestones
                    if 0 < milestone <= config.n_successful_cases
                }
            )
        )
        self.completed_success_eval_milestones: set[int] = (
            self._completed_eval_milestones_from_memory()
        )

    def run(self) -> dict[str, Any]:
        max_attempts = max(self.config.n_successful_cases * MAX_ATTEMPT_MULTIPLIER, 100)
        self._log(
            "Examination run started"
            f" | run_dir={self.memory.run_dir}"
            f" | target_successes={self.config.n_successful_cases}"
            f" | model={self.config.model}"
        )
        if self.successful_cases or self.attempted_patients:
            self._log(
                "Resumed persisted examination run"
                f" | attempts={self.attempted_patients}"
                f" | successes={self.successful_cases}"
                f" | reflections={self.validated_reflection_count}"
            )
        while self.successful_cases < self.config.n_successful_cases:
            if self.attempted_patients >= max_attempts:
                raise RuntimeError(
                    f"Reached attempt limit ({max_attempts}) before hitting "
                    f"{self.config.n_successful_cases} successful examination cases."
                )
            module = self._sample_module()
            attempted_patient_id = f"exam_patient_{self.attempted_patients + 1:06d}"
            self._log_event(
                f"[{attempted_patient_id}] sampling"
                f" | condition={module['condition_name']}"
            )

            patient_case = self._generate_patient_case(module)
            self.attempted_patients += 1
            if not patient_case:
                self.patient_generation_empty_count += 1
                self.memory.record_discard(
                    attempted_patient_id=attempted_patient_id,
                    module=module,
                    reason="patient_generation_empty",
                )
                self._maybe_run_periodic_evaluation()
                continue

            qc_result = self._evaluate_qc(module, patient_case)
            if not qc_result.usable:
                self.qc_discarded_patients += 1
                self.memory.record_discard(
                    attempted_patient_id=attempted_patient_id,
                    module=module,
                    reason=f"qc_discard:{qc_result.reason or 'qc_unusable'}",
                    details=(
                        {
                            "examination_qc_parse_error": qc_result.parse_error,
                            "examination_qc_raw_output": qc_result.raw_output,
                        }
                        if qc_result.parse_error
                        else None
                    ),
                )
                self._maybe_run_periodic_evaluation()
                continue

            memory_context = self.memory.build_memory_context(
                patient_case=patient_case,
                module=module,
                n_success_memory=self.config.n_success_memory,
                n_reflection_memory=self.config.n_reflection_memory,
            )
            first_answer = self.backend.generate(
                prompt=build_examination_doctor_prompt(
                    memory_context=memory_context,
                    patient_case=patient_case,
                ),
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            first_judgment = self._judge(module, patient_case, first_answer)
            if first_judgment.correct:
                self._record_success(
                    attempted_patient_id=attempted_patient_id,
                    module=module,
                    patient_case=patient_case,
                    doctor_answer=first_answer,
                    judge_result=first_judgment,
                    solved_on="first_attempt",
                    validated_reflection_used=None,
                )
                self._log_event(f"[{attempted_patient_id}] success | first_attempt")
            else:
                self.retry_attempts += 1
                if not first_judgment.reflection:
                    self.no_reflection_count += 1
                    self.retry_fail_discards += 1
                    reason = (
                        "malformed_judge_output"
                        if first_judgment.parse_error
                        else "retry_failed:no_reflection"
                    )
                    self.memory.record_discard(
                        attempted_patient_id=attempted_patient_id,
                        module=module,
                        reason=reason,
                        details=_judge_error_details(first_judgment),
                    )
                    self._maybe_run_periodic_evaluation()
                    continue

                self.reflections_emitted_count += 1
                retry_answer = self.backend.generate(
                    prompt=build_examination_doctor_prompt(
                        memory_context=memory_context,
                        patient_case=patient_case,
                        previous_answer=first_answer,
                        provisional_reflection=first_judgment.reflection,
                    ),
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                retry_judgment = self._judge(module, patient_case, retry_answer)
                if retry_judgment.correct:
                    success_record = self._record_success(
                        attempted_patient_id=attempted_patient_id,
                        module=module,
                        patient_case=patient_case,
                        doctor_answer=retry_answer,
                        judge_result=retry_judgment,
                        solved_on="retry",
                        validated_reflection_used=first_judgment.reflection,
                    )
                    self.memory.record_validated_reflection(
                        successful_case_id=success_record["successful_case_id"],
                        attempted_patient_id=attempted_patient_id,
                        module=module,
                        patient_case=patient_case,
                        doctor_answer=retry_answer,
                        reflection=first_judgment.reflection,
                    )
                    self.validated_reflection_count += 1
                    self._log_event(f"[{attempted_patient_id}] success | retry")
                else:
                    self.failed_reflection_count += 1
                    self.retry_fail_discards += 1
                    reason = (
                        "retry_failed:malformed_judge_output"
                        if retry_judgment.parse_error
                        else "retry_failed:unrecovered"
                    )
                    self.memory.record_discard(
                        attempted_patient_id=attempted_patient_id,
                        module=module,
                        reason=reason,
                        details=_judge_error_details(retry_judgment),
                    )

            self._maybe_run_periodic_evaluation()

        final_summary = self._build_summary()
        self.memory.write_final_summary(final_summary)
        self._log(
            "Examination run finished"
            f" | successes={self.successful_cases}/{self.config.n_successful_cases}"
            f" | attempts={self.attempted_patients}"
            f" | final_summary={self.memory.final_summary_path}"
        )
        return {"run_dir": str(self.memory.run_dir), "summary": final_summary}

    def _sample_module(self) -> dict[str, Any]:
        counts = self.memory.success_counts_by_condition()
        least_seen = min(counts.get(module["condition_id"], 0) for module in self.modules)
        candidates = [
            module
            for module in self.modules
            if counts.get(module["condition_id"], 0) == least_seen
        ]
        return self.rng.choice(candidates)

    def _generate_patient_case(self, module: dict[str, Any]) -> str | None:
        prompt = build_examination_case_generation_prompt(module)
        for temperature in [self.config.temperature, 0.0]:
            patient_case = self.backend.generate(
                prompt=prompt,
                max_tokens=min(self.config.max_tokens, 400),
                temperature=temperature,
            ).strip()
            if patient_case:
                return patient_case
        return None

    def _evaluate_qc(self, module: dict[str, Any], patient_case: str) -> ExaminationQCResult:
        response = self.backend.generate(
            prompt=build_examination_qc_prompt(module=module, patient_case=patient_case),
            max_tokens=160,
            temperature=0.0,
        )
        return parse_examination_qc_response(response)

    def _judge(
        self,
        module: dict[str, Any],
        patient_case: str,
        doctor_answer: str,
    ) -> ExaminationJudgeResult:
        response = self.backend.generate(
            prompt=build_examination_judge_prompt(
                module_or_item=module,
                patient_case=patient_case,
                doctor_answer=doctor_answer,
                hidden_gold=_exam_hidden_gold(module),
            ),
            max_tokens=260,
            temperature=0.0,
        )
        return parse_examination_judge_response(response)

    def _record_success(
        self,
        attempted_patient_id: str,
        module: dict[str, Any],
        patient_case: str,
        doctor_answer: str,
        judge_result: ExaminationJudgeResult,
        solved_on: str,
        validated_reflection_used: str | None,
    ) -> dict[str, Any]:
        self.successful_cases += 1
        if solved_on == "first_attempt":
            self.first_attempt_successes += 1
        elif solved_on == "retry":
            self.retry_successes += 1
        condition_id = module["condition_id"]
        self.coverage_by_condition.add(condition_id)
        for lane in _as_list(module.get("curriculum_lanes")):
            self.coverage_by_curriculum_lane[lane].add(condition_id)
        for focus in _as_list(module.get("exam_focus")):
            self.coverage_by_exam_focus[focus].add(condition_id)
        return self.memory.record_success(
            attempted_patient_id=attempted_patient_id,
            module=module,
            patient_case=patient_case,
            doctor_answer=doctor_answer,
            judge_result=judge_result,
            solved_on=solved_on,
            validated_reflection_used=validated_reflection_used,
        )

    def _build_summary(self) -> dict[str, Any]:
        answered_cases = self.attempted_patients - self.qc_discarded_patients
        total_discards = self.qc_discarded_patients + self.retry_fail_discards
        return {
            "task_type": EXAMINATION_TASK_TYPE,
            "written_at": iso_now(),
            "successful_cases": self.successful_cases,
            "attempted_patients": self.attempted_patients,
            "first_attempt_successes": self.first_attempt_successes,
            "retry_successes": self.retry_successes,
            "retry_attempts": self.retry_attempts,
            "patient_generation_empty_count": self.patient_generation_empty_count,
            "qc_discarded_patients": self.qc_discarded_patients,
            "retry_fail_discards": self.retry_fail_discards,
            "reflections_emitted_count": self.reflections_emitted_count,
            "validated_reflection_count": self.validated_reflection_count,
            "failed_reflection_count": self.failed_reflection_count,
            "no_reflection_count": self.no_reflection_count,
            "first_attempt_success_rate": safe_rate(
                self.first_attempt_successes,
                answered_cases,
            ),
            "retry_rescue_rate": safe_rate(self.retry_successes, self.retry_attempts),
            "discard_rate": safe_rate(total_discards, self.attempted_patients),
            "coverage_by_condition": len(self.coverage_by_condition),
            "coverage_by_curriculum_lane": {
                lane: len(condition_ids)
                for lane, condition_ids in sorted(self.coverage_by_curriculum_lane.items())
            },
            "coverage_by_exam_focus": {
                focus: len(condition_ids)
                for focus, condition_ids in sorted(self.coverage_by_exam_focus.items())
            },
            "successful_case_memory_size": len(self.memory.successful_cases),
            "validated_reflection_memory_size": len(self.memory.validated_reflections),
        }

    def _restore_metrics_from_memory(self) -> None:
        success_records = self.memory.successful_cases
        discard_records = self.memory.discards

        self.successful_cases = len(success_records)
        self.attempted_patients = len(success_records) + len(discard_records)
        self.first_attempt_successes = sum(
            1 for record in success_records if record.get("solved_on") == "first_attempt"
        )
        self.retry_successes = sum(
            1 for record in success_records if record.get("solved_on") == "retry"
        )
        self.patient_generation_empty_count = sum(
            1
            for record in discard_records
            if record.get("reason") == "patient_generation_empty"
        )
        self.qc_discarded_patients = sum(
            1
            for record in discard_records
            if str(record.get("reason", "")).startswith("qc_discard:")
        )
        self.retry_fail_discards = sum(
            1 for record in discard_records if self._is_retry_discard(record)
        )
        self.retry_attempts = self.retry_successes + self.retry_fail_discards
        self.no_reflection_count = sum(
            1 for record in discard_records if self._is_no_reflection_discard(record)
        )
        self.failed_reflection_count = sum(
            1 for record in discard_records if self._is_failed_reflection_discard(record)
        )
        self.validated_reflection_count = len(self.memory.validated_reflections)
        self.reflections_emitted_count = (
            self.validated_reflection_count + self.failed_reflection_count
        )

        self.coverage_by_condition: set[str] = set()
        self.coverage_by_curriculum_lane: dict[str, set[str]] = defaultdict(set)
        self.coverage_by_exam_focus: dict[str, set[str]] = defaultdict(set)
        for record in success_records:
            condition_id = str(record.get("condition_id", ""))
            if not condition_id:
                continue
            self.coverage_by_condition.add(condition_id)
            for lane in _as_list(record.get("training_lanes")):
                self.coverage_by_curriculum_lane[lane].add(condition_id)
            for focus in _as_list(record.get("exam_focus")):
                self.coverage_by_exam_focus[focus].add(condition_id)

    def _is_retry_discard(self, record: dict[str, Any]) -> bool:
        reason = str(record.get("reason", ""))
        return reason == "malformed_judge_output" or reason.startswith("retry_failed:")

    def _is_no_reflection_discard(self, record: dict[str, Any]) -> bool:
        reason = str(record.get("reason", ""))
        return reason in {"malformed_judge_output", "retry_failed:no_reflection"}

    def _is_failed_reflection_discard(self, record: dict[str, Any]) -> bool:
        reason = str(record.get("reason", ""))
        return reason in {
            "retry_failed:malformed_judge_output",
            "retry_failed:unrecovered",
        }

    def _completed_eval_milestones_from_memory(self) -> set[int]:
        milestones: set[int] = set()
        for summary in [*self.memory.eval_summaries, *self.memory.heldout_eval_summaries]:
            value = summary.get("trigger_successful_cases")
            if isinstance(value, int):
                milestones.add(value)
        return milestones

    def _maybe_run_periodic_evaluation(self) -> None:
        summary = self._build_summary()
        if self.config.log_every > 0 and self.attempted_patients % self.config.log_every == 0:
            self._log(
                "Metrics"
                f" | attempts={self.attempted_patients}"
                f" successes={self.successful_cases}/{self.config.n_successful_cases}"
                f" first={self.first_attempt_successes}"
                f" retry_success={self.retry_successes}"
                f" qc_discards={self.qc_discarded_patients}"
                f" retry_fail_discards={self.retry_fail_discards}"
            )
        if not self.config.run_evaluation:
            return
        self._maybe_run_success_milestone_evaluation(summary)
        if self.config.eval_every <= 0:
            return
        if self.attempted_patients <= 0:
            return
        if self.attempted_patients % self.config.eval_every != 0:
            return
        self.memory.append_eval_summary(summary)
        if self.config.eval_file:
            eval_summary = self._run_heldout_evaluation()
            self.memory.append_heldout_eval_summary(eval_summary)

    def _maybe_run_success_milestone_evaluation(self, summary: dict[str, Any]) -> None:
        if not self.config.eval_file:
            return
        due_milestones = [
            milestone
            for milestone in self.eval_success_milestones
            if milestone <= self.successful_cases
            and milestone not in self.completed_success_eval_milestones
        ]
        for milestone in due_milestones:
            self.completed_success_eval_milestones.add(milestone)
            checkpoint_summary = dict(summary)
            checkpoint_summary["trigger_successful_cases"] = milestone
            self.memory.append_eval_summary(checkpoint_summary)
            output_dir = self.memory.run_dir / f"eval_success_{milestone:06d}"
            eval_summary = self._run_heldout_evaluation(
                output_dir=output_dir,
                trigger_successful_cases=milestone,
            )
            self.memory.append_heldout_eval_summary(eval_summary)
            self._log(
                "Held-out examination milestone evaluation written"
                f" | successful_cases={milestone}"
                f" | accuracy={eval_summary['accuracy']}"
                f" | path={output_dir / 'eval_summary.json'}"
            )

    def _run_heldout_evaluation(
        self,
        output_dir: Path | None = None,
        trigger_successful_cases: int | None = None,
    ) -> dict[str, Any]:
        runner = ExaminationEvaluationRunner(
            backend=self.backend,
            config=ExaminationEvalConfig(
                eval_file=self.config.eval_file or "",
                exam_kb_path=self.config.exam_kb_path,
                run_name=self.config.run_name,
                backend=self.config.backend,
                model=self.config.model,
                eval_mode=self.config.eval_mode,
                n_success_memory=self.config.n_success_memory,
                n_reflection_memory=self.config.n_reflection_memory,
                retrieval_mode=self.config.retrieval_mode,
                embedding_model=self.config.embedding_model,
                embedding_device=self.config.embedding_device,
                embedding_batch_size=self.config.embedding_batch_size,
                temperature=0.0,
                max_tokens=min(self.config.max_tokens, 512),
                eval_limit=self.config.eval_limit,
                output_dir=str(output_dir or self.memory.run_dir),
                quiet=self.config.quiet,
            ),
            successful_case_records=list(self.memory.successful_cases),
            validated_reflection_records=list(self.memory.validated_reflections),
            successful_case_index=self.memory.success_index,
            validated_reflection_index=self.memory.reflection_index,
        )
        return runner.run(
            trigger_attempted_patients=self.attempted_patients,
            trigger_successful_cases=trigger_successful_cases,
        )

    def _log(self, message: str) -> None:
        if self.config.quiet:
            return
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    def _log_event(self, message: str) -> None:
        if self.config.verbose_events:
            self._log(message)


class ExaminationEvaluationRunner:
    def __init__(
        self,
        backend: Any,
        config: ExaminationEvalConfig,
        successful_case_records: list[dict[str, Any]] | None = None,
        validated_reflection_records: list[dict[str, Any]] | None = None,
        successful_case_index: SemanticRecordIndex | None = None,
        validated_reflection_index: SemanticRecordIndex | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        loaded_items = load_examination_eval_items(config.eval_file)
        self.items = (
            loaded_items[: config.eval_limit]
            if config.eval_limit is not None
            else loaded_items
        )
        self.modules_by_condition = {
            module["condition_id"]: module
            for module in load_examination_modules(config.exam_kb_path)
        }
        self.successful_cases = (
            list(successful_case_records)
            if successful_case_records is not None
            else _load_optional_jsonl(config.successful_cases)
        )
        self.validated_reflections = (
            list(validated_reflection_records)
            if validated_reflection_records is not None
            else _load_optional_jsonl(config.validated_reflections)
        )
        if config.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unsupported retrieval_mode: {config.retrieval_mode}")
        self.eval_mode = self._resolve_eval_mode(config.eval_mode)
        self.success_index: SemanticRecordIndex | None = None
        self.reflection_index: SemanticRecordIndex | None = None
        if config.retrieval_mode == "semantic" and (
            successful_case_index is not None or validated_reflection_index is not None
        ):
            self.success_index = successful_case_index
            self.reflection_index = validated_reflection_index
        elif config.retrieval_mode == "semantic":
            embedding_config = embedding_config_from_env(
                model=config.embedding_model,
                device=config.embedding_device,
                batch_size=config.embedding_batch_size,
            )
            embedder = LocalTextEmbedder(embedding_config)
            self.success_index = SemanticRecordIndex(
                records=self.successful_cases,
                embedder=embedder,
                text_builder=exam_case_retrieval_text,
            )
            self.reflection_index = SemanticRecordIndex(
                records=self.validated_reflections,
                embedder=embedder,
                text_builder=exam_reflection_retrieval_text,
            )

        self.output_dir = (
            Path(config.output_dir)
            if config.output_dir
            else Path(config.runs_root) / config.run_name
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_dir / "eval_results.jsonl"
        self.summary_path = self.output_dir / "eval_summary.json"
        self.metadata_path = self.output_dir / "eval_metadata.json"
        write_json(
            self.metadata_path,
            {
                **asdict(config),
                "task_type": EXAMINATION_TASK_TYPE,
                "resolved_eval_mode": self.eval_mode,
                "n_items": len(self.items),
                "n_available_items": len(loaded_items),
                "n_loaded_successful_cases": len(self.successful_cases),
                "n_loaded_validated_reflections": len(self.validated_reflections),
            },
        )

    def run(
        self,
        trigger_attempted_patients: int | None = None,
        trigger_successful_cases: int | None = None,
    ) -> dict[str, Any]:
        if self.results_path.exists():
            self.results_path.unlink()

        correct_total = 0
        essential_hits_total = 0
        essential_possible_total = 0
        dangerous_miss_total = 0
        dangerous_possible_total = 0

        self._log(
            f"Examination evaluation started | mode={self.eval_mode} | items={len(self.items)} | output_dir={self.output_dir}"
        )
        for index, item in enumerate(self.items, start=1):
            success_records, reflection_records = self._retrieve_memories(item)
            memory_context = format_examination_memory_context(
                success_records,
                reflection_records,
            )
            raw_answer = self.backend.generate(
                prompt=build_examination_doctor_prompt(
                    memory_context=memory_context,
                    patient_case=str(item.get("presenting_case", "")),
                    question=str(item.get("question", "")),
                ),
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            judgment = self._judge_item(item, raw_answer)
            correct_total += int(judgment.correct)
            essential_possible = len(_as_list(item.get("gold_essential")))
            dangerous_possible = len(_as_list(item.get("dangerous_misses")))
            essential_possible_total += essential_possible
            dangerous_possible_total += dangerous_possible
            essential_hits_total += min(len(judgment.essential_hits), essential_possible)
            dangerous_miss_total += min(len(judgment.dangerous_misses), dangerous_possible)
            result = {
                "question_id": item.get("question_id"),
                "task_type": EXAMINATION_TASK_TYPE,
                "condition_id": item.get("condition_id"),
                "model_raw_answer": raw_answer,
                "judge_raw_output": judgment.raw_output,
                "correct": judgment.correct,
                "essential_hits": judgment.essential_hits,
                "dangerous_misses": judgment.dangerous_misses,
                "judge_parse_error": judgment.parse_error,
                "eval_mode": self.eval_mode,
                "retrieved_success_case_ids": [
                    record.get("successful_case_id") for record in success_records
                ],
                "retrieved_reflection_ids": [
                    record.get("reflection_id") for record in reflection_records
                ],
            }
            append_jsonl(self.results_path, result)
            self._log(
                f"[exam eval {index:03d}/{len(self.items):03d}]"
                f" question_id={item.get('question_id')}"
                f" correct={judgment.correct}"
            )

        incorrect_total = len(self.items) - correct_total
        summary = {
            "task_type": EXAMINATION_TASK_TYPE,
            "item_count": len(self.items),
            "n_questions": len(self.items),
            "correct_count": correct_total,
            "incorrect_count": incorrect_total,
            "accuracy": safe_rate(correct_total, len(self.items)),
            "essential_hit_rate": safe_rate(
                essential_hits_total,
                essential_possible_total,
            ),
            "dangerous_miss_rate": safe_rate(
                dangerous_miss_total,
                dangerous_possible_total,
            ),
            "eval_mode": self.eval_mode,
            "n_success_memory": self.config.n_success_memory,
            "n_reflection_memory": self.config.n_reflection_memory,
            "n_loaded_successful_cases": len(self.successful_cases),
            "n_loaded_validated_reflections": len(self.validated_reflections),
            "retrieval_mode": self.config.retrieval_mode,
            "embedding_model": self.config.embedding_model,
        }
        if trigger_attempted_patients is not None:
            summary["trigger_attempted_patients"] = trigger_attempted_patients
        if trigger_successful_cases is not None:
            summary["trigger_successful_cases"] = trigger_successful_cases
        write_json(self.summary_path, summary)
        self._log(
            f"Examination evaluation finished | accuracy={summary['accuracy']} | summary={self.summary_path}"
        )
        return summary

    def _retrieve_memories(
        self,
        item: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self.eval_mode in {"no_memory", "kb_only"}:
            return [], []
        if not self.successful_cases and not self.validated_reflections:
            return [], []
        query_text = str(item.get("presenting_case") or item.get("question") or "")
        if self.config.retrieval_mode == "semantic" and (
            self.success_index is not None or self.reflection_index is not None
        ):
            if not query_text.strip():
                return [], []
            embedder = (
                self.success_index.embedder
                if self.success_index is not None
                else self.reflection_index.embedder
            )
            query_embedding = embedder.encode([query_text])
            success_records = (
                self.success_index.search_embedding(
                    query_embedding,
                    self.config.n_success_memory,
                )
                if self.successful_cases and self.success_index is not None
                else []
            )
            reflection_records = (
                self.reflection_index.search_embedding(
                    query_embedding,
                    self.config.n_reflection_memory,
                )
                if self.validated_reflections and self.reflection_index is not None
                else []
            )
            return success_records, reflection_records

        return (
            _tag_records(
                self.successful_cases,
                _as_list(item.get("retrieval_tags")),
                self.config.n_success_memory,
            ),
            _tag_records(
                self.validated_reflections,
                _as_list(item.get("retrieval_tags")),
                self.config.n_reflection_memory,
            ),
        )

    def _judge_item(
        self,
        item: dict[str, Any],
        doctor_answer: str,
    ) -> ExaminationJudgeResult:
        response = self.backend.generate(
            prompt=build_examination_judge_prompt(
                module_or_item=item,
                patient_case=str(item.get("presenting_case", "")),
                doctor_answer=doctor_answer,
                hidden_gold=_exam_eval_hidden_gold(item),
            ),
            max_tokens=260,
            temperature=0.0,
        )
        return parse_examination_judge_response(response)

    def _resolve_eval_mode(self, requested_mode: str | None) -> str:
        if requested_mode is not None:
            if requested_mode not in EVAL_MODES:
                raise ValueError(f"Unsupported eval_mode: {requested_mode}")
            return requested_mode
        has_memory = bool(self.successful_cases or self.validated_reflections)
        return "with_memory" if has_memory else "no_memory"

    def _log(self, message: str) -> None:
        if self.config.quiet:
            return
        print(message, flush=True)


def _load_optional_jsonl(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    resolved = Path(path)
    return _load_jsonl_allow_truncated_final_line(resolved)


def _load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _load_jsonl_allow_truncated_final_line(path)


def _load_jsonl_allow_truncated_final_line(path: Path) -> list[dict[str, Any]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            records.append(_loads_json(line))
        except ValueError:
            if index == len(lines) - 1:
                break
            raise
    return records


def _max_numeric_suffix(
    records: list[dict[str, Any]],
    field: str,
    fallback: int,
) -> int:
    maximum = fallback
    for record in records:
        value = record.get(field)
        if not isinstance(value, str):
            continue
        suffix = value.rsplit("_", 1)[-1]
        if suffix.isdigit():
            maximum = max(maximum, int(suffix))
    return maximum


def _loads_json(text: str) -> dict[str, Any]:
    import json

    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")
    return payload


def _tag_records(
    records: list[dict[str, Any]],
    target_tags: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or not records:
        return []
    target_tag_set = set(target_tags)
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for index, record in enumerate(records):
        overlap = len(target_tag_set & set(_as_list(record.get("retrieval_tags"))))
        scored.append((overlap, index, record))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [record for _, _, record in scored[:limit]]


def _judge_result_payload(result: ExaminationJudgeResult) -> dict[str, Any]:
    return {
        "correct": result.correct,
        "essential_hits": result.essential_hits,
        "dangerous_misses": result.dangerous_misses,
        "raw_output": result.raw_output,
        "parse_error": result.parse_error,
    }


def _judge_error_details(result: ExaminationJudgeResult) -> dict[str, Any] | None:
    if not result.parse_error:
        return None
    return {
        "judge_parse_error": result.parse_error,
        "judge_raw_output": result.raw_output,
    }


def _build_exam_expert_lesson(module: dict[str, Any]) -> str:
    focus = _as_list(module.get("exam_focus"))
    essential = _as_list(module.get("essential_examination_or_tests"))
    focus_phrase = focus[0].replace("_", " ") if focus else "the visible syndrome"
    essential_phrase = essential[0] if essential else "the syndrome-defining specimen or test"
    return (
        f"For {focus_phrase} presentations, prioritize {essential_phrase} "
        "and focused examination/history before broad low-value testing."
    )


def _classify_exam_failure_mode(reflection: str | None) -> str:
    if not reflection:
        return "examination_selection_miss"
    lowered = reflection.lower()
    if "culture" in lowered or "specimen" in lowered or "sample" in lowered:
        return "missed_specimen_strategy"
    if "safety" in lowered or "shock" in lowered or "sepsis" in lowered:
        return "missed_safety_assessment"
    if "exposure" in lowered or "travel" in lowered or "sexual" in lowered:
        return "missed_context_history"
    if "low-value" in lowered or "broad" in lowered or "panel" in lowered:
        return "low_value_testing"
    return "examination_selection_miss"


def _now_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
