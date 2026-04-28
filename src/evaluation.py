from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.kb import load_condition_index
from src.prompts import build_eval_kb_context, build_eval_prompt
from src.semantic_retrieval import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    RETRIEVAL_MODES,
    LocalTextEmbedder,
    SemanticRecordIndex,
    case_retrieval_text,
    embedding_config_from_env,
    experience_retrieval_text,
)
from src.utils import (
    append_jsonl,
    ensure_file_exists,
    load_json,
    load_jsonl,
    safe_rate,
    tokenize_for_overlap,
    unique_strings,
    write_json,
)

EVAL_MODES = {"no_memory", "with_memory", "kb_only", "memory_only"}


@dataclass(frozen=True)
class EvaluationConfig:
    eval_dataset: str
    kb_path: str
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
    allow_eval_learning: bool = False
    runs_root: str = "runs"
    output_dir: str | None = None
    quiet: bool = False


def load_evaluation_dataset(path: str) -> list[dict[str, Any]]:
    resolved = ensure_file_exists(path)
    if resolved.suffix == ".jsonl":
        records = load_jsonl(resolved)
    else:
        payload = load_json(resolved)
        if not isinstance(payload, list):
            raise ValueError(f"Expected evaluation dataset to be a list or JSONL: {resolved}")
        records = []
        for index, row in enumerate(payload, start=1):
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected object at evaluation dataset row {index}, got {type(row).__name__}."
                )
            records.append(row)
    if not records:
        raise RuntimeError(f"Evaluation dataset is empty: {resolved}")
    return records


class EvaluationRunner:
    def __init__(
        self,
        backend: Any,
        config: EvaluationConfig,
        successful_case_records: list[dict[str, Any]] | None = None,
        validated_reflection_records: list[dict[str, Any]] | None = None,
        successful_case_index: SemanticRecordIndex | None = None,
        validated_reflection_index: SemanticRecordIndex | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.questions = load_evaluation_dataset(config.eval_dataset)
        self.condition_index = load_condition_index(config.kb_path)
        self.successful_cases = self._prepare_records(
            successful_case_records
            if successful_case_records is not None
            else self._load_optional_jsonl(config.successful_cases)
        )
        self.validated_reflections = self._prepare_records(
            validated_reflection_records
            if validated_reflection_records is not None
            else self._load_optional_jsonl(config.validated_reflections)
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
                text_builder=case_retrieval_text,
            )
            self.reflection_index = SemanticRecordIndex(
                records=self.validated_reflections,
                embedder=embedder,
                text_builder=experience_retrieval_text,
            )

        output_dir = Path(config.output_dir) if config.output_dir else Path(config.runs_root) / config.run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.results_path = self.output_dir / "eval_results.jsonl"
        self.summary_path = self.output_dir / "eval_summary.json"
        self.metadata_path = self.output_dir / "eval_metadata.json"
        write_json(
            self.metadata_path,
            {
                **asdict(config),
                "resolved_eval_mode": self.eval_mode,
                "n_loaded_successful_cases": len(self.successful_cases),
                "n_loaded_validated_reflections": len(self.validated_reflections),
                "n_questions": len(self.questions),
            },
        )

    def run(
        self,
        trigger_attempted_patients: int | None = None,
        trigger_successful_cases: int | None = None,
    ) -> dict[str, Any]:
        if self.results_path.exists():
            self.results_path.unlink()

        results: list[dict[str, Any]] = []
        correct_total = 0
        lane_stats: dict[str, dict[str, int]] = {}
        must_not_miss_correct = 0
        must_not_miss_total = 0
        non_must_not_miss_correct = 0
        non_must_not_miss_total = 0

        self._log(
            f"Evaluation started | mode={self.eval_mode} | questions={len(self.questions)} | output_dir={self.output_dir}"
        )

        for index, question in enumerate(self.questions, start=1):
            success_records, reflection_records = self._retrieve_memories(question)
            memory_context = self._build_memory_context(
                success_records=success_records,
                reflection_records=reflection_records,
            )
            kb_context = self._build_kb_context(question)
            prompt = build_eval_prompt(
                question=question,
                memory_context=memory_context,
                kb_context=kb_context,
            )
            raw_answer = self.backend.generate(
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            predicted_letter, predicted_diagnosis = self._parse_prediction(raw_answer)
            correct = self._is_correct(
                question=question,
                predicted_letter=predicted_letter,
                predicted_diagnosis=predicted_diagnosis,
            )

            lane = str(question.get("expert_training_lane", "unknown"))
            lane_bucket = lane_stats.setdefault(lane, {"correct": 0, "total": 0})
            lane_bucket["total"] += 1
            if correct:
                lane_bucket["correct"] += 1
                correct_total += 1

            if bool(question.get("must_not_miss")):
                must_not_miss_total += 1
                if correct:
                    must_not_miss_correct += 1
            else:
                non_must_not_miss_total += 1
                if correct:
                    non_must_not_miss_correct += 1

            result = {
                "question_id": question.get("question_id"),
                "expert_training_lane": lane,
                "condition_id": question.get("condition_id"),
                "gold_diagnosis": question.get("gold_diagnosis"),
                "answer_letter": str(question.get("answer_letter", "")).upper(),
                "model_raw_answer": raw_answer,
                "predicted_letter": predicted_letter,
                "predicted_diagnosis": predicted_diagnosis,
                "correct": correct,
                "eval_mode": self.eval_mode,
                "retrieved_success_case_ids": [
                    record.get("successful_case_id") for record in success_records
                ],
                "retrieved_reflection_ids": [
                    record.get("reflection_id") for record in reflection_records
                ],
            }
            results.append(result)
            append_jsonl(self.results_path, result)

            self._log(
                f"[eval {index:03d}/{len(self.questions):03d}]"
                f" question_id={question.get('question_id')}"
                f" correct={correct}"
                f" predicted_letter={predicted_letter or '-'}"
                f" predicted_diagnosis={predicted_diagnosis or '-'}"
            )

        summary = {
            "n_questions": len(self.questions),
            "accuracy": safe_rate(correct_total, len(self.questions)),
            "accuracy_by_lane": {
                lane: safe_rate(bucket["correct"], bucket["total"])
                for lane, bucket in sorted(lane_stats.items())
            },
            "accuracy_must_not_miss": safe_rate(
                must_not_miss_correct, must_not_miss_total
            ),
            "accuracy_non_must_not_miss": safe_rate(
                non_must_not_miss_correct, non_must_not_miss_total
            ),
            "eval_mode": self.eval_mode,
            "n_success_memory": self.config.n_success_memory,
            "n_reflection_memory": self.config.n_reflection_memory,
            "n_loaded_successful_cases": len(self.successful_cases),
            "n_loaded_validated_reflections": len(self.validated_reflections),
            "allow_eval_learning": self.config.allow_eval_learning,
            "retrieval_mode": self.config.retrieval_mode,
            "embedding_model": self.config.embedding_model,
        }
        if trigger_attempted_patients is not None:
            summary["trigger_attempted_patients"] = trigger_attempted_patients
        if trigger_successful_cases is not None:
            summary["trigger_successful_cases"] = trigger_successful_cases

        write_json(self.summary_path, summary)
        self._log(
            f"Evaluation finished | accuracy={summary['accuracy']} | summary={self.summary_path}"
        )
        return summary

    def _load_optional_jsonl(self, path: str | None) -> list[dict[str, Any]]:
        if not path:
            return []
        return load_jsonl(path)

    def _prepare_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            normalized = dict(record)
            normalized["_record_index"] = index
            normalized["retrieval_tags"] = unique_strings(normalized.get("retrieval_tags", []))
            prepared.append(normalized)
        return prepared

    def _resolve_eval_mode(self, requested_mode: str | None) -> str:
        if requested_mode is not None:
            if requested_mode not in EVAL_MODES:
                raise ValueError(f"Unsupported eval_mode: {requested_mode}")
            return requested_mode
        has_memory = bool(self.successful_cases or self.validated_reflections)
        return "with_memory" if has_memory else "no_memory"

    def _retrieve_memories(
        self,
        question: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self.eval_mode in {"no_memory", "kb_only"}:
            return [], []
        if self.config.retrieval_mode == "semantic" and (
            self.success_index is not None or self.reflection_index is not None
        ):
            query_text = str(question.get("question", "")).strip()
            if not query_text:
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

        success_records = self._top_k_records(
            question=question,
            records=self.successful_cases,
            limit=self.config.n_success_memory,
        )
        reflection_records = self._top_k_records(
            question=question,
            records=self.validated_reflections,
            limit=self.config.n_reflection_memory,
        )
        return success_records, reflection_records

    def _top_k_records(
        self,
        question: dict[str, Any],
        records: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or not records:
            return []

        query_tags = unique_strings(question.get("retrieval_tags", []))
        query_tokens = set(query_tags)
        query_tokens |= tokenize_for_overlap(query_tags)
        query_tokens |= tokenize_for_overlap(question.get("condition_id"))
        query_tokens |= tokenize_for_overlap(question.get("expert_training_lane"))
        query_tokens |= tokenize_for_overlap(question.get("expert_value"))
        query_has_tags = bool(query_tags)

        scored: list[tuple[int, int, dict[str, Any]]] = []
        for record in records:
            record_tokens = set(record.get("retrieval_tags", []))
            record_tokens |= tokenize_for_overlap(record.get("retrieval_tags", []))
            record_tokens |= tokenize_for_overlap(record.get("condition_name"))
            record_tokens |= tokenize_for_overlap(record.get("condition_id"))
            record_tokens |= tokenize_for_overlap(record.get("training_lane"))
            record_tokens |= tokenize_for_overlap(record.get("expert_value"))
            overlap = len(query_tokens & record_tokens)
            score = overlap
            if record.get("training_lane") == question.get("expert_training_lane"):
                score += 2
            if record.get("condition_id") == question.get("condition_id"):
                score += 2
            if record.get("expert_value") == question.get("expert_value"):
                score += 1
            scored.append((score, int(record["_record_index"]), record))

        if not query_has_tags:
            scored.sort(key=lambda item: item[1], reverse=True)
        else:
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored[:limit]]

    def _build_memory_context(
        self,
        success_records: list[dict[str, Any]],
        reflection_records: list[dict[str, Any]],
    ) -> str:
        if not success_records and not reflection_records:
            return "No prior successful cases or validated reflections available."

        sections: list[str] = []
        if success_records:
            lines = ["Successful case experience:"]
            for record in success_records:
                lines.append(
                    "\n".join(
                        [
                            f"- Case ID: {record.get('successful_case_id')}",
                            f"  Condition: {record.get('condition_name')}",
                            f"  Training lane: {record.get('training_lane')}",
                            f"  General rule: {record.get('expert_lesson', 'Use syndrome and key discriminators to separate close lookalikes.')}",
                            f"  Example case pattern: {record.get('patient_case', '')}",
                        ]
                    )
                )
            sections.append("\n".join(lines))

        if reflection_records:
            lines = ["Validated reflection rules:"]
            for record in reflection_records:
                lines.append(
                    "\n".join(
                        [
                            f"- Reflection ID: {record.get('reflection_id')}",
                            f"  Condition: {record.get('condition_name')}",
                            f"  Training lane: {record.get('training_lane')}",
                            f"  Failure mode: {record.get('failure_mode')}",
                            f"  Reusable rule: {record.get('reflection')}",
                        ]
                    )
                )
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def _build_kb_context(self, question: dict[str, Any]) -> str | None:
        if self.eval_mode != "kb_only":
            return None
        condition_id = question.get("condition_id")
        condition = self.condition_index.get(condition_id)
        if not condition:
            return None
        return build_eval_kb_context(condition)

    def _parse_prediction(self, raw_answer: str) -> tuple[str | None, str | None]:
        letter_match = re.search(r"^Answer:\s*([A-Za-z])\b", raw_answer, flags=re.MULTILINE)
        diagnosis_match = re.search(r"^Diagnosis:\s*(.+)$", raw_answer, flags=re.MULTILINE)
        predicted_letter = letter_match.group(1).upper() if letter_match else None
        predicted_diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else None
        return predicted_letter, predicted_diagnosis

    def _is_correct(
        self,
        question: dict[str, Any],
        predicted_letter: str | None,
        predicted_diagnosis: str | None,
    ) -> bool:
        gold_letter = str(question.get("answer_letter", "")).upper()
        if predicted_letter and predicted_letter == gold_letter:
            return True

        accepted_answers = [
            str(question.get("gold_diagnosis", "")),
            *[str(answer) for answer in question.get("accepted_answers", [])],
        ]
        normalized_prediction = self._normalize_text(predicted_diagnosis)
        if not normalized_prediction:
            return False
        normalized_answers = {
            self._normalize_text(answer)
            for answer in accepted_answers
            if self._normalize_text(answer)
        }
        return normalized_prediction in normalized_answers

    def _normalize_text(self, value: str | None) -> str:
        if not value:
            return ""
        lowered = value.lower().strip()
        lowered = re.sub(r"\s+", " ", lowered)
        return re.sub(r"[^a-z0-9 ]+", "", lowered)

    def _log(self, message: str) -> None:
        if self.config.quiet:
            return
        print(message, flush=True)
