from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils import (
    append_jsonl,
    build_expert_lesson,
    classify_failure_mode,
    iso_now,
    now_timestamp,
    safe_rate,
    slugify,
    unique_strings,
    write_json,
)


class MemoryStore:
    def __init__(
        self,
        runs_root: str,
        run_name: str,
        config_snapshot: dict[str, Any],
    ) -> None:
        run_directory_name = f"{now_timestamp()}_{slugify(run_name)}"
        self.run_dir = Path(runs_root) / run_directory_name
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.success_path = self.run_dir / "successful_cases.jsonl"
        self.reflection_path = self.run_dir / "validated_reflections.jsonl"
        self.discard_path = self.run_dir / "discard_summary.jsonl"
        self.eval_summary_path = self.run_dir / "eval_summaries.jsonl"
        self.final_summary_path = self.run_dir / "final_summary.json"
        self.metadata_path = self.run_dir / "run_metadata.json"

        self.successful_cases: list[dict[str, Any]] = []
        self.validated_reflections: list[dict[str, Any]] = []
        self.discards: list[dict[str, Any]] = []
        self.eval_summaries: list[dict[str, Any]] = []

        self._success_counter = 0
        self._reflection_counter = 0
        write_json(self.metadata_path, config_snapshot)

    def success_counts_by_condition(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.successful_cases:
            condition_id = record["condition_id"]
            counts[condition_id] = counts.get(condition_id, 0) + 1
        return counts

    def record_success(
        self,
        attempted_patient_id: str,
        condition: dict[str, Any],
        training_lane: str,
        patient_case: str,
        doctor_answer: str,
        solved_on: str,
        validated_reflection_used: str | None,
    ) -> dict[str, Any]:
        self._success_counter += 1
        record = {
            "successful_case_id": f"success_{self._success_counter:06d}",
            "attempted_patient_id": attempted_patient_id,
            "condition_id": condition["condition_id"],
            "condition_name": condition["condition_name"],
            "training_lane": training_lane,
            "expert_value": condition.get("expert_curriculum", {}).get("expert_value"),
            "patient_case": patient_case,
            "doctor_answer": doctor_answer,
            "solved_on": solved_on,
            "validated_reflection_used": validated_reflection_used,
            "expert_lesson": build_expert_lesson(condition),
            "retrieval_tags": unique_strings(
                condition.get("expert_curriculum", {}).get("retrieval_tags", [])
            ),
        }
        self.successful_cases.append(record)
        append_jsonl(self.success_path, record)
        return record

    def record_validated_reflection(
        self,
        successful_case_id: str,
        attempted_patient_id: str,
        condition: dict[str, Any],
        training_lane: str,
        reflection: str,
    ) -> dict[str, Any]:
        self._reflection_counter += 1
        record = {
            "reflection_id": f"reflection_{self._reflection_counter:06d}",
            "successful_case_id": successful_case_id,
            "attempted_patient_id": attempted_patient_id,
            "condition_id": condition["condition_id"],
            "condition_name": condition["condition_name"],
            "training_lane": training_lane,
            "expert_value": condition.get("expert_curriculum", {}).get("expert_value"),
            "failure_mode": classify_failure_mode(reflection),
            "reflection": reflection,
            "validated_by": "same_patient_retry",
            "retrieval_tags": unique_strings(
                condition.get("expert_curriculum", {}).get("retrieval_tags", [])
            ),
        }
        self.validated_reflections.append(record)
        append_jsonl(self.reflection_path, record)
        return record

    def record_discard(
        self,
        attempted_patient_id: str,
        condition: dict[str, Any],
        training_lane: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "attempted_patient_id": attempted_patient_id,
            "condition_id": condition["condition_id"],
            "condition_name": condition["condition_name"],
            "training_lane": training_lane,
            "reason": reason,
        }
        if details:
            record["details"] = details
        self.discards.append(record)
        append_jsonl(self.discard_path, record)
        return record

    def build_memory_context(
        self,
        condition: dict[str, Any],
        training_lane: str,
        n_success_memory: int,
        n_reflection_memory: int,
    ) -> str:
        success_records = self._retrieve_records(
            records=self.successful_cases,
            target_tags=condition.get("expert_curriculum", {}).get("retrieval_tags", []),
            training_lane=training_lane,
            limit=n_success_memory,
        )
        reflection_records = self._retrieve_records(
            records=self.validated_reflections,
            target_tags=condition.get("expert_curriculum", {}).get("retrieval_tags", []),
            training_lane=training_lane,
            limit=n_reflection_memory,
        )

        if not success_records and not reflection_records:
            return "No previous successful cases or validated reflections available."

        sections: list[str] = []
        if success_records:
            lines = ["Successful cases:"]
            for record in success_records:
                lines.append(
                    "\n".join(
                        [
                            f"- Case ID: {record['successful_case_id']}",
                            f"  Condition: {record['condition_name']}",
                            f"  Training lane: {record['training_lane']}",
                            f"  General rule: {record['expert_lesson']}",
                            f"  Example case pattern: {record['patient_case']}",
                            f"  Example diagnosis: {record['doctor_answer']}",
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
                            f"- Reflection ID: {record['reflection_id']}",
                            f"  Condition: {record['condition_name']}",
                            f"  Training lane: {record['training_lane']}",
                            f"  Failure mode: {record['failure_mode']}",
                            f"  Reusable rule: {record['reflection']}",
                        ]
                    )
                )
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def append_eval_summary(self, summary: dict[str, Any]) -> None:
        self.eval_summaries.append(summary)
        append_jsonl(self.eval_summary_path, summary)

    def write_final_summary(self, summary: dict[str, Any]) -> None:
        write_json(self.final_summary_path, summary)

    def build_metrics_snapshot(
        self,
        successful_cases: int,
        attempted_patients: int,
        first_attempt_successes: int,
        retry_successes: int,
        retry_attempts: int,
        patient_generation_empty_count: int,
        qc_discarded_patients: int,
        retry_fail_discards: int,
        reflections_emitted_count: int,
        validated_reflection_count: int,
        failed_reflection_count: int,
        no_reflection_count: int,
        success_count_by_training_lane: dict[str, int],
        retry_success_count_by_training_lane: dict[str, int],
        discard_count_by_training_lane: dict[str, int],
        coverage_by_training_lane: dict[str, set[str]],
        must_not_miss_success_count: int,
    ) -> dict[str, Any]:
        answered_cases = attempted_patients - qc_discarded_patients
        total_discards = qc_discarded_patients + retry_fail_discards
        coverage_counts = {
            lane: len(condition_ids)
            for lane, condition_ids in coverage_by_training_lane.items()
        }
        return {
            "written_at": iso_now(),
            "successful_cases": successful_cases,
            "attempted_patients": attempted_patients,
            "first_attempt_successes": first_attempt_successes,
            "retry_successes": retry_successes,
            "retry_attempts": retry_attempts,
            "patient_generation_empty_count": patient_generation_empty_count,
            "qc_discarded_patients": qc_discarded_patients,
            "retry_fail_discards": retry_fail_discards,
            "reflections_emitted_count": reflections_emitted_count,
            "validated_reflection_count": validated_reflection_count,
            "failed_reflection_count": failed_reflection_count,
            "no_reflection_count": no_reflection_count,
            "first_attempt_success_rate": safe_rate(
                first_attempt_successes, answered_cases
            ),
            "retry_rescue_rate": safe_rate(retry_successes, retry_attempts),
            "discard_rate": safe_rate(total_discards, attempted_patients),
            "coverage_by_training_lane": coverage_counts,
            "success_count_by_training_lane": dict(sorted(success_count_by_training_lane.items())),
            "retry_success_count_by_training_lane": dict(
                sorted(retry_success_count_by_training_lane.items())
            ),
            "discard_count_by_training_lane": dict(
                sorted(discard_count_by_training_lane.items())
            ),
            "must_not_miss_success_count": must_not_miss_success_count,
            "successful_case_memory_size": len(self.successful_cases),
            "validated_reflection_memory_size": len(self.validated_reflections),
        }

    def _retrieve_records(
        self,
        records: list[dict[str, Any]],
        target_tags: list[str],
        training_lane: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or not records:
            return []

        target_tag_set = set(unique_strings(target_tags))
        scored_records: list[tuple[int, int, dict[str, Any]]] = []
        for index, record in enumerate(records):
            record_tags = set(unique_strings(record.get("retrieval_tags", [])))
            overlap_score = len(target_tag_set & record_tags)
            lane_bonus = 2 if record.get("training_lane") == training_lane else 0
            score = overlap_score + lane_bonus
            scored_records.append((score, index, record))

        scored_records.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored_records[:limit]]
