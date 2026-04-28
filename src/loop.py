from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation import EvaluationConfig, EvaluationRunner
from src.judge import DiagnosisJudge
from src.kb import load_conditions, load_sampler_config
from src.memory import MemoryStore
from src.patient_qc import PatientQC
from src.prompts import build_doctor_prompt, build_patient_generation_prompt
from src.sampler import ExpertCurriculumSampler
from src.semantic_retrieval import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
)

MAX_ATTEMPT_MULTIPLIER = 50


@dataclass(frozen=True)
class SimulationConfig:
    backend: str
    model: str
    kb_path: str
    sampler_config: str
    n_successful_cases: int
    run_name: str
    eval_every: int = 5
    run_evaluation: bool = False
    eval_dataset: str | None = None
    eval_mode: str | None = None
    eval_success_milestones: tuple[int, ...] = ()
    allow_eval_learning: bool = False
    log_every: int = 1
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
    ollama_host: str = "http://localhost:11434"
    patient_qc: bool = False
    runs_root: str = "runs"


class SimulationLoop:
    def __init__(self, backend: Any, config: SimulationConfig) -> None:
        self.backend = backend
        self.config = config

        self.conditions = load_conditions(config.kb_path)
        self.sampler_config = load_sampler_config(config.sampler_config)
        self.sampler = ExpertCurriculumSampler(
            conditions=self.conditions,
            sampler_config=self.sampler_config,
            seed=config.seed,
        )
        self.judge = DiagnosisJudge(backend=backend)
        self.patient_qc = PatientQC(backend=backend, enabled=config.patient_qc)
        self.memory = MemoryStore(
            runs_root=config.runs_root,
            run_name=config.run_name,
            config_snapshot={
                **asdict(config),
                "filtered_condition_count": len(self.conditions),
            },
            retrieval_mode=config.retrieval_mode,
            embedding_model=config.embedding_model,
            embedding_device=config.embedding_device,
            embedding_batch_size=config.embedding_batch_size,
        )

        self.successful_cases = 0
        self.attempted_patients = 0
        self.first_attempt_successes = 0
        self.retry_successes = 0
        self.retry_attempts = 0
        self.patient_generation_empty_count = 0
        self.qc_discarded_patients = 0
        self.retry_fail_discards = 0
        self.reflections_emitted_count = 0
        self.validated_reflection_count = 0
        self.failed_reflection_count = 0
        self.no_reflection_count = 0
        self.must_not_miss_success_count = 0

        self.success_count_by_training_lane: dict[str, int] = defaultdict(int)
        self.retry_success_count_by_training_lane: dict[str, int] = defaultdict(int)
        self.discard_count_by_training_lane: dict[str, int] = defaultdict(int)
        self.coverage_by_training_lane: dict[str, set[str]] = defaultdict(set)
        self.eval_success_milestones = tuple(
            sorted(
                {
                    milestone
                    for milestone in config.eval_success_milestones
                    if 0 < milestone <= config.n_successful_cases
                }
            )
        )
        self.completed_success_eval_milestones: set[int] = set()

    def run(self) -> dict[str, Any]:
        max_attempts = max(
            self.config.n_successful_cases * MAX_ATTEMPT_MULTIPLIER,
            100,
        )
        self._log(
            "Run started"
            f" | run_dir={self.memory.run_dir}"
            f" | target_successes={self.config.n_successful_cases}"
            f" | model={self.config.model}"
            f" | evaluation={'on' if self.config.run_evaluation else 'off'}"
        )

        while self.successful_cases < self.config.n_successful_cases:
            if self.attempted_patients >= max_attempts:
                raise RuntimeError(
                    f"Reached attempt limit ({max_attempts}) before hitting "
                    f"{self.config.n_successful_cases} successful cases."
                )

            sampled = self.sampler.sample(self.memory.success_counts_by_condition())
            condition = sampled.condition
            training_lane = sampled.training_lane

            attempted_patient_id = f"patient_{self.attempted_patients + 1:06d}"
            self._log_event(
                f"[{attempted_patient_id}] sampling"
                f" | lane={training_lane}"
                f" | condition={condition['condition_name']}"
            )
            patient_case = self._generate_patient_case(condition)
            self.attempted_patients += 1
            if not patient_case:
                self.patient_generation_empty_count += 1
                self.discard_count_by_training_lane[training_lane] += 1
                self.memory.record_discard(
                    attempted_patient_id=attempted_patient_id,
                    condition=condition,
                    training_lane=training_lane,
                    reason="patient_generation_empty",
                )
                self._log_event(
                    f"[{attempted_patient_id}] discard"
                    " | reason=patient_generation_empty"
                )
                self._maybe_run_periodic_evaluation()
                continue

            qc_result = self.patient_qc.evaluate(condition=condition, patient_case=patient_case)
            if not qc_result.usable:
                self.qc_discarded_patients += 1
                self.discard_count_by_training_lane[training_lane] += 1
                reason = qc_result.reason or "qc_unusable"
                self.memory.record_discard(
                    attempted_patient_id=attempted_patient_id,
                    condition=condition,
                    training_lane=training_lane,
                    reason=f"qc_discard:{reason}",
                    details=(
                        {
                            "patient_qc_parse_error": qc_result.parse_error,
                            "patient_qc_raw_output": qc_result.raw_output,
                        }
                        if qc_result.parse_error
                        else None
                    ),
                )
                self._log_event(
                    f"[{attempted_patient_id}] discard"
                    f" | reason=qc_discard:{reason}"
                )
                self._maybe_run_periodic_evaluation()
                continue

            memory_context = self.memory.build_memory_context(
                condition=condition,
                training_lane=training_lane,
                patient_case=patient_case,
                n_success_memory=self.config.n_success_memory,
                n_reflection_memory=self.config.n_reflection_memory,
            )
            first_answer = self.backend.generate(
                prompt=build_doctor_prompt(
                    memory_context=memory_context,
                    patient_case=patient_case,
                ),
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            first_judgment = self.judge.evaluate(
                condition=condition,
                patient_case=patient_case,
                doctor_answer=first_answer,
            )

            if first_judgment.correct:
                self._record_success(
                    attempted_patient_id=attempted_patient_id,
                    condition=condition,
                    training_lane=training_lane,
                    patient_case=patient_case,
                    doctor_answer=first_answer,
                    solved_on="first_attempt",
                    validated_reflection_used=None,
                )
                self._log_event(
                    f"[{attempted_patient_id}] success"
                    " | solved_on=first_attempt"
                    f" | condition={condition['condition_name']}"
                )
            else:
                self.retry_attempts += 1
                if not first_judgment.reflection:
                    self.no_reflection_count += 1
                    self.retry_fail_discards += 1
                    self.discard_count_by_training_lane[training_lane] += 1
                    discard_reason = (
                        "malformed_judge_output"
                        if first_judgment.parse_error
                        else "retry_failed:no_reflection"
                    )
                    self.memory.record_discard(
                        attempted_patient_id=attempted_patient_id,
                        condition=condition,
                        training_lane=training_lane,
                        reason=discard_reason,
                        details=(
                            {
                                "judge_parse_error": first_judgment.parse_error,
                                "judge_raw_output": first_judgment.raw_output,
                            }
                            if first_judgment.parse_error
                            else None
                        ),
                    )
                    self._log_event(
                        f"[{attempted_patient_id}] discard"
                        f" | reason={discard_reason}"
                    )
                    self._maybe_run_periodic_evaluation()
                    continue

                self.reflections_emitted_count += 1
                self._log_event(
                    f"[{attempted_patient_id}] retrying"
                    f" | reflection={first_judgment.reflection}"
                )
                retry_answer = self.backend.generate(
                    prompt=build_doctor_prompt(
                        memory_context=memory_context,
                        patient_case=patient_case,
                        previous_answer=first_answer,
                        provisional_reflection=first_judgment.reflection,
                    ),
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                retry_judgment = self.judge.evaluate(
                    condition=condition,
                    patient_case=patient_case,
                    doctor_answer=retry_answer,
                )

                if retry_judgment.correct:
                    success_record = self._record_success(
                        attempted_patient_id=attempted_patient_id,
                        condition=condition,
                        training_lane=training_lane,
                        patient_case=patient_case,
                        doctor_answer=retry_answer,
                        solved_on="retry",
                        validated_reflection_used=first_judgment.reflection,
                    )
                    self.memory.record_validated_reflection(
                        successful_case_id=success_record["successful_case_id"],
                        attempted_patient_id=attempted_patient_id,
                        condition=condition,
                        training_lane=training_lane,
                        patient_case=patient_case,
                        reflection=first_judgment.reflection,
                    )
                    self.validated_reflection_count += 1
                    self._log_event(
                        f"[{attempted_patient_id}] success"
                        " | solved_on=retry"
                        f" | condition={condition['condition_name']}"
                    )
                else:
                    self.failed_reflection_count += 1
                    self.retry_fail_discards += 1
                    self.discard_count_by_training_lane[training_lane] += 1
                    discard_reason = (
                        "retry_failed:malformed_judge_output"
                        if retry_judgment.parse_error
                        else "retry_failed:unrecovered"
                    )
                    self.memory.record_discard(
                        attempted_patient_id=attempted_patient_id,
                        condition=condition,
                        training_lane=training_lane,
                        reason=discard_reason,
                        details=(
                            {
                                "judge_parse_error": retry_judgment.parse_error,
                                "judge_raw_output": retry_judgment.raw_output,
                            }
                            if retry_judgment.parse_error
                            else None
                        ),
                    )
                    self._log_event(
                        f"[{attempted_patient_id}] discard"
                        f" | reason={discard_reason}"
                    )

            self._maybe_run_periodic_evaluation()

        final_summary = self._build_summary()
        self.memory.write_final_summary(final_summary)
        self._log(
            "Run finished"
            f" | {self._format_metrics_line(final_summary)}"
            f" | final_summary={self.memory.final_summary_path}"
        )
        return {
            "run_dir": str(self.memory.run_dir),
            "summary": final_summary,
        }

    def _generate_patient_case(self, condition: dict[str, Any]) -> str | None:
        prompt = build_patient_generation_prompt(condition)
        temperatures = [self.config.temperature, 0.0]
        for temperature in temperatures:
            patient_case = self.backend.generate(
                prompt=prompt,
                max_tokens=min(self.config.max_tokens, 400),
                temperature=temperature,
            ).strip()
            if patient_case:
                return patient_case
        return None

    def _record_success(
        self,
        attempted_patient_id: str,
        condition: dict[str, Any],
        training_lane: str,
        patient_case: str,
        doctor_answer: str,
        solved_on: str,
        validated_reflection_used: str | None,
    ) -> dict[str, Any]:
        self.successful_cases += 1
        self.success_count_by_training_lane[training_lane] += 1
        self.coverage_by_training_lane[training_lane].add(condition["condition_id"])
        if solved_on == "first_attempt":
            self.first_attempt_successes += 1
        elif solved_on == "retry":
            self.retry_successes += 1
            self.retry_success_count_by_training_lane[training_lane] += 1

        if condition.get("expert_curriculum", {}).get("must_not_miss") is True:
            self.must_not_miss_success_count += 1

        return self.memory.record_success(
            attempted_patient_id=attempted_patient_id,
            condition=condition,
            training_lane=training_lane,
            patient_case=patient_case,
            doctor_answer=doctor_answer,
            solved_on=solved_on,
            validated_reflection_used=validated_reflection_used,
        )

    def _build_summary(self) -> dict[str, Any]:
        return self.memory.build_metrics_snapshot(
            successful_cases=self.successful_cases,
            attempted_patients=self.attempted_patients,
            first_attempt_successes=self.first_attempt_successes,
            retry_successes=self.retry_successes,
            retry_attempts=self.retry_attempts,
            patient_generation_empty_count=self.patient_generation_empty_count,
            qc_discarded_patients=self.qc_discarded_patients,
            retry_fail_discards=self.retry_fail_discards,
            reflections_emitted_count=self.reflections_emitted_count,
            validated_reflection_count=self.validated_reflection_count,
            failed_reflection_count=self.failed_reflection_count,
            no_reflection_count=self.no_reflection_count,
            success_count_by_training_lane=dict(self.success_count_by_training_lane),
            retry_success_count_by_training_lane=dict(self.retry_success_count_by_training_lane),
            discard_count_by_training_lane=dict(self.discard_count_by_training_lane),
            coverage_by_training_lane=dict(self.coverage_by_training_lane),
            must_not_miss_success_count=self.must_not_miss_success_count,
        )

    def evaluate(self) -> dict[str, Any]:
        return self._build_summary()

    def _maybe_run_periodic_evaluation(self) -> None:
        summary = self._build_summary()
        if self.config.log_every > 0 and self.attempted_patients % self.config.log_every == 0:
            self._log(f"Metrics | {self._format_metrics_line(summary)}")
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
        self._log(
            "Periodic summary written"
            f" | attempts={self.attempted_patients}"
            f" | path={self.memory.eval_summary_path}"
        )
        if self.config.eval_dataset:
            eval_summary = self._run_periodic_dataset_evaluation()
            self.memory.append_heldout_eval_summary(eval_summary)
            self._log(
                "Held-out evaluation written"
                f" | accuracy={eval_summary['accuracy']}"
                f" | path={self.memory.run_dir / 'eval_summary.json'}"
            )

    def _maybe_run_success_milestone_evaluation(self, summary: dict[str, Any]) -> None:
        if not self.config.eval_dataset:
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
            eval_summary = self._run_periodic_dataset_evaluation(
                output_dir=self.memory.run_dir / f"eval_success_{milestone:06d}",
                trigger_successful_cases=milestone,
            )
            self.memory.append_heldout_eval_summary(eval_summary)
            self._log(
                "Held-out milestone evaluation written"
                f" | successful_cases={milestone}"
                f" | accuracy={eval_summary['accuracy']}"
                f" | path={self.memory.run_dir / f'eval_success_{milestone:06d}' / 'eval_summary.json'}"
            )

    def _format_metrics_line(self, summary: dict[str, Any]) -> str:
        return (
            f"attempts={summary['attempted_patients']}"
            f" successes={summary['successful_cases']}/{self.config.n_successful_cases}"
            f" first={summary['first_attempt_successes']}"
            f" retry_success={summary['retry_successes']}"
            f" retry_attempts={summary['retry_attempts']}"
            f" patient_empty={summary['patient_generation_empty_count']}"
            f" qc_discards={summary['qc_discarded_patients']}"
            f" reflections={summary['reflections_emitted_count']}"
            f" validated_reflections={summary['validated_reflection_count']}"
            f" failed_reflections={summary['failed_reflection_count']}"
            f" no_reflection={summary['no_reflection_count']}"
            f" retry_fail_discards={summary['retry_fail_discards']}"
        )

    def _log(self, message: str) -> None:
        if self.config.quiet:
            return
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    def _log_event(self, message: str) -> None:
        if self.config.verbose_events:
            self._log(message)

    def _run_periodic_dataset_evaluation(
        self,
        output_dir: Path | None = None,
        trigger_successful_cases: int | None = None,
    ) -> dict[str, Any]:
        evaluator = EvaluationRunner(
            backend=self.backend,
            config=EvaluationConfig(
                eval_dataset=self.config.eval_dataset,
                kb_path=self.config.kb_path,
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
                allow_eval_learning=self.config.allow_eval_learning,
                output_dir=str(output_dir or self.memory.run_dir),
                quiet=self.config.quiet,
            ),
            successful_case_records=list(self.memory.successful_cases),
            validated_reflection_records=list(self.memory.validated_reflections),
            successful_case_index=self.memory.success_index,
            validated_reflection_index=self.memory.reflection_index,
        )
        return evaluator.run(
            trigger_attempted_patients=self.attempted_patients,
            trigger_successful_cases=trigger_successful_cases,
        )
