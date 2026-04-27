from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.prompts import build_judge_prompt
from src.utils import coerce_binary_flag, extract_json_object


@dataclass(frozen=True)
class JudgeResult:
    correct: bool
    reflection: str | None
    raw_output: str
    parse_error: str | None = None


class DiagnosisJudge:
    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def evaluate(
        self,
        condition: dict[str, Any],
        patient_case: str,
        doctor_answer: str,
    ) -> JudgeResult:
        prompt = build_judge_prompt(
            condition=condition,
            patient_case=patient_case,
            doctor_answer=doctor_answer,
        )
        response = self.backend.generate(prompt=prompt, max_tokens=220, temperature=0.0)
        try:
            payload = extract_json_object(response)
            correct = bool(coerce_binary_flag(payload.get("correct"), "correct"))
        except (TypeError, ValueError) as exc:
            return JudgeResult(
                correct=False,
                reflection=None,
                raw_output=response,
                parse_error=str(exc),
            )
        reflection = payload.get("reflection")
        if correct:
            reflection = None
        elif not isinstance(reflection, str) or not reflection.strip():
            reflection = None
        else:
            reflection = reflection.strip()
        return JudgeResult(correct=correct, reflection=reflection, raw_output=response)
