from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.prompts import build_patient_qc_prompt
from src.utils import coerce_binary_flag, extract_json_object


@dataclass(frozen=True)
class PatientQCResult:
    usable: bool
    reason: str | None


class PatientQC:
    def __init__(self, backend: Any, enabled: bool) -> None:
        self.backend = backend
        self.enabled = enabled

    def evaluate(self, condition: dict[str, Any], patient_case: str) -> PatientQCResult:
        if not self.enabled:
            return PatientQCResult(usable=True, reason=None)

        prompt = build_patient_qc_prompt(condition=condition, patient_case=patient_case)
        response = self.backend.generate(prompt=prompt, max_tokens=160, temperature=0.0)
        payload = extract_json_object(response)
        usable = bool(coerce_binary_flag(payload.get("usable"), "usable"))
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = None
        return PatientQCResult(usable=usable, reason=reason)
