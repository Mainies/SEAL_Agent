from __future__ import annotations

from typing import Any

from src.utils import pretty_json


def _patient_generation_payload(condition: dict[str, Any]) -> dict[str, Any]:
    patient_generation = condition.get("seal_diagnosis_task", {}).get("patient_generation", {})
    return {
        "condition_id": condition["condition_id"],
        "condition_name": condition["condition_name"],
        "patient_generation": {
            "body_system": patient_generation.get("body_system", []),
            "time_course_hint": patient_generation.get("time_course_hint"),
            "symptoms_to_include": patient_generation.get("symptoms_to_include", []),
            "syndrome_tags": patient_generation.get("syndrome_tags", []),
            "exposure_clues": patient_generation.get("exposure_clues", []),
            "host_factor_clues": patient_generation.get("host_factor_clues", []),
            "objective_or_discriminating_clues": patient_generation.get(
                "objective_or_discriminating_clues", []
            ),
            "minimum_case_requirements": patient_generation.get(
                "minimum_case_requirements", []
            ),
        },
    }


def _expert_profile_payload(condition: dict[str, Any]) -> dict[str, Any]:
    expert_profile = condition["expert_curriculum"]
    return {
        "primary_training_lane": expert_profile.get("primary_training_lane"),
        "curriculum_lanes": expert_profile.get("curriculum_lanes", []),
        "expert_value": expert_profile.get("expert_value"),
        "must_not_miss": expert_profile.get("must_not_miss"),
        "key_discriminators": expert_profile.get("key_discriminators", []),
        "lookalike_differentials": expert_profile.get("lookalike_differentials", []),
        "diagnostic_traps": expert_profile.get("diagnostic_traps", []),
        "expert_reason": expert_profile.get("expert_reason"),
    }


def build_patient_generation_prompt(condition: dict[str, Any]) -> str:
    return f"""You are generating a synthetic patient vignette for a closed infectious disease diagnosis simulation.

The goal is to train infectious diseases expert diagnostic reasoning, not to reflect general-population prevalence.

Hidden condition:
{pretty_json(_patient_generation_payload(condition))}

Expert curriculum profile:
{pretty_json(_expert_profile_payload(condition))}

Generate a concise but diagnostically useful patient case.

Rules:
- Do not reveal the disease name.
- Do not reveal the pathogen name.
- Do not reveal the condition ID.
- Include age, sex, chief complaint, symptom timeline, relevant exposure or host factor, relevant negatives, and basic objective clues.
- Because this is the diagnosis task, include enough already-available examination, laboratory, imaging, microbiology, pathology, or other test results to support a diagnosis when they are relevant.
- Supportive test results may be specific, but they must not explicitly name the hidden disease, pathogen, or condition ID.
- Do not write meta-instructions to the doctor, such as "the team must distinguish" or "this points toward".
- Finish the vignette as a complete clinical note; do not end mid-sentence or with an unfinished diagnostic phrase.
- Include at least one key discriminator from the expert profile when possible.
- If lookalike differentials are listed, create a case where the gold diagnosis is distinguishable but not artificially obvious.
- Keep it focused and clinically dense; concise is better than exhaustive.
- Return plain text only.
"""


def build_patient_qc_prompt(condition: dict[str, Any], patient_case: str) -> str:
    qc_rules = condition.get("seal_diagnosis_task", {}).get("patient_qc", {})
    return f"""You are doing binary patient-case quality control for a closed infectious disease diagnosis simulation.

Hidden condition:
{pretty_json(_patient_generation_payload(condition))}

Expert curriculum profile:
{pretty_json(_expert_profile_payload(condition))}

Patient QC criteria:
{pretty_json(qc_rules)}

Generated patient case:
{patient_case}

Mark unusable if:
- diagnosis, pathogen, or condition ID is directly leaked
- the vignette is unrelated to the condition
- the vignette lacks enough diagnostic information
- the vignette contradicts the key host or exposure context
- the vignette ends mid-sentence or contains an unfinished phrase that implies the hidden answer

Do not mark unusable merely because the case includes supportive examination, laboratory,
imaging, microbiology, pathology, or other test results. In the diagnosis task, those
results are expected. Supportive results are only leakage if they explicitly name the
hidden disease, pathogen, or condition ID.

Return JSON only:
{{
  "usable": 1 or 0,
  "reason": null or "short reason"
}}
"""


def build_doctor_prompt(
    memory_context: str,
    patient_case: str,
    previous_answer: str | None = None,
    provisional_reflection: str | None = None,
) -> str:
    retry_block = ""
    if previous_answer and provisional_reflection:
        retry_block = f"""
Previous answer for this same patient:
{previous_answer}

Provisional reflection for this same patient:
{provisional_reflection}

Use the reflection to revise the diagnosis without adding treatment or management content.
"""

    return f"""You are an infectious diseases doctor-agent in a closed diagnosis simulation.

Your goal is expert diagnosis, not generic population-level guessing.

Previous successful cases and validated diagnostic reflections:
{memory_context}

Current patient case:
{patient_case}
{retry_block}
Task:
Give the most likely infectious diagnosis and a short diagnostic justification.

Return using exactly these headings:

Diagnosis:
Differential:
Diagnostic justification:
Uncertainty:

Rules:
- Focus only on diagnosis.
- Do not provide treatment.
- Do not provide drug doses.
- Do not provide IPC, public-health, antimicrobial stewardship, source-control, or management plans.
- Use syndrome, time course, host context, exposure, and key discriminators.
- Treat prior cases and reflections as reusable diagnostic rules, not as exact templates to copy.
- Keep the answer concise.
"""


def build_judge_prompt(
    condition: dict[str, Any],
    patient_case: str,
    doctor_answer: str,
) -> str:
    judge_config = condition.get("seal_diagnosis_task", {}).get("judge", {})
    judge_payload = {
        "condition_id": condition["condition_id"],
        "condition_name": condition["condition_name"],
        "gold_diagnosis": judge_config.get("gold_diagnosis"),
        "accepted_answers": judge_config.get("accepted_answers", []),
        "parent_disease": judge_config.get("parent_disease"),
        "subtype_or_variant": judge_config.get("subtype_or_variant"),
        "correct_if": judge_config.get("correct_if", []),
        "incorrect_if": judge_config.get("incorrect_if", []),
        "expert_curriculum": _expert_profile_payload(condition),
    }
    return f"""You are the hidden judge for a closed infectious disease diagnosis simulation.

The judge is binary and diagnosis-only.

Hidden condition:
{pretty_json(judge_payload)}

Generated patient case:
{patient_case}

Doctor answer:
{doctor_answer}

Judging rules:
- Mark correct if the answer gives the exact diagnosis, accepted synonym, or clinically acceptable subtype or parent match for this generated case.
- Mark correct if the exact diagnosis is in the differential and the justification clearly tracks the correct syndrome and key discriminator.
- Mark incorrect if the answer follows the intended diagnostic trap, misses the key exposure or host clue, or gives a diagnosis that would derail the case.
- Do not judge treatment, IPC, public health, stewardship, source control, or management.
- If incorrect, the reflection must be a concise reusable diagnostic rule that identifies the missed discriminator or wrong lookalike.
- Do not include hidden chain-of-thought.

Return JSON only:
{{
  "correct": 1 or 0,
  "reflection": null if correct, otherwise "one concise reusable diagnostic rule"
}}
"""


def build_eval_kb_context(condition: dict[str, Any]) -> str:
    kb_payload = {
        "condition_id": condition["condition_id"],
        "condition_name": condition["condition_name"],
        "expert_curriculum": _expert_profile_payload(condition),
        "patient_generation": _patient_generation_payload(condition)["patient_generation"],
    }
    return pretty_json(kb_payload)


def build_eval_prompt(
    question: dict[str, Any],
    memory_context: str,
    kb_context: str | None,
) -> str:
    kb_block = ""
    if kb_context:
        kb_block = f"""
Relevant compact condition metadata:
{kb_context}
"""
    return f"""You are an infectious diseases diagnosis expert.

Use the provided case experience only if it is relevant.
Treat prior cases and reflections as general diagnostic rules, not exact case templates.
{kb_block}
Prior successful cases and validated reflections:
{memory_context}

Question:
{question["question"]}

Task:
Choose the single best diagnosis.

Return exactly:
Answer: <letter>
Diagnosis: <diagnosis name>
Justification: <one or two sentences>

Rules:
- Focus only on diagnosis.
- Do not provide treatment, IPC, public-health, stewardship, or management.
- Use syndrome, time course, host context, exposure, and key discriminators.
- If options are provided, choose one option letter.
"""
