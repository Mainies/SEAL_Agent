from __future__ import annotations

from typing import Any

from src.utils import humanize_condition_id, load_json, unique_strings


def _resolve_condition_name(condition: dict[str, Any]) -> str:
    existing = condition.get("condition_name")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    gold_diagnosis = (
        condition.get("seal_diagnosis_task", {})
        .get("judge", {})
        .get("gold_diagnosis")
    )
    if isinstance(gold_diagnosis, str) and gold_diagnosis.strip():
        return gold_diagnosis.strip()
    return humanize_condition_id(condition.get("condition_id", "unknown_condition"))


def load_conditions(kb_path: str) -> list[dict[str, Any]]:
    kb = load_json(kb_path)
    conditions = kb.get("core_infectious_disease_conditions")
    if not isinstance(conditions, list):
        raise ValueError("KB is missing core_infectious_disease_conditions")

    filtered_conditions: list[dict[str, Any]] = []
    for raw_condition in conditions:
        task = raw_condition.get("seal_diagnosis_task", {})
        curriculum = raw_condition.get("expert_curriculum", {})
        if task.get("task_inclusion") != "include":
            continue
        if curriculum.get("initial_loop_use") != "enabled":
            continue

        normalized = dict(raw_condition)
        normalized["condition_name"] = _resolve_condition_name(raw_condition)

        normalized_curriculum = dict(curriculum)
        normalized_curriculum["curriculum_lanes"] = unique_strings(
            normalized_curriculum.get("curriculum_lanes", [])
        )
        normalized_curriculum["retrieval_tags"] = unique_strings(
            normalized_curriculum.get("retrieval_tags", [])
        )
        normalized_curriculum["key_discriminators"] = unique_strings(
            normalized_curriculum.get("key_discriminators", [])
        )
        normalized_curriculum["lookalike_differentials"] = unique_strings(
            normalized_curriculum.get("lookalike_differentials", [])
        )

        normalized["expert_curriculum"] = normalized_curriculum
        filtered_conditions.append(normalized)

    if not filtered_conditions:
        raise RuntimeError("No conditions remained after task and initial-loop filtering.")
    return filtered_conditions


def load_sampler_config(sampler_config_path: str) -> dict[str, Any]:
    config = load_json(sampler_config_path)
    active_lanes = config.get("active_lanes_initial")
    if not isinstance(active_lanes, dict) or not active_lanes:
        raise ValueError("Sampler config is missing active_lanes_initial")
    return config


def load_condition_index(kb_path: str) -> dict[str, dict[str, Any]]:
    return {
        condition["condition_id"]: condition
        for condition in load_conditions(kb_path)
    }
