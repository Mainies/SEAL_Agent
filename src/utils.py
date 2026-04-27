from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_file_exists(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Required path does not exist: {resolved}")
    return resolved


def load_json(path: str | Path) -> Any:
    resolved = ensure_file_exists(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = ensure_file_exists(path)
    records: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            cleaned = line.strip()
            if not cleaned:
                continue
            payload = json.loads(cleaned)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number} of {resolved}, got {type(payload).__name__}."
                )
            records.append(payload)
    return records


def write_json(path: str | Path, payload: Any) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True))
        handle.write("\n")


def pretty_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True)


def now_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "run"


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find JSON object in model output: {text!r}")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"Model output did not decode to a JSON object: {text!r}")
    return parsed


def coerce_binary_flag(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return 1
        if normalized in {"0", "false", "no"}:
            return 0
    raise ValueError(f"Expected {field_name} to be 0 or 1, got {value!r}")


def unique_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def humanize_condition_id(condition_id: str) -> str:
    trimmed = condition_id
    if condition_id.startswith("inf_"):
        pieces = condition_id.split("_", 2)
        if len(pieces) == 3:
            trimmed = pieces[2]
    return trimmed.replace("_", " ").strip().title()


def build_expert_lesson(condition: dict[str, Any]) -> str:
    curriculum = condition.get("expert_curriculum", {})
    patient_generation = condition.get("seal_diagnosis_task", {}).get("patient_generation", {})
    condition_name = condition["condition_name"]
    discriminators = unique_strings(curriculum.get("key_discriminators", []))
    lookalikes = unique_strings(curriculum.get("lookalike_differentials", []))
    traps = unique_strings(curriculum.get("diagnostic_traps", []))
    syndromes = unique_strings(patient_generation.get("syndrome_tags", []))

    syndrome_phrase = syndromes[0].replace("_", " ") if syndromes else "the dominant syndrome"
    discriminator_phrase = discriminators[0] if discriminators else "the highest-yield discriminator"
    lookalike_phrase = lookalikes[0] if lookalikes else "the nearest lookalike"
    trap_phrase = traps[0] if traps else None

    lesson = (
        f"General rule: in {syndrome_phrase} presentations, favor {condition_name} "
        f"when {discriminator_phrase} is present, and use it to separate the case from {lookalike_phrase}."
    )
    if trap_phrase:
        lesson += f" Avoid the trap of {trap_phrase}."
    return lesson


def classify_failure_mode(reflection: str | None) -> str:
    if not reflection:
        return "diagnostic_miss"
    lowered = reflection.lower()
    if "exposure" in lowered or "travel" in lowered or "zoonotic" in lowered:
        return "missed_exposure"
    if "host" in lowered or "immuno" in lowered or "healthcare" in lowered:
        return "missed_host_context"
    if "time course" in lowered or "timeline" in lowered:
        return "missed_time_course"
    if "lookalike" in lowered or "differential" in lowered or "trap" in lowered:
        return "lookalike_confusion"
    if "discriminator" in lowered or "clue" in lowered:
        return "missed_discriminator"
    return "diagnostic_miss"


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def tokenize_for_overlap(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        candidates = [values]
    elif isinstance(values, list):
        candidates = [str(value) for value in values]
    else:
        candidates = [str(values)]
    tokens: set[str] = set()
    for candidate in candidates:
        for token in re.findall(r"[a-z0-9]+", candidate.lower()):
            tokens.add(token)
    return tokens
