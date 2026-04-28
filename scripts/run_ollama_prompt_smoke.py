from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.backends import OllamaBackend
from src.examination import (
    build_examination_case_generation_prompt,
    build_examination_doctor_prompt,
    build_examination_judge_prompt,
    build_examination_qc_prompt,
    load_examination_eval_items,
    load_examination_modules,
    parse_examination_judge_response,
    parse_examination_qc_response,
)
from src.kb import load_conditions
from src.prompts import (
    build_doctor_prompt,
    build_eval_prompt,
    build_judge_prompt,
    build_patient_generation_prompt,
    build_patient_qc_prompt,
)
from src.utils import coerce_binary_flag, extract_json_object, load_json, slugify

NO_DIAGNOSIS_MEMORY = "No previous successful cases or validated reflections available."
NO_EXAM_MEMORY = (
    "No previous successful examination cases or validated examination reflections available."
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a tiny Ollama prompt smoke and save prompt/output pairs for "
            "diagnosis and examination-selection stages."
        )
    )
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--per-task", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=positive_int, default=160)
    parser.add_argument("--case-max-tokens", type=positive_int, default=260)
    parser.add_argument("--qc-max-tokens", type=positive_int, default=180)
    parser.add_argument("--judge-max-tokens", type=positive_int, default=180)
    parser.add_argument(
        "--dx-kb-path",
        default="data/infectious_diseases_seal_expert_rcl_kb_v6_compact.json",
    )
    parser.add_argument(
        "--dx-eval-path",
        default="evaluation/id_expert_hard_eval_200.json",
    )
    parser.add_argument(
        "--exam-kb-path",
        default="data/id_examination_extension_kb_v1.json",
    )
    parser.add_argument(
        "--exam-eval-path",
        default="evaluation/id_expert_examination_eval_200.json",
    )
    parser.add_argument("--output-root", default="smoke_tests")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--continue-on-qc-fail",
        action="store_true",
        help=(
            "Continue to doctor/judge stages when QC marks a generated case "
            "unusable. Default behavior matches the real loop and stops that case."
        ),
    )
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help=(
            "Also save held-out eval doctor prompts/outputs for diagnosis and "
            "examination. Off by default to keep the smoke faster."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.dx_kb_path = str(resolve_repo_path(args.dx_kb_path))
    args.dx_eval_path = str(resolve_repo_path(args.dx_eval_path))
    args.exam_kb_path = str(resolve_repo_path(args.exam_kb_path))
    args.exam_eval_path = str(resolve_repo_path(args.exam_eval_path))
    args.output_root = str(resolve_output_root(args.output_root))

    run_name = args.run_name or f"ollama_prompt_smoke_{timestamp()}"
    output_dir = Path(args.output_root) / slugify(run_name)
    if output_dir.exists():
        raise FileExistsError(f"Smoke output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    backend = OllamaBackend(model=args.model, host=args.ollama_host)
    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []

    metadata = {
        "run_name": run_name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": args.model,
        "ollama_host": args.ollama_host,
        "per_task": args.per_task,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "case_max_tokens": args.case_max_tokens,
        "qc_max_tokens": args.qc_max_tokens,
        "judge_max_tokens": args.judge_max_tokens,
        "include_eval": args.include_eval,
        "continue_on_qc_fail": args.continue_on_qc_fail,
    }
    write_json(output_dir / "smoke_metadata.json", metadata)

    diagnosis_conditions = choose_items(load_conditions(args.dx_kb_path), args.per_task, rng)
    examination_modules = choose_items(
        load_examination_modules(args.exam_kb_path), args.per_task, rng
    )

    for index, condition in enumerate(diagnosis_conditions, start=1):
        run_diagnosis_case(
            backend=backend,
            args=args,
            condition=condition,
            case_index=index,
            output_dir=output_dir / "diagnosis" / case_folder_name(index, condition),
            records=records,
        )

    for index, module in enumerate(examination_modules, start=1):
        run_examination_case(
            backend=backend,
            args=args,
            module=module,
            case_index=index,
            output_dir=output_dir / "examination_selection" / case_folder_name(index, module),
            records=records,
        )

    if args.include_eval:
        run_eval_prompt_smokes(
            backend=backend,
            args=args,
            output_dir=output_dir / "heldout_eval_prompts",
            records=records,
        )

    write_json(output_dir / "summary.json", {"records": records})
    print(f"Smoke prompts and outputs written to: {output_dir}")


def run_diagnosis_case(
    backend: OllamaBackend,
    args: argparse.Namespace,
    condition: dict[str, Any],
    case_index: int,
    output_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "case_metadata.json",
        {
            "task_type": "diagnosis",
            "case_index": case_index,
            "condition_id": condition.get("condition_id"),
            "condition_name": condition.get("condition_name"),
        },
    )

    patient_case = run_stage(
        backend=backend,
        args=args,
        task_type="diagnosis",
        case_index=case_index,
        stage_name="01_patient_generation",
        prompt=build_patient_generation_prompt(condition),
        output_dir=output_dir,
        records=records,
        max_tokens=args.case_max_tokens,
        metadata={
            "condition_id": condition.get("condition_id"),
            "condition_name": condition.get("condition_name"),
        },
    )

    qc_output = run_stage(
        backend=backend,
        args=args,
        task_type="diagnosis",
        case_index=case_index,
        stage_name="02_patient_qc",
        prompt=build_patient_qc_prompt(condition, patient_case),
        output_dir=output_dir,
        records=records,
        max_tokens=args.qc_max_tokens,
        parser=parse_qc_json,
    )
    qc_result = parse_qc_json(qc_output)
    if should_stop_after_qc(args, qc_result):
        record_skipped_after_qc(
            task_type="diagnosis",
            case_index=case_index,
            output_dir=output_dir,
            records=records,
            qc_result=qc_result,
        )
        return

    doctor_answer = run_stage(
        backend=backend,
        args=args,
        task_type="diagnosis",
        case_index=case_index,
        stage_name="03_doctor_answer",
        prompt=build_doctor_prompt(
            memory_context=NO_DIAGNOSIS_MEMORY,
            patient_case=patient_case,
        ),
        output_dir=output_dir,
        records=records,
        max_tokens=args.max_tokens,
        metadata={"qc_output_preview": qc_output[:240]},
    )

    run_stage(
        backend=backend,
        args=args,
        task_type="diagnosis",
        case_index=case_index,
        stage_name="04_hidden_judge",
        prompt=build_judge_prompt(condition, patient_case, doctor_answer),
        output_dir=output_dir,
        records=records,
        max_tokens=args.judge_max_tokens,
        parser=parse_diagnosis_judge_json,
    )


def run_examination_case(
    backend: OllamaBackend,
    args: argparse.Namespace,
    module: dict[str, Any],
    case_index: int,
    output_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "case_metadata.json",
        {
            "task_type": "examination_selection",
            "case_index": case_index,
            "condition_id": module.get("condition_id"),
            "condition_name": module.get("condition_name"),
            "exam_focus": module.get("exam_focus", []),
        },
    )

    patient_case = run_stage(
        backend=backend,
        args=args,
        task_type="examination_selection",
        case_index=case_index,
        stage_name="01_case_generation",
        prompt=build_examination_case_generation_prompt(module),
        output_dir=output_dir,
        records=records,
        max_tokens=args.case_max_tokens,
        metadata={
            "condition_id": module.get("condition_id"),
            "condition_name": module.get("condition_name"),
        },
    )

    qc_output = run_stage(
        backend=backend,
        args=args,
        task_type="examination_selection",
        case_index=case_index,
        stage_name="02_examination_qc",
        prompt=build_examination_qc_prompt(module, patient_case),
        output_dir=output_dir,
        records=records,
        max_tokens=args.qc_max_tokens,
        parser=parse_examination_qc_json,
    )
    qc_result = parse_examination_qc_json(qc_output)
    if should_stop_after_qc(args, qc_result):
        record_skipped_after_qc(
            task_type="examination_selection",
            case_index=case_index,
            output_dir=output_dir,
            records=records,
            qc_result=qc_result,
        )
        return

    doctor_answer = run_stage(
        backend=backend,
        args=args,
        task_type="examination_selection",
        case_index=case_index,
        stage_name="03_doctor_answer",
        prompt=build_examination_doctor_prompt(
            memory_context=NO_EXAM_MEMORY,
            patient_case=patient_case,
        ),
        output_dir=output_dir,
        records=records,
        max_tokens=args.max_tokens,
        metadata={"qc_output_preview": qc_output[:240]},
    )

    run_stage(
        backend=backend,
        args=args,
        task_type="examination_selection",
        case_index=case_index,
        stage_name="04_hidden_judge",
        prompt=build_examination_judge_prompt(
            module_or_item=module,
            patient_case=patient_case,
            doctor_answer=doctor_answer,
            hidden_gold=exam_hidden_gold(module),
        ),
        output_dir=output_dir,
        records=records,
        max_tokens=args.judge_max_tokens,
        parser=parse_examination_judge_json,
    )


def run_eval_prompt_smokes(
    backend: OllamaBackend,
    args: argparse.Namespace,
    output_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    dx_questions = choose_items(load_json(args.dx_eval_path), args.per_task, random.Random(args.seed))
    exam_items = choose_items(
        load_examination_eval_items(args.exam_eval_path),
        args.per_task,
        random.Random(args.seed + 1),
    )

    for index, question in enumerate(dx_questions, start=1):
        run_stage(
            backend=backend,
            args=args,
            task_type="diagnosis_eval",
            case_index=index,
            stage_name="01_eval_doctor_answer",
            prompt=build_eval_prompt(
                question=question,
                memory_context=NO_DIAGNOSIS_MEMORY,
                kb_context=None,
            ),
            output_dir=output_dir / "diagnosis" / f"eval_{index:03d}",
            records=records,
            max_tokens=args.max_tokens,
            metadata={"question_id": question.get("question_id")},
        )

    for index, item in enumerate(exam_items, start=1):
        run_stage(
            backend=backend,
            args=args,
            task_type="examination_eval",
            case_index=index,
            stage_name="01_eval_doctor_answer",
            prompt=build_examination_doctor_prompt(
                memory_context=NO_EXAM_MEMORY,
                patient_case=str(item.get("presenting_case", "")),
                question=str(item.get("question", "")),
            ),
            output_dir=output_dir / "examination_selection" / f"eval_{index:03d}",
            records=records,
            max_tokens=args.max_tokens,
            metadata={"question_id": item.get("question_id")},
        )


def run_stage(
    backend: OllamaBackend,
    args: argparse.Namespace,
    task_type: str,
    case_index: int,
    stage_name: str,
    prompt: str,
    output_dir: Path,
    records: list[dict[str, Any]],
    max_tokens: int,
    parser: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / f"{stage_name}_prompt.txt"
    output_path = output_dir / f"{stage_name}_output.txt"
    stage_metadata_path = output_dir / f"{stage_name}_metadata.json"

    prompt_path.write_text(prompt, encoding="utf-8")
    started = time.monotonic()
    output = backend.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=args.temperature,
    )
    elapsed_seconds = round(time.monotonic() - started, 3)
    output_path.write_text(output, encoding="utf-8")

    parsed: dict[str, Any] | None = None
    if parser is not None:
        parsed = parser(output)

    stage_metadata = {
        "task_type": task_type,
        "case_index": case_index,
        "stage_name": stage_name,
        "prompt_path": str(prompt_path),
        "output_path": str(output_path),
        "max_tokens": max_tokens,
        "elapsed_seconds": elapsed_seconds,
        "output_chars": len(output),
    }
    if metadata:
        stage_metadata["metadata"] = metadata
    if parsed is not None:
        stage_metadata["parsed"] = parsed
    write_json(stage_metadata_path, stage_metadata)
    records.append(stage_metadata)
    print(
        f"{task_type} case={case_index:03d} stage={stage_name} "
        f"chars={len(output)} elapsed={elapsed_seconds}s",
        flush=True,
    )
    return output


def should_stop_after_qc(args: argparse.Namespace, qc_result: dict[str, Any]) -> bool:
    if args.continue_on_qc_fail:
        return False
    return not (qc_result.get("parse_ok") is True and qc_result.get("usable") is True)


def record_skipped_after_qc(
    task_type: str,
    case_index: int,
    output_dir: Path,
    records: list[dict[str, Any]],
    qc_result: dict[str, Any],
) -> None:
    stage_metadata = {
        "task_type": task_type,
        "case_index": case_index,
        "stage_name": "03_skipped_after_qc",
        "skipped": True,
        "reason": "qc_failed_or_malformed",
        "qc_result": qc_result,
    }
    write_json(output_dir / "03_skipped_after_qc_metadata.json", stage_metadata)
    records.append(stage_metadata)
    print(
        f"{task_type} case={case_index:03d} stage=03_skipped_after_qc "
        "reason=qc_failed_or_malformed",
        flush=True,
    )


def parse_qc_json(output: str) -> dict[str, Any]:
    try:
        payload = extract_json_object(output)
        return {
            "parse_ok": True,
            "usable": bool(coerce_binary_flag(payload.get("usable"), "usable")),
            "reason": payload.get("reason"),
        }
    except (TypeError, ValueError) as exc:
        return {"parse_ok": False, "parse_error": str(exc)}


def parse_diagnosis_judge_json(output: str) -> dict[str, Any]:
    try:
        payload = extract_json_object(output)
        return {
            "parse_ok": True,
            "correct": bool(coerce_binary_flag(payload.get("correct"), "correct")),
            "reflection": payload.get("reflection"),
        }
    except (TypeError, ValueError) as exc:
        return {"parse_ok": False, "parse_error": str(exc)}


def parse_examination_qc_json(output: str) -> dict[str, Any]:
    result = parse_examination_qc_response(output)
    return {
        "parse_ok": result.parse_error is None,
        "usable": result.usable,
        "reason": result.reason,
        "parse_error": result.parse_error,
    }


def parse_examination_judge_json(output: str) -> dict[str, Any]:
    result = parse_examination_judge_response(output)
    return {
        "parse_ok": result.parse_error is None,
        "correct": result.correct,
        "reflection": result.reflection,
        "essential_hits": result.essential_hits,
        "dangerous_misses": result.dangerous_misses,
        "parse_error": result.parse_error,
    }


def exam_hidden_gold(module: dict[str, Any]) -> dict[str, Any]:
    return {
        "core_exam_or_history": as_list(module.get("core_exam_or_history")),
        "essential_examination_or_tests": as_list(
            module.get("essential_examination_or_tests")
        ),
        "accepted_alternatives": as_list(module.get("accepted_alternatives")),
        "conditional_or_second_line_tests": as_list(
            module.get("conditional_or_second_line_tests")
        ),
        "avoid_or_low_value": as_list(module.get("avoid_or_low_value")),
        "dangerous_misses": as_list(module.get("dangerous_misses")),
    }


def as_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def choose_items(items: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count >= len(items):
        return list(items)
    return rng.sample(items, count)


def case_folder_name(index: int, record: dict[str, Any]) -> str:
    condition_id = str(record.get("condition_id", "unknown"))
    condition_name = str(record.get("condition_name", "unknown"))
    return f"case_{index:03d}_{slugify(condition_id)}_{slugify(condition_name)}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, sort_keys=True)
        handle.write("\n")


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def resolve_output_root(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
