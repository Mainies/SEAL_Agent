from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.backends import HuggingFaceBackend, OllamaBackend
from src.examination import (
    EXAMINATION_TASK_TYPE,
    ExaminationLoop,
    ExaminationLoopConfig,
)
from src.loop import SimulationConfig, SimulationLoop
from src.semantic_retrieval import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_EMBEDDING_MODEL,
    RETRIEVAL_MODES,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be a non-negative integer.")
    return parsed


def success_milestones(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    milestones: list[int] = []
    for raw_piece in value.split(","):
        piece = raw_piece.strip()
        if not piece:
            continue
        parsed = positive_int(piece)
        milestones.append(parsed)
    return tuple(milestones)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal expert-oriented infectious disease SEAL/RCL training loop."
    )
    parser.add_argument(
        "--task_type",
        "--task-type",
        choices=["diagnosis", EXAMINATION_TASK_TYPE],
        default="diagnosis",
    )
    parser.add_argument("--backend", required=True, choices=["ollama", "hf"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--kb_path")
    parser.add_argument("--sampler_config")
    parser.add_argument("--exam_kb_path", "--exam-kb-path")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        help="Optional exact output directory for training artifacts.",
    )
    parser.add_argument(
        "--current_run",
        "--current-run",
        action="store_true",
        help="Use <runs_root>/current_run for training artifacts and resume existing memory.",
    )
    parser.add_argument(
        "--current_run_dir",
        "--current-run-dir",
        default="current_run",
        help="Directory name/path used by --current_run. Relative paths are under --runs_root.",
    )
    parser.add_argument(
        "--n_successful_cases",
        "--target-successful-cases",
        required=True,
        type=positive_int,
    )
    parser.add_argument("--run_name", "--run-name")
    parser.add_argument(
        "--eval_every",
        type=non_negative_int,
        default=0,
        help="When --run_evaluation is enabled, run attempted-patient evaluation every N attempts. Set 0 to disable.",
    )
    parser.add_argument(
        "--run_evaluation",
        action="store_true",
        help="Enable periodic evaluation summaries during the loop.",
    )
    parser.add_argument(
        "--eval_dataset",
        help="Optional held-out evaluation dataset to run every --eval_every attempted patients when --run_evaluation is enabled.",
    )
    parser.add_argument("--eval_file", "--eval-file")
    parser.add_argument(
        "--eval_mode",
        "--eval-mode",
        choices=["no_memory", "with_memory", "kb_only", "memory_only"],
        help="Optional held-out evaluation mode for periodic eval runs.",
    )
    parser.add_argument(
        "--eval_success_milestones",
        type=success_milestones,
        default=(),
        help="Comma-separated successful-case checkpoints for held-out evaluation, e.g. 5000,10000,15000,20000.",
    )
    parser.add_argument(
        "--allow_eval_learning",
        action="store_true",
        help="Reserved flag for allowing evaluation examples to update learning memory. Default behavior keeps eval read-only.",
    )
    parser.add_argument(
        "--log_every",
        type=positive_int,
        default=1,
        help="Print running metrics every N attempted patients.",
    )
    parser.add_argument("--runs_root", "--runs-root", default="runs")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress running progress logs.",
    )
    parser.add_argument(
        "--verbose_events",
        action="store_true",
        help="Print every sampled patient, success, retry, and discard event.",
    )

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", "--max-tokens", type=positive_int, default=768)
    parser.add_argument("--eval_limit", "--eval-limit", type=positive_int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n_success_memory", type=non_negative_int, default=3)
    parser.add_argument("--n_reflection_memory", type=non_negative_int, default=4)
    parser.add_argument(
        "--retrieval_mode",
        "--retrieval-mode",
        choices=sorted(RETRIEVAL_MODES),
        default="semantic",
        help="Memory retrieval mode: semantic cosine retrieval or legacy tag overlap.",
    )
    parser.add_argument("--embedding_model", "--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding_device", "--embedding-device", default=DEFAULT_EMBEDDING_DEVICE)
    parser.add_argument(
        "--embedding_batch_size",
        "--embedding-batch-size",
        type=positive_int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
    )
    parser.add_argument("--ollama_host", "--ollama-host", default="http://localhost:11434")
    parser.add_argument("--patient_qc", action="store_true")
    return parser


def build_backend(args: argparse.Namespace) -> Any:
    if args.backend == "ollama":
        return OllamaBackend(model=args.model, host=args.ollama_host)
    if args.backend == "hf":
        return HuggingFaceBackend(model=args.model)
    raise ValueError(f"Unsupported backend: {args.backend}")


def resolve_current_run_dir(args: argparse.Namespace) -> str | None:
    if not args.current_run:
        return None
    current_run_dir = Path(args.current_run_dir)
    if current_run_dir.is_absolute():
        return str(current_run_dir)
    return str(Path(args.runs_root) / current_run_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    backend = build_backend(args)
    if args.current_run and args.output_dir:
        parser.error("--current_run and --output_dir cannot be used together")
    output_dir = resolve_current_run_dir(args) or args.output_dir
    run_name = args.run_name or (Path(output_dir).name if output_dir else None)

    if args.task_type == EXAMINATION_TASK_TYPE:
        if not run_name:
            parser.error("--run_name is required unless --output_dir is provided")
        if not args.exam_kb_path:
            parser.error("--exam_kb_path is required when --task_type examination_selection")
        eval_file = args.eval_file or args.eval_dataset
        config = ExaminationLoopConfig(
            backend=args.backend,
            model=args.model,
            exam_kb_path=args.exam_kb_path,
            n_successful_cases=args.n_successful_cases,
            run_name=run_name,
            eval_every=args.eval_every,
            run_evaluation=args.run_evaluation,
            eval_file=eval_file,
            eval_mode=args.eval_mode,
            eval_limit=args.eval_limit,
            eval_success_milestones=args.eval_success_milestones,
            log_every=args.log_every,
            verbose_events=args.verbose_events,
            quiet=args.quiet,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            n_success_memory=args.n_success_memory,
            n_reflection_memory=args.n_reflection_memory,
            retrieval_mode=args.retrieval_mode,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            runs_root=args.runs_root,
            output_dir=output_dir,
            resume_existing=args.current_run,
        )
        loop = ExaminationLoop(backend=backend, config=config)
        result = loop.run()
        summary = result["summary"]
        print(f"Run directory: {result['run_dir']}")
        print(f"Successful cases: {summary['successful_cases']}")
        print(f"Attempted patients: {summary['attempted_patients']}")
        print(f"First-attempt successes: {summary['first_attempt_successes']}")
        print(f"Retry successes: {summary['retry_successes']}")
        print(f"QC discarded patients: {summary['qc_discarded_patients']}")
        print(f"Retry-fail discards: {summary['retry_fail_discards']}")
        return

    if not args.kb_path:
        parser.error("--kb_path is required when --task_type diagnosis")
    if not args.sampler_config:
        parser.error("--sampler_config is required when --task_type diagnosis")
    if not run_name:
        parser.error("--run_name is required when --task_type diagnosis")

    config = SimulationConfig(
        backend=args.backend,
        model=args.model,
        kb_path=args.kb_path,
        sampler_config=args.sampler_config,
        n_successful_cases=args.n_successful_cases,
        run_name=run_name,
        eval_every=args.eval_every,
        run_evaluation=args.run_evaluation,
        eval_dataset=args.eval_dataset,
        eval_mode=args.eval_mode,
        eval_limit=args.eval_limit,
        eval_success_milestones=args.eval_success_milestones,
        allow_eval_learning=args.allow_eval_learning,
        log_every=args.log_every,
        verbose_events=args.verbose_events,
        quiet=args.quiet,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        n_success_memory=args.n_success_memory,
        n_reflection_memory=args.n_reflection_memory,
        retrieval_mode=args.retrieval_mode,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        ollama_host=args.ollama_host,
        patient_qc=args.patient_qc,
        runs_root=args.runs_root,
        output_dir=output_dir,
        resume_existing=args.current_run,
    )

    loop = SimulationLoop(backend=backend, config=config)
    result = loop.run()
    summary = result["summary"]

    print(f"Run directory: {result['run_dir']}")
    print(f"Successful cases: {summary['successful_cases']}")
    print(f"Attempted patients: {summary['attempted_patients']}")
    print(f"First-attempt successes: {summary['first_attempt_successes']}")
    print(f"Retry successes: {summary['retry_successes']}")
    print(f"QC discarded patients: {summary['qc_discarded_patients']}")
    print(f"Retry-fail discards: {summary['retry_fail_discards']}")


if __name__ == "__main__":
    main()
