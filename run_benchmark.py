from __future__ import annotations

import argparse
from typing import Any

from src.backends import HuggingFaceBackend, OllamaBackend
from src.evaluation import EvaluationConfig, EvaluationRunner
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
        if piece:
            milestones.append(positive_int(piece))
    return tuple(milestones)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-process baseline plus disease-only SEAL/RCL benchmark."
    )
    parser.add_argument("--backend", required=True, choices=["ollama", "hf"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--kb_path", required=True)
    parser.add_argument("--sampler_config", required=True)
    parser.add_argument("--eval_dataset", required=True)
    parser.add_argument("--baseline_run_name", required=True)
    parser.add_argument("--train_run_name", required=True)
    parser.add_argument("--n_successful_cases", required=True, type=positive_int)
    parser.add_argument("--eval_success_milestones", type=success_milestones, default=())
    parser.add_argument("--eval_every", type=non_negative_int, default=0)
    parser.add_argument("--log_every", type=positive_int, default=10)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose_events", action="store_true")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=positive_int, default=260)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n_success_memory", type=non_negative_int, default=3)
    parser.add_argument("--n_reflection_memory", type=non_negative_int, default=4)
    parser.add_argument(
        "--retrieval_mode",
        choices=sorted(RETRIEVAL_MODES),
        default="semantic",
    )
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding_device", default=DEFAULT_EMBEDDING_DEVICE)
    parser.add_argument(
        "--embedding_batch_size",
        type=positive_int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
    )
    parser.add_argument("--ollama_host", default="http://localhost:11434")
    parser.add_argument("--patient_qc", action="store_true")
    return parser


def build_backend(args: argparse.Namespace) -> Any:
    if args.backend == "ollama":
        return OllamaBackend(model=args.model, host=args.ollama_host)
    if args.backend == "hf":
        return HuggingFaceBackend(model=args.model)
    raise ValueError(f"Unsupported backend: {args.backend}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    backend = build_backend(args)

    baseline = EvaluationRunner(
        backend=backend,
        config=EvaluationConfig(
            eval_dataset=args.eval_dataset,
            kb_path=args.kb_path,
            run_name=args.baseline_run_name,
            backend=args.backend,
            model=args.model,
            eval_mode="no_memory",
            n_success_memory=args.n_success_memory,
            n_reflection_memory=args.n_reflection_memory,
            retrieval_mode=args.retrieval_mode,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            quiet=args.quiet,
        ),
    )
    baseline_summary = baseline.run(trigger_successful_cases=0)
    print(f"Baseline evaluation directory: {baseline.output_dir}")
    print(f"Baseline accuracy: {baseline_summary['accuracy']}")

    loop = SimulationLoop(
        backend=backend,
        config=SimulationConfig(
            backend=args.backend,
            model=args.model,
            kb_path=args.kb_path,
            sampler_config=args.sampler_config,
            n_successful_cases=args.n_successful_cases,
            run_name=args.train_run_name,
            eval_every=args.eval_every,
            run_evaluation=True,
            eval_dataset=args.eval_dataset,
            eval_mode="with_memory",
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
            ollama_host=args.ollama_host,
            patient_qc=args.patient_qc,
        ),
    )
    result = loop.run()
    summary = result["summary"]
    print(f"Training run directory: {result['run_dir']}")
    print(f"Successful cases: {summary['successful_cases']}")
    print(f"Attempted patients: {summary['attempted_patients']}")
    print(f"Final successful-case memory size: {summary['successful_case_memory_size']}")
    print(f"Final reflection memory size: {summary['validated_reflection_memory_size']}")


if __name__ == "__main__":
    main()
