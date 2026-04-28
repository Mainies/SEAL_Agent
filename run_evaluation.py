from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.backends import HuggingFaceBackend, OllamaBackend
from src.evaluation import EvaluationConfig, EvaluationRunner
from src.examination import (
    EXAMINATION_TASK_TYPE,
    ExaminationEvalConfig,
    ExaminationEvaluationRunner,
)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Held-out evaluation runner for diagnosis or text-only examination selection."
    )
    parser.add_argument(
        "--task_type",
        "--task-type",
        choices=["diagnosis", EXAMINATION_TASK_TYPE],
        default="diagnosis",
    )
    parser.add_argument("--eval_dataset")
    parser.add_argument("--eval_file", "--eval-file")
    parser.add_argument("--kb_path")
    parser.add_argument("--exam_kb_path", "--exam-kb-path")
    parser.add_argument("--backend", required=True, choices=["ollama", "hf"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--run_name", "--run-name")
    parser.add_argument("--output_dir", "--output-dir")

    parser.add_argument("--successful_cases")
    parser.add_argument("--validated_reflections")
    parser.add_argument("--eval_mode", "--eval-mode", choices=["no_memory", "with_memory", "kb_only", "memory_only"])
    parser.add_argument("--n_success_memory", type=positive_int, default=3)
    parser.add_argument("--n_reflection_memory", type=positive_int, default=4)
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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", "--max-tokens", type=positive_int, default=512)
    parser.add_argument("--eval_limit", "--eval-limit", type=positive_int)
    parser.add_argument("--allow_eval_learning", action="store_true")
    parser.add_argument("--ollama_host", default="http://localhost:11434")
    parser.add_argument("--quiet", action="store_true")
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
    run_name = args.run_name or (
        Path(args.output_dir).name if args.output_dir else None
    )
    if not run_name:
        parser.error("--run_name is required unless --output_dir is provided")

    backend = build_backend(args)

    if args.task_type == EXAMINATION_TASK_TYPE:
        eval_file = args.eval_file or args.eval_dataset
        if not eval_file:
            parser.error("--eval_file is required when --task_type examination_selection")
        if not args.exam_kb_path:
            parser.error("--exam_kb_path is required when --task_type examination_selection")
        runner = ExaminationEvaluationRunner(
            backend=backend,
            config=ExaminationEvalConfig(
                eval_file=eval_file,
                exam_kb_path=args.exam_kb_path,
                run_name=run_name,
                backend=args.backend,
                model=args.model,
                successful_cases=args.successful_cases,
                validated_reflections=args.validated_reflections,
                eval_mode=args.eval_mode,
                n_success_memory=args.n_success_memory,
                n_reflection_memory=args.n_reflection_memory,
                retrieval_mode=args.retrieval_mode,
                embedding_model=args.embedding_model,
                embedding_device=args.embedding_device,
                embedding_batch_size=args.embedding_batch_size,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                eval_limit=args.eval_limit,
                output_dir=args.output_dir,
                quiet=args.quiet,
            ),
        )
    else:
        if not args.eval_dataset:
            parser.error("--eval_dataset is required when --task_type diagnosis")
        if not args.kb_path:
            parser.error("--kb_path is required when --task_type diagnosis")
        runner = EvaluationRunner(
            backend=backend,
            config=EvaluationConfig(
                eval_dataset=args.eval_dataset,
                kb_path=args.kb_path,
                run_name=run_name,
                backend=args.backend,
                model=args.model,
                successful_cases=args.successful_cases,
                validated_reflections=args.validated_reflections,
                eval_mode=args.eval_mode,
                n_success_memory=args.n_success_memory,
                n_reflection_memory=args.n_reflection_memory,
                retrieval_mode=args.retrieval_mode,
                embedding_model=args.embedding_model,
                embedding_device=args.embedding_device,
                embedding_batch_size=args.embedding_batch_size,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                eval_limit=args.eval_limit,
                allow_eval_learning=args.allow_eval_learning,
                output_dir=args.output_dir,
                quiet=args.quiet,
            ),
        )
    summary = runner.run()

    print(f"Evaluation directory: {runner.output_dir}")
    print(f"Questions: {summary['n_questions']}")
    print(f"Accuracy: {summary['accuracy']}")
    print(f"Mode: {summary['eval_mode']}")


if __name__ == "__main__":
    main()
