from __future__ import annotations

import argparse
from typing import Any

from src.backends import HuggingFaceBackend, OllamaBackend
from src.evaluation import EvaluationConfig, EvaluationRunner


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal held-out diagnosis evaluation runner with KB and optional case experience."
    )
    parser.add_argument("--eval_dataset", required=True)
    parser.add_argument("--kb_path", required=True)
    parser.add_argument("--backend", required=True, choices=["ollama", "hf"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--run_name", required=True)

    parser.add_argument("--successful_cases")
    parser.add_argument("--validated_reflections")
    parser.add_argument("--eval_mode", choices=["no_memory", "with_memory", "kb_only", "memory_only"])
    parser.add_argument("--n_success_memory", type=positive_int, default=5)
    parser.add_argument("--n_reflection_memory", type=positive_int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=positive_int, default=512)
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

    config = EvaluationConfig(
        eval_dataset=args.eval_dataset,
        kb_path=args.kb_path,
        run_name=args.run_name,
        backend=args.backend,
        model=args.model,
        successful_cases=args.successful_cases,
        validated_reflections=args.validated_reflections,
        eval_mode=args.eval_mode,
        n_success_memory=args.n_success_memory,
        n_reflection_memory=args.n_reflection_memory,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        allow_eval_learning=args.allow_eval_learning,
        quiet=args.quiet,
    )

    backend = build_backend(args)
    runner = EvaluationRunner(backend=backend, config=config)
    summary = runner.run()

    print(f"Evaluation directory: {runner.output_dir}")
    print(f"Questions: {summary['n_questions']}")
    print(f"Accuracy: {summary['accuracy']}")
    print(f"Mode: {summary['eval_mode']}")


if __name__ == "__main__":
    main()
