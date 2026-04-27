from __future__ import annotations

import argparse
from typing import Any

from src.backends import HuggingFaceBackend, OllamaBackend
from src.loop import SimulationConfig, SimulationLoop


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal expert-oriented infectious disease SEAL/RCL diagnosis loop."
    )
    parser.add_argument("--backend", required=True, choices=["ollama", "hf"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--kb_path", required=True)
    parser.add_argument("--sampler_config", required=True)
    parser.add_argument("--n_successful_cases", required=True, type=positive_int)
    parser.add_argument("--run_name", required=True)
    parser.add_argument(
        "--eval_every",
        type=positive_int,
        default=5,
        help="When --run_evaluation is enabled, write eval summaries every N loop iterations.",
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
    parser.add_argument(
        "--eval_mode",
        choices=["no_memory", "with_memory", "kb_only", "memory_only"],
        help="Optional held-out evaluation mode for periodic eval runs.",
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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress running progress logs.",
    )

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=positive_int, default=768)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n_success_memory", type=non_negative_int, default=5)
    parser.add_argument("--n_reflection_memory", type=non_negative_int, default=5)
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

    config = SimulationConfig(
        backend=args.backend,
        model=args.model,
        kb_path=args.kb_path,
        sampler_config=args.sampler_config,
        n_successful_cases=args.n_successful_cases,
        run_name=args.run_name,
        eval_every=args.eval_every,
        run_evaluation=args.run_evaluation,
        eval_dataset=args.eval_dataset,
        eval_mode=args.eval_mode,
        allow_eval_learning=args.allow_eval_learning,
        log_every=args.log_every,
        quiet=args.quiet,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        n_success_memory=args.n_success_memory,
        n_reflection_memory=args.n_reflection_memory,
        ollama_host=args.ollama_host,
        patient_qc=args.patient_qc,
    )

    backend = build_backend(args)
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
