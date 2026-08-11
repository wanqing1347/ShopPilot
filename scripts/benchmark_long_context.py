from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.agent.llm import get_llm
from app.evaluation.long_context import (
    MAX_BENCHMARK_ROUNDS,
    MIN_BENCHMARK_ROUNDS,
    render_long_context_markdown,
    run_long_context_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ShopPilot's real-provider 10~20 round long-context benchmark and "
            "measure input tokens, context compaction and prompt-cache reuse."
        )
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=16,
        help=f"Benchmark rounds, {MIN_BENCHMARK_ROUNDS}~{MAX_BENCHMARK_ROUNDS} (default: 16)",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=3,
        help="Rounds excluded from the steady-state cache ratio (default: 3)",
    )
    parser.add_argument(
        "--tool-payload-chars",
        type=int,
        default=9_000,
        help="Approximate raw ToolMessage payload size per round (default: 9000)",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=30_000,
        help="Hard target for the maximum provider-reported input tokens (default: 30000)",
    )
    parser.add_argument(
        "--min-cache-hit-ratio",
        type=float,
        default=0.80,
        help="Steady-state cached/input token target (default: 0.80)",
    )
    parser.add_argument(
        "--prompt-cache-key",
        action="store_true",
        help="Pass the optional stable prompt_cache_key through CacheBreakpointMiddleware",
    )
    parser.add_argument(
        "--output-dir",
        default="output/benchmarks",
        help="Directory for JSON/Markdown reports (default: output/benchmarks)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless both token-budget and cache-hit targets are proven PASS",
    )
    return parser


def _ratio(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


async def _run(args: argparse.Namespace) -> int:
    model = get_llm()
    result = await run_long_context_benchmark(
        model,
        rounds=args.rounds,
        warmup_rounds=args.warmup_rounds,
        tool_payload_chars=args.tool_payload_chars,
        token_budget_target=args.max_input_tokens,
        cache_hit_target=args.min_cache_hit_ratio,
        enable_prompt_cache_key=args.prompt_cache_key,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.generated_at.replace(":", "-").replace("+", "_")
    stem = f"long-context-{result.model_name.replace('/', '_')}-{stamp}"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_long_context_markdown(result),
        encoding="utf-8",
    )

    summary = result.summary
    print("ShopPilot Long-Context Benchmark")
    print(f"model={result.model_name} rounds={summary.rounds} warmup={summary.warmup_rounds}")
    print(
        "max_input_tokens="
        f"{summary.max_input_tokens if summary.max_input_tokens is not None else 'N/A'} "
        f"target<={summary.token_budget_target} status={summary.token_budget_passed}"
    )
    print(
        "steady_cache_hit_ratio="
        f"{_ratio(summary.steady_state_cache_hit_ratio)} "
        f"target>={summary.cache_hit_target:.0%} status={summary.cache_hit_passed}"
    )
    print(f"epochs={summary.epochs_observed} verdict={summary.verdict}")
    print(f"json_report={json_path}")
    print(f"markdown_report={markdown_path}")

    if args.strict and summary.verdict != "PASS":
        return 2
    return 0


def main() -> int:
    args = _parser().parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
