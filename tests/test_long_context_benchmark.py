from __future__ import annotations

from typing import Any

import pytest
from langchain.messages import AIMessage

from app.evaluation.long_context import (
    LongContextRound,
    run_long_context_benchmark,
    summarize_long_context_rounds,
)


class _UsageAwareBenchmarkModel:
    def __init__(self, *, cache_ratio: float = 0.85) -> None:
        self.calls = 0
        self.cache_ratio = cache_ratio
        self.model_name = "fake-long-context-model"

    async def ainvoke(
        self,
        messages: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> AIMessage:
        first_content = str(getattr(messages[0], "content", "")) if messages else ""
        if "上下文压缩器" in first_content:
            return AIMessage(content="稳定摘要：预算300，排除塑料，优先陶瓷，保留工具结论。")

        self.calls += 1
        input_tokens = 12_000 + min(self.calls * 450, 5_000)
        cache_read_tokens = (
            0 if self.calls <= 2 else int(input_tokens * self.cache_ratio)
        )
        return AIMessage(
            content="ROUND_OK",
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": 4,
                "total_tokens": input_tokens + 4,
                "input_token_details": {"cache_read": cache_read_tokens},
            },
        )


def _sample(
    index: int,
    *,
    input_tokens: int,
    cache_read_tokens: int,
    cache_metadata_observed: bool = True,
) -> LongContextRound:
    return LongContextRound(
        round_index=index,
        input_tokens=input_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=0,
        input_tokens_observed=True,
        cache_metadata_observed=cache_metadata_observed,
        cache_hit_ratio=(
            cache_read_tokens / input_tokens
            if cache_metadata_observed and input_tokens
            else None
        ),
        cache_epoch=1 if index >= 7 else 0,
        summary_until=12 if index >= 7 else 0,
        breakpoint_index=12 if index >= 7 else 0,
        original_message_count=index * 4,
        model_message_count=index * 3,
        original_chars=index * 9_000,
        model_chars=min(index * 4_000, 24_000),
        saved_chars=max(0, index * 5_000 - 10_000),
        compacted_tool_messages=max(0, index - 1),
    )


@pytest.mark.asyncio
async def test_real_harness_runs_12_rounds_and_advances_cache_epoch() -> None:
    model = _UsageAwareBenchmarkModel(cache_ratio=0.86)

    result = await run_long_context_benchmark(
        model,
        rounds=12,
        warmup_rounds=2,
        tool_payload_chars=7_000,
        token_budget_target=30_000,
        cache_hit_target=0.80,
        enable_prompt_cache_key=True,
    )

    assert len(result.rounds) == 12
    assert result.summary.max_input_tokens is not None
    assert result.summary.max_input_tokens < 30_000
    assert result.summary.token_budget_passed is True
    assert result.summary.cache_hit_passed is True
    assert result.summary.verdict == "PASS"
    assert max(sample.cache_epoch for sample in result.rounds) >= 1
    assert any(sample.saved_chars > 0 for sample in result.rounds)


def test_summary_marks_missing_provider_cache_metadata_as_unsupported() -> None:
    samples = [
        _sample(
            index,
            input_tokens=10_000 + index * 100,
            cache_read_tokens=0,
            cache_metadata_observed=False,
        )
        for index in range(1, 11)
    ]

    summary = summarize_long_context_rounds(
        samples,
        warmup_rounds=2,
        token_budget_target=30_000,
        cache_hit_target=0.80,
    )

    assert summary.token_budget_passed is True
    assert summary.steady_state_cache_hit_ratio is None
    assert summary.cache_hit_passed is None
    assert summary.verdict == "UNSUPPORTED"


def test_summary_fails_when_steady_state_cache_ratio_is_below_target() -> None:
    samples = [
        _sample(
            index,
            input_tokens=15_000,
            cache_read_tokens=9_000,
        )
        for index in range(1, 11)
    ]

    summary = summarize_long_context_rounds(
        samples,
        warmup_rounds=2,
        token_budget_target=30_000,
        cache_hit_target=0.80,
    )

    assert summary.steady_state_cache_hit_ratio == pytest.approx(0.60)
    assert summary.cache_hit_passed is False
    assert summary.verdict == "FAIL"


@pytest.mark.parametrize("rounds", [9, 21])
@pytest.mark.asyncio
async def test_benchmark_rejects_round_counts_outside_10_to_20(rounds: int) -> None:
    with pytest.raises(ValueError, match="10~20"):
        await run_long_context_benchmark(
            _UsageAwareBenchmarkModel(),
            rounds=rounds,
        )
