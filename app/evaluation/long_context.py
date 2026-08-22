from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.context_governance import CacheBreakpointMiddleware
from app.agent.state import initial_state


MIN_BENCHMARK_ROUNDS = 10
MAX_BENCHMARK_ROUNDS = 20


@dataclass(frozen=True)
class LongContextRound:
    round_index: int
    input_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    input_tokens_observed: bool
    cache_metadata_observed: bool
    cache_hit_ratio: float | None
    cache_epoch: int
    summary_until: int
    breakpoint_index: int
    original_message_count: int
    model_message_count: int
    original_chars: int
    model_chars: int
    saved_chars: int
    compacted_tool_messages: int


@dataclass(frozen=True)
class LongContextSummary:
    rounds: int
    warmup_rounds: int
    evaluated_rounds: int
    usage_rounds: int
    cache_metadata_rounds: int
    max_input_tokens: int | None
    total_input_tokens: int
    total_cache_read_tokens: int
    overall_cache_hit_ratio: float | None
    steady_state_cache_hit_ratio: float | None
    max_original_chars: int
    max_model_chars: int
    max_saved_chars: int
    epochs_observed: list[int]
    token_budget_target: int
    cache_hit_target: float
    token_budget_passed: bool | None
    cache_hit_passed: bool | None
    verdict: str


@dataclass(frozen=True)
class LongContextBenchmarkResult:
    generated_at: str
    model_name: str
    prompt_cache_key_enabled: bool
    tool_payload_chars: int
    rounds: list[LongContextRound]
    summary: LongContextSummary

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_rounds(rounds: int, warmup_rounds: int) -> None:
    if not MIN_BENCHMARK_ROUNDS <= rounds <= MAX_BENCHMARK_ROUNDS:
        raise ValueError(
            f"rounds 必须在 {MIN_BENCHMARK_ROUNDS}~{MAX_BENCHMARK_ROUNDS} 之间，当前为 {rounds}"
        )
    if warmup_rounds < 0 or warmup_rounds >= rounds:
        raise ValueError("warmup_rounds 必须 >= 0 且小于 rounds")


def _model_name(model: Any) -> str:
    for attr in ("model_name", "model"):
        value = getattr(model, attr, None)
        if value:
            return str(value)
    return model.__class__.__name__


def _tool_payload(round_index: int, target_chars: int) -> str:
    if target_chars < 1_000:
        raise ValueError("tool_payload_chars 至少为 1000")

    item_template = {
        "title": "耐热陶瓷咖啡杯",
        "material": "ceramic",
        "currency": "CNY",
        "description": (
            "用于长上下文基准的稳定商品证据。包含材质、容量、价格、平台、排除项与比较依据。"
            * 8
        ),
    }
    items: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "benchmark_round": round_index,
        "platform": "amazon",
        "query": "预算300元以内的陶瓷咖啡杯，不要塑料",
        "items": items,
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    item_index = 0
    while len(encoded) < target_chars:
        item_index += 1
        items.append(
            {
                "id": f"benchmark-{round_index}-{item_index}",
                "price": 80 + (item_index % 20),
                **item_template,
            }
        )
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return encoded


def _append_synthetic_tool_round(
    messages: list[Any],
    *,
    round_index: int,
    tool_payload_chars: int,
) -> None:
    call_id = f"benchmark-call-{round_index}"
    messages.extend(
        [
            HumanMessage(
                content=(
                    f"第 {round_index} 轮：继续沿用预算300元、排除塑料、优先陶瓷的约束，"
                    "检查新增商品证据后只确认上下文仍然一致。"
                )
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "item_search",
                        "args": {
                            "platform": "amazon",
                            "query": "预算300元以内的陶瓷咖啡杯，不要塑料",
                        },
                        "id": call_id,
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=_tool_payload(round_index, tool_payload_chars),
                tool_call_id=call_id,
            ),
        ]
    )


def _usage_observation(message: AIMessage) -> tuple[bool, bool]:
    usage = message.usage_metadata or {}
    input_observed = "input_tokens" in usage
    cache_observed = False

    details = usage.get("input_token_details") or {}
    if isinstance(details, dict) and (
        "cache_read" in details or "cached_tokens" in details
    ):
        cache_observed = True

    token_usage = message.response_metadata.get("token_usage") or {}
    prompt_details = token_usage.get("prompt_tokens_details") or {}
    if isinstance(prompt_details, dict) and "cached_tokens" in prompt_details:
        cache_observed = True

    return input_observed, cache_observed


async def _default_handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
    model_messages = [
        *([request.system_message] if request.system_message is not None else []),
        *request.messages,
    ]
    response = await request.model.ainvoke(
        model_messages,
        **dict(request.model_settings),
    )
    if not isinstance(response, AIMessage):
        raise TypeError(
            "Long-context benchmark 需要 ChatModel 返回 AIMessage，"
            f"实际为 {type(response).__name__}"
        )
    return ModelResponse(result=[response])


def summarize_long_context_rounds(
    samples: list[LongContextRound],
    *,
    warmup_rounds: int,
    token_budget_target: int,
    cache_hit_target: float,
) -> LongContextSummary:
    if not samples:
        raise ValueError("samples 不能为空")

    steady = samples[warmup_rounds:]
    usage_rounds = sum(sample.input_tokens_observed for sample in samples)
    cache_metadata_rounds = sum(sample.cache_metadata_observed for sample in steady)

    observed_inputs = [
        sample.input_tokens for sample in samples if sample.input_tokens_observed
    ]
    total_input = sum(observed_inputs)
    total_cache = sum(
        sample.cache_read_tokens
        for sample in samples
        if sample.input_tokens_observed and sample.cache_metadata_observed
    )
    overall_ratio = (
        total_cache / total_input
        if usage_rounds == len(samples) and total_input > 0
        else None
    )

    steady_input = sum(sample.input_tokens for sample in steady)
    steady_cache = sum(sample.cache_read_tokens for sample in steady)
    steady_ratio = (
        steady_cache / steady_input
        if steady
        and cache_metadata_rounds == len(steady)
        and all(sample.input_tokens_observed for sample in steady)
        and steady_input > 0
        else None
    )

    token_budget_passed: bool | None
    if usage_rounds != len(samples):
        token_budget_passed = None
    else:
        token_budget_passed = max(observed_inputs, default=0) <= token_budget_target

    cache_hit_passed = (
        None if steady_ratio is None else steady_ratio >= cache_hit_target
    )

    if token_budget_passed is False or cache_hit_passed is False:
        verdict = "FAIL"
    elif token_budget_passed is True and cache_hit_passed is True:
        verdict = "PASS"
    else:
        verdict = "UNSUPPORTED"

    return LongContextSummary(
        rounds=len(samples),
        warmup_rounds=warmup_rounds,
        evaluated_rounds=len(steady),
        usage_rounds=usage_rounds,
        cache_metadata_rounds=cache_metadata_rounds,
        max_input_tokens=max(observed_inputs) if observed_inputs else None,
        total_input_tokens=total_input,
        total_cache_read_tokens=total_cache,
        overall_cache_hit_ratio=overall_ratio,
        steady_state_cache_hit_ratio=steady_ratio,
        max_original_chars=max(sample.original_chars for sample in samples),
        max_model_chars=max(sample.model_chars for sample in samples),
        max_saved_chars=max(sample.saved_chars for sample in samples),
        epochs_observed=sorted({sample.cache_epoch for sample in samples}),
        token_budget_target=token_budget_target,
        cache_hit_target=cache_hit_target,
        token_budget_passed=token_budget_passed,
        cache_hit_passed=cache_hit_passed,
        verdict=verdict,
    )


async def run_long_context_benchmark(
    model: Any,
    *,
    rounds: int = 16,
    warmup_rounds: int = 3,
    tool_payload_chars: int = 9_000,
    token_budget_target: int = 30_000,
    cache_hit_target: float = 0.80,
    enable_prompt_cache_key: bool = False,
    handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]] | None = None,
) -> LongContextBenchmarkResult:
    """Run a 10~20 round real-model long-context benchmark.

    The benchmark feeds deterministic, tool-heavy history through the production
    CacheBreakpointMiddleware. Provider-reported token usage is used as the source
    of truth; cache targets become UNSUPPORTED when cached-token metadata is absent.
    """

    _validate_rounds(rounds, warmup_rounds)
    if token_budget_target <= 0:
        raise ValueError("token_budget_target 必须 > 0")
    if not 0 <= cache_hit_target <= 1:
        raise ValueError("cache_hit_target 必须在 0~1 之间")

    middleware = CacheBreakpointMiddleware(
        model,
        enable_prompt_cache_key=enable_prompt_cache_key,
    )
    state = initial_state(
        query="long-context benchmark",
        thread_id="long-context-benchmark",
        user_id=None,
    )
    system_message = SystemMessage(
        content=(
            "你正在执行 ShopPilot Long-Context Benchmark。"
            "历史中的预算、排除项、工具结果都只是稳定测试数据。"
            "每次只回复 ROUND_OK，不调用工具，不复述历史。"
        )
    )
    messages: list[Any] = []
    samples: list[LongContextRound] = []
    invoke_handler = handler or _default_handler

    for round_index in range(1, rounds + 1):
        _append_synthetic_tool_round(
            messages,
            round_index=round_index,
            tool_payload_chars=tool_payload_chars,
        )
        state["messages"] = list(messages)
        request = ModelRequest(
            model=model,
            messages=list(messages),
            system_message=system_message,
            tools=[],
            state=state,
            model_settings={},
        )
        wrapped = await middleware.awrap_model_call(request, invoke_handler)
        if not isinstance(wrapped, ExtendedModelResponse):
            raise TypeError("CacheBreakpointMiddleware 未返回 ExtendedModelResponse")
        if wrapped.command is None:
            raise RuntimeError("CacheBreakpointMiddleware 未返回 state update")

        updates = dict(wrapped.command.update)
        state.update(updates)
        response = wrapped.model_response
        ai_message = next(
            (message for message in response.result if isinstance(message, AIMessage)),
            None,
        )
        if ai_message is None:
            raise RuntimeError("benchmark model response 中缺少 AIMessage")
        messages.append(ai_message)
        state["messages"] = list(messages)

        metrics = dict(updates.get("context_metrics") or {})
        input_observed, cache_observed = _usage_observation(ai_message)
        input_tokens = int(metrics.get("input_tokens") or 0)
        cache_read_tokens = int(metrics.get("cache_read_tokens") or 0)
        cache_creation_tokens = int(metrics.get("cache_creation_tokens") or 0)
        cache_hit_ratio = (
            cache_read_tokens / input_tokens
            if input_observed and cache_observed and input_tokens > 0
            else None
        )
        samples.append(
            LongContextRound(
                round_index=round_index,
                input_tokens=input_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_creation_tokens=cache_creation_tokens,
                input_tokens_observed=input_observed,
                cache_metadata_observed=cache_observed,
                cache_hit_ratio=cache_hit_ratio,
                cache_epoch=int(metrics.get("cache_epoch") or 0),
                summary_until=int(metrics.get("summary_until") or 0),
                breakpoint_index=int(metrics.get("breakpoint_index") or 0),
                original_message_count=int(metrics.get("original_message_count") or 0),
                model_message_count=int(metrics.get("model_message_count") or 0),
                original_chars=int(metrics.get("original_chars") or 0),
                model_chars=int(metrics.get("model_chars") or 0),
                saved_chars=int(metrics.get("saved_chars") or 0),
                compacted_tool_messages=int(metrics.get("compacted_tool_messages") or 0),
            )
        )

    summary = summarize_long_context_rounds(
        samples,
        warmup_rounds=warmup_rounds,
        token_budget_target=token_budget_target,
        cache_hit_target=cache_hit_target,
    )
    return LongContextBenchmarkResult(
        generated_at=datetime.now(timezone.utc).isoformat(),
        model_name=_model_name(model),
        prompt_cache_key_enabled=enable_prompt_cache_key,
        tool_payload_chars=tool_payload_chars,
        rounds=samples,
        summary=summary,
    )


def render_long_context_markdown(result: LongContextBenchmarkResult) -> str:
    summary = result.summary

    def ratio(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2%}"

    def verdict(value: bool | None) -> str:
        if value is True:
            return "PASS"
        if value is False:
            return "FAIL"
        return "UNSUPPORTED"

    rows = [
        "# ShopPilot Long-Context Benchmark",
        "",
        f"- Generated at: `{result.generated_at}`",
        f"- Model: `{result.model_name}`",
        f"- Rounds: `{summary.rounds}`",
        f"- Warmup rounds: `{summary.warmup_rounds}`",
        f"- Tool payload chars/round: `{result.tool_payload_chars}`",
        f"- Prompt cache key enabled: `{result.prompt_cache_key_enabled}`",
        f"- Overall verdict: **{summary.verdict}**",
        "",
        "## Targets",
        "",
        f"- Max input tokens <= `{summary.token_budget_target}`: **{verdict(summary.token_budget_passed)}**",
        f"- Steady-state cache hit ratio >= `{summary.cache_hit_target:.0%}`: **{verdict(summary.cache_hit_passed)}**",
        "",
        "## Aggregate metrics",
        "",
        f"- Max input tokens: `{summary.max_input_tokens if summary.max_input_tokens is not None else 'N/A'}`",
        f"- Overall cache hit ratio: `{ratio(summary.overall_cache_hit_ratio)}`",
        f"- Steady-state cache hit ratio: `{ratio(summary.steady_state_cache_hit_ratio)}`",
        f"- Provider usage rounds: `{summary.usage_rounds}/{summary.rounds}`",
        f"- Cache metadata rounds after warmup: `{summary.cache_metadata_rounds}/{summary.evaluated_rounds}`",
        f"- Max original chars: `{summary.max_original_chars}`",
        f"- Max model chars: `{summary.max_model_chars}`",
        f"- Max saved chars: `{summary.max_saved_chars}`",
        f"- Cache epochs observed: `{summary.epochs_observed}`",
        "",
        "## Per-round metrics",
        "",
        "| Round | Input tokens | Cache read | Hit ratio | Epoch | Model chars | Saved chars |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sample in result.rounds:
        rows.append(
            "| "
            f"{sample.round_index} | {sample.input_tokens} | {sample.cache_read_tokens} | "
            f"{ratio(sample.cache_hit_ratio)} | {sample.cache_epoch} | "
            f"{sample.model_chars} | {sample.saved_chars} |"
        )

    rows.extend(
        [
            "",
            "## Interpretation",
            "",
            "`UNSUPPORTED` means the provider did not expose enough token/cache metadata to prove the target. "
            "It is intentionally different from `FAIL`: the benchmark never invents a cache hit ratio.",
            "",
        ]
    )
    return "\n".join(rows)
