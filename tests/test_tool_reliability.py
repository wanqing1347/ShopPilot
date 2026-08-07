from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command

from app.agent import tool_reliability
from app.agent.tool_reliability import (
    ToolBusinessError,
    ToolErrorCategory,
    ToolExecutionRegistry,
    ToolPolicy,
    ToolReliabilityMiddleware,
    classify_tool_error,
)


def _request(
    *,
    tool_name: str,
    call_id: str,
    args: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": tool_name,
            "args": args or {},
            "id": call_id,
            "type": "tool_call",
        },
        tool=None,
        state=state or {"thread_id": "tool-test", "tool_idempotency": {}},
        runtime=None,  # type: ignore[arg-type]
    )


def _tool_message(result: Command[Any]) -> ToolMessage:
    assert isinstance(result.update, dict)
    messages = result.update.get("messages") or []
    assert messages and isinstance(messages[-1], ToolMessage)
    return messages[-1]


def test_error_classification_separates_transient_and_business_failures() -> None:
    assert classify_tool_error(asyncio.TimeoutError()) == ToolErrorCategory.TIMEOUT
    assert classify_tool_error(ConnectionError("offline")) == ToolErrorCategory.NETWORK
    assert classify_tool_error(ToolBusinessError("invalid scope")) == ToolErrorCategory.BUSINESS
    assert classify_tool_error(RuntimeError("bug")) == ToolErrorCategory.INTERNAL


@pytest.mark.asyncio
async def test_transient_failure_retries_then_succeeds() -> None:
    middleware = ToolReliabilityMiddleware(
        registry=ToolExecutionRegistry(),
        max_retries=2,
        initial_delay_sec=0,
        max_delay_sec=0,
        jitter=False,
    )
    attempts = 0

    async def handler(request: ToolCallRequest):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("temporary network failure")
        return ToolMessage(
            content="ok",
            tool_call_id=str(request.tool_call["id"]),
            name=str(request.tool_call["name"]),
        )

    result = await middleware.awrap_tool_call(
        _request(tool_name="item_search", call_id="retry-call"),
        handler,
    )
    assert isinstance(result, Command)
    assert attempts == 3
    assert _tool_message(result).content == "ok"
    assert result.update["tool_reliability"]["item_search"]["attempts"] == 3


@pytest.mark.asyncio
async def test_tool_timeout_returns_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_reliability,
        "_tool_policy",
        lambda _: ToolPolicy(timeout_sec=0.01, cache_ttl_sec=0),
    )
    middleware = ToolReliabilityMiddleware(
        registry=ToolExecutionRegistry(),
        max_retries=0,
        jitter=False,
    )

    async def handler(_: ToolCallRequest):
        await asyncio.sleep(0.05)
        raise AssertionError("unreachable")

    result = await middleware.awrap_tool_call(
        _request(tool_name="web_search", call_id="timeout-call"),
        handler,
    )
    assert isinstance(result, Command)
    message = _tool_message(result)
    assert message.status == "error"
    payload = json.loads(str(message.content))
    assert payload["tool_error"]["category"] == "timeout"
    assert payload["tool_error"]["attempts"] == 1


@pytest.mark.asyncio
async def test_checkpoint_idempotency_replays_state_with_current_tool_call_id() -> None:
    middleware = ToolReliabilityMiddleware(
        registry=ToolExecutionRegistry(),
        max_retries=0,
        jitter=False,
    )
    calls = 0
    base_state = {
        "thread_id": "idempotency-thread",
        "search_outputs": {"amazon": {"candidates": []}},
        "tool_idempotency": {},
    }

    async def handler(request: ToolCallRequest):
        nonlocal calls
        calls += 1
        return Command(
            update={
                "compared": {"ranked": [], "base_currency": "CNY"},
                "messages": [
                    ToolMessage(
                        content='{"ranked":[]}',
                        tool_call_id=str(request.tool_call["id"]),
                        name="price_compare",
                        artifact={"ranked": []},
                    )
                ],
            }
        )

    first = await middleware.awrap_tool_call(
        _request(
            tool_name="price_compare",
            call_id="idem-first",
            args={"base_currency": "CNY"},
            state=base_state,
        ),
        handler,
    )
    assert isinstance(first, Command)
    cache = first.update["tool_idempotency"]

    async def must_not_run(_: ToolCallRequest):
        raise AssertionError("checkpoint cache should have been used")

    second_state = {**base_state, "tool_idempotency": cache}
    second = await middleware.awrap_tool_call(
        _request(
            tool_name="price_compare",
            call_id="idem-second",
            args={"base_currency": "CNY"},
            state=second_state,
        ),
        must_not_run,
    )
    assert isinstance(second, Command)
    assert calls == 1
    assert _tool_message(second).tool_call_id == "idem-second"
    assert second.update["compared"]["base_currency"] == "CNY"
    assert second.update["tool_reliability"]["price_compare"]["source"] == "checkpoint_cache"


@pytest.mark.asyncio
async def test_concurrent_equivalent_calls_share_one_inflight_execution() -> None:
    middleware = ToolReliabilityMiddleware(
        registry=ToolExecutionRegistry(),
        max_retries=0,
        jitter=False,
    )
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()
    state = {
        "thread_id": "inflight-thread",
        "query": "陶瓷杯",
        "plan": None,
        "long_term_preferences": [],
        "allowed_platforms": None,
        "allowed_category": None,
        "tool_idempotency": {},
    }

    async def handler(request: ToolCallRequest):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return ToolMessage(
            content="shared",
            tool_call_id=str(request.tool_call["id"]),
            name="item_search",
        )

    first_task = asyncio.create_task(
        middleware.awrap_tool_call(
            _request(
                tool_name="item_search",
                call_id="inflight-a",
                args={"platform": "amazon"},
                state=state,
            ),
            handler,
        )
    )
    await started.wait()
    second_task = asyncio.create_task(
        middleware.awrap_tool_call(
            _request(
                tool_name="item_search",
                call_id="inflight-b",
                args={"platform": "amazon"},
                state=state,
            ),
            handler,
        )
    )
    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert calls == 1
    assert isinstance(first, Command) and isinstance(second, Command)
    assert _tool_message(first).tool_call_id == "inflight-a"
    assert _tool_message(second).tool_call_id == "inflight-b"


@pytest.mark.asyncio
async def test_circuit_opens_fast_rejects_and_half_open_recovers() -> None:
    registry = ToolExecutionRegistry()
    middleware = ToolReliabilityMiddleware(
        registry=registry,
        max_retries=0,
        jitter=False,
        circuit_failure_threshold=1,
        circuit_reset_sec=0.02,
    )
    calls = 0

    async def failing(_: ToolCallRequest):
        nonlocal calls
        calls += 1
        raise ConnectionError("service unavailable")

    request = _request(tool_name="web_search", call_id="circuit-a", args={"query": "x"})
    first = await middleware.awrap_tool_call(request, failing)
    assert isinstance(first, Command)
    assert json.loads(str(_tool_message(first).content))["tool_error"]["category"] == "network"

    second = await middleware.awrap_tool_call(
        _request(tool_name="web_search", call_id="circuit-b", args={"query": "x"}),
        failing,
    )
    assert isinstance(second, Command)
    assert calls == 1
    assert json.loads(str(_tool_message(second).content))["tool_error"]["category"] == "circuit_open"

    await asyncio.sleep(0.03)

    async def recovered(request: ToolCallRequest):
        nonlocal calls
        calls += 1
        return ToolMessage(
            content="recovered",
            tool_call_id=str(request.tool_call["id"]),
            name="web_search",
        )

    third = await middleware.awrap_tool_call(
        _request(tool_name="web_search", call_id="circuit-c", args={"query": "x"}),
        recovered,
    )
    assert isinstance(third, Command)
    assert calls == 2
    assert _tool_message(third).content == "recovered"
