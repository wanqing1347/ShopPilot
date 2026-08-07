from __future__ import annotations

import json
from typing import Any

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from app.agent.context_governance import (
    CacheBreakpointMiddleware,
    compact_tool_message,
    compute_cache_breakpoint,
)
from app.agent.state import ShopPilotState, initial_state


class _ToolCapableSequenceModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any):
        return self


class _SummaryModel:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls += 1
        return AIMessage(content=f"稳定摘要-{self.calls}：保留预算、平台和搜索结论")


def _tool_round(call_id: str, payload_size: int = 16) -> list[Any]:
    payload = {
        "platform": "amazon",
        "items": [
            {"id": f"item-{index}", "title": "陶瓷咖啡杯" * payload_size}
            for index in range(8)
        ],
    }
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "item_search",
                    "args": {"platform": "amazon"},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id=call_id,
        ),
    ]


def _history() -> list[Any]:
    messages: list[Any] = [HumanMessage(content="预算 300 元，不要塑料咖啡杯")]
    for index in range(1, 5):
        messages.extend(_tool_round(f"call-{index}"))
    return messages


def test_cache_breakpoint_never_splits_tool_call_pair() -> None:
    messages = _history()
    breakpoint_index = compute_cache_breakpoint(
        messages,
        keep_recent_tool_calls=2,
        keep_recent_messages=4,
    )

    assert isinstance(messages[breakpoint_index], AIMessage)
    assert messages[breakpoint_index].tool_calls[0]["id"] == "call-3"
    assert isinstance(messages[breakpoint_index + 1], ToolMessage)
    assert messages[breakpoint_index + 1].tool_call_id == "call-3"


def test_compacted_tool_message_remains_valid_json() -> None:
    original = _tool_round("call-json", payload_size=80)[1]
    assert isinstance(original, ToolMessage)

    compacted = compact_tool_message(original, max_chars=600)
    payload = json.loads(str(compacted.content))

    assert payload["_context_compacted"] is True
    assert payload["original_chars"] > 600
    assert compacted.tool_call_id == original.tool_call_id
    assert compacted.artifact is None


@pytest.mark.asyncio
async def test_cache_epoch_is_stable_and_usage_metrics_are_checkpointed() -> None:
    summary_model = _SummaryModel()
    middleware = CacheBreakpointMiddleware(
        summary_model,
        trigger_messages=6,
        trigger_chars=1_000_000,
        keep_recent_tools=2,
        keep_recent_messages=4,
        min_compaction_messages=3,
        summary_max_chars=1_000,
        tool_message_max_chars=600,
        enable_prompt_cache_key=True,
    )
    messages = _history()
    state = initial_state(
        query="买咖啡杯",
        thread_id="cache-epoch-test",
        user_id=None,
    )
    state["messages"] = messages
    captured: list[ModelRequest[Any]] = []

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        captured.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content="继续调用工具",
                    usage_metadata={
                        "input_tokens": 900,
                        "output_tokens": 20,
                        "total_tokens": 920,
                        "input_token_details": {"cache_read": 240},
                    },
                )
            ]
        )

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
        messages=messages,
        system_message=SystemMessage(content="固定 ShopPilot system prompt"),
        tools=[],
        state=state,
        model_settings={},
    )
    response = await middleware.awrap_model_call(request, handler)

    assert isinstance(response, ExtendedModelResponse)
    assert response.command is not None
    updates = response.command.update
    assert updates["context_cache_epoch"] == 1
    assert updates["context_summary"].startswith("稳定摘要-1")
    assert updates["context_metrics"]["cache_read_tokens"] == 240
    assert updates["context_metrics"]["saved_chars"] > 0
    assert captured[0].model_settings["prompt_cache_key"].startswith("shoppilot-agent-v1-")
    assert isinstance(captured[0].messages[0], SystemMessage)
    stable_summary_message = captured[0].messages[0].content
    assert "cache epoch 1" in str(stable_summary_message)
    for message in captured[0].messages:
        if isinstance(message, ToolMessage):
            json.loads(str(message.content))

    # Apply checkpoint updates and append one new message. The safe breakpoint has
    # not advanced, so the previous summary must be reused byte-for-byte.
    state.update(updates)
    second_messages = [*messages, HumanMessage(content="继续")]
    state["messages"] = second_messages
    second_request = request.override(messages=second_messages, state=state)
    second_response = await middleware.awrap_model_call(second_request, handler)

    assert isinstance(second_response, ExtendedModelResponse)
    assert summary_model.calls == 1
    assert captured[1].messages[0].content == stable_summary_message
    assert second_response.command is not None
    assert "context_summary" not in second_response.command.update
    assert second_response.command.update["context_metrics"]["cache_epoch"] == 1


@pytest.mark.asyncio
async def test_create_agent_persists_full_history_while_model_sees_compacted_window() -> None:
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver

    model = _ToolCapableSequenceModel(
        responses=[
            AIMessage(content="集成摘要：预算 300，不要塑料，已完成多轮搜索"),
            AIMessage(content="集成最终回答"),
        ]
    )
    middleware = CacheBreakpointMiddleware(
        model,
        trigger_messages=6,
        trigger_chars=1_000_000,
        keep_recent_tools=2,
        min_compaction_messages=3,
        tool_message_max_chars=600,
    )
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="固定 ShopPilot system prompt",
        state_schema=ShopPilotState,
        checkpointer=InMemorySaver(),
        middleware=[middleware],
    )
    messages = _history()
    state = initial_state(
        query="买咖啡杯",
        thread_id="cache-integration-test",
        user_id=None,
    )
    state["messages"] = messages

    result = await agent.ainvoke(
        state,
        config={"configurable": {"thread_id": "cache-integration-test"}},
    )

    assert result["context_cache_epoch"] == 1
    assert result["context_summary"].startswith("集成摘要")
    assert len(result["messages"]) == len(messages) + 1
    assert result["messages"][-1].content == "集成最终回答"


@pytest.mark.asyncio
async def test_summary_failure_keeps_full_tail_without_data_loss() -> None:
    class _BrokenSummaryModel:
        async def ainvoke(self, messages: Any) -> AIMessage:
            raise RuntimeError("summary unavailable")

    middleware = CacheBreakpointMiddleware(
        _BrokenSummaryModel(),
        trigger_messages=6,
        trigger_chars=1_000_000,
        keep_recent_tools=2,
        min_compaction_messages=3,
        tool_message_max_chars=600,
    )
    messages = _history()
    state = initial_state(
        query="买咖啡杯",
        thread_id="cache-fallback-test",
        user_id=None,
    )
    state["messages"] = messages
    captured: list[ModelRequest[Any]] = []

    async def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        captured.append(request)
        return ModelResponse(result=[AIMessage(content="fallback works")])

    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[AIMessage(content="unused")]),
        messages=messages,
        system_message=SystemMessage(content="fixed"),
        tools=[],
        state=state,
        model_settings={},
    )
    response = await middleware.awrap_model_call(request, handler)

    assert isinstance(response, ExtendedModelResponse)
    assert response.command is not None
    assert "context_summary" not in response.command.update
    assert len(captured[0].messages) == len(messages)
    assert response.command.update["context_metrics"]["summary_until"] == 0
