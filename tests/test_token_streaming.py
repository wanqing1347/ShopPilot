from __future__ import annotations

from typing import Any

import pytest
from langchain.messages import AIMessage, AIMessageChunk

from app.agent import graph_runtime
from app.api.connection import ConnectionManager


class _FakeStreamAgent:
    def __init__(self) -> None:
        self.stream_mode: Any = None

    async def astream(self, graph_input, *, config, stream_mode, version):
        self.stream_mode = stream_mode
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(content="你", id="message-1"),
                {"langgraph_node": "model", "tags": []},
            ),
        }
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(content="好", id="message-1"),
                {"langgraph_node": "model", "tags": []},
            ),
        }
        # Internal context summary tokens must never reach the user.
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(content="内部摘要", id="summary-1"),
                {
                    "langgraph_node": "model",
                    "tags": ["shoppilot-context-compaction"],
                },
            ),
        }
        # Structured-output calls executed inside a tool node are also hidden.
        yield {
            "type": "messages",
            "data": (
                AIMessageChunk(content="工具内部输出", id="tool-model-1"),
                {"langgraph_node": "tools", "tags": []},
            ),
        }
        yield {
            "type": "updates",
            "data": {
                "model": {
                    "messages": [
                        AIMessage(
                            content="你好",
                            tool_calls=[
                                {
                                    "name": "planner",
                                    "args": {"user_input": "买杯子"},
                                    "id": "call-1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ]
                }
            },
        }


@pytest.mark.asyncio
async def test_stream_updates_emits_main_model_tokens_and_step_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _FakeStreamAgent()
    tokens: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []

    async def record_token(**kwargs: Any) -> None:
        tokens.append(kwargs)

    async def record_call(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(graph_runtime.monitor, "report_assistant_token", record_token)
    monkeypatch.setattr(graph_runtime.monitor, "report_assistant_call", record_call)

    await graph_runtime._stream_updates(agent, {"messages": []}, {"configurable": {}})

    assert agent.stream_mode == ["messages", "updates"]
    assert [event["delta"] for event in tokens] == ["你", "好"]
    assert [event["token_index"] for event in tokens] == [1, 2]
    assert calls[0]["tool_calls"] == ["planner"]
    assert calls[0]["preview"] == "你好"


@pytest.mark.asyncio
async def test_transient_token_event_does_not_evict_durable_history() -> None:
    manager = ConnectionManager(history_size=2)
    await manager.send_to_thread(
        {"event": "assistant_token", "data": {"delta": "A"}},
        "thread",
        record_history=False,
    )
    await manager.send_to_thread(
        {"event": "tool_retry", "data": {}},
        "thread",
    )
    assert manager.get_history("thread") == [{"event": "tool_retry", "data": {}}]
