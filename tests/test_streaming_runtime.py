from __future__ import annotations

import pytest
from langchain.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from app.agent import graph_runtime
from app.agent.state import initial_state
from app.models import ItemPickerOutput, QueryPlan


class _ToolCapableFakeChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class _FakeStructuredModel:
    async def ainvoke(self, messages):
        return {
            "final_text": "# 终结清单",
            "learned_preferences": ["偏好小众"],
        }


class _FakeSummaryLLM:
    def with_structured_output(self, schema, **kwargs):
        return _FakeStructuredModel()


@pytest.mark.asyncio
async def test_agent_streaming_returns_checkpoint_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _ToolCapableFakeChatModel(
        responses=[AIMessage(content="fake final answer")]
    )
    monkeypatch.setattr(graph_runtime, "get_llm", lambda: fake_model)

    state = initial_state(
        query="你好",
        thread_id="fake-stream-test",
        user_id=None,
    )
    result = await graph_runtime.ainvoke_agent(
        query="你好",
        thread_id="fake-stream-test",
        system_prompt="You are a test agent.",
        initial=state,
    )

    assert result.final_text == "fake final answer"
    assert result.state["messages"][-1].content == "fake final answer"


@pytest.mark.asyncio
async def test_leaf_sub_agent_returns_direct_after_scoped_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _ToolCapableFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "item_search",
                        "args": {},
                        "id": "call-leaf-search",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    monkeypatch.setattr(graph_runtime, "get_llm", lambda: fake_model)

    state = initial_state(
        query="只搜 Amazon 咖啡杯",
        thread_id="leaf-search-test",
        user_id=None,
        allowed_platforms=["amazon"],
        allowed_category="咖啡杯",
        is_sub_agent=True,
    )
    state["plan"] = QueryPlan(
        original_query="四平台搜索咖啡杯",
        category="咖啡杯",
        category_key="coffee_cup",
        budget_cny=300,
        platforms=["amazon"],
        hard_constraints=["不要塑料"],
        soft_preferences=["偏好小众手作"],
    )

    result = await graph_runtime.ainvoke_agent(
        query=state["query"],
        thread_id="leaf-search-test",
        system_prompt="Call item_search now.",
        initial=state,
        leaf_sub_agent=True,
    )

    assert list(result.state["search_outputs"]) == ["amazon"]
    assert result.state["compared"] is None
    assert result.state["shipping"] is None
    assert result.state["picker"] is None
    assert result.state["summary"] is None
    assert result.state["is_sub_agent"] is True


@pytest.mark.asyncio
async def test_terminal_summary_updates_state_and_stops_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_model = _ToolCapableFakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shopping_summary",
                        "args": {},
                        "id": "call-summary",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    monkeypatch.setattr(graph_runtime, "get_llm", lambda: fake_model)
    monkeypatch.setattr(
        "app.tools.shopping_summary.get_llm",
        lambda: _FakeSummaryLLM(),
    )

    state = initial_state(
        query="给我最终清单",
        thread_id="terminal-summary-test",
        user_id=None,
    )
    state["plan"] = QueryPlan(
        original_query="给我最终清单",
        category="咖啡杯",
        platforms=["amazon"],
    )
    state["search_outputs"] = {"amazon": {"placeholder": True}}
    state["compared"] = {"placeholder": True}
    state["shipping"] = {"placeholder": True}
    state["picker"] = ItemPickerOutput(picks=[])

    result = await graph_runtime.ainvoke_agent(
        query="给我最终清单",
        thread_id="terminal-summary-test",
        system_prompt="Call shopping_summary now.",
        initial=state,
    )

    assert result.final_text.startswith("# 推荐购物清单")
    assert "当前没有通过全部硬约束的候选商品" in result.final_text
    assert result.state["terminated"] is True
    assert result.state["terminal_tool"] == "shopping_summary"
    assert result.state["summary"].final_text == result.final_text
