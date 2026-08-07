from __future__ import annotations

import pytest

from app.agent.graph_runtime import select_tool_names
from app.agent.llm import AgentConfigurationError, clear_llm_cache, get_llm
from app.agent.prompts import get_system_prompt
from app.agent.state import initial_state, merge_mappings
from langchain.tools import ToolRuntime


def test_prompt_contains_agentloop_and_fork_policy() -> None:
    prompt = get_system_prompt(["不要塑料", "偏好小众"])
    assert "Think → Act → Observe → Reflect" in prompt
    assert "dispatch_tool" in prompt
    assert "独立 thread_id 和 checkpoint" in prompt
    assert "不要塑料" in prompt


def test_langgraph_mode_requires_model_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "LLM_BASE_URL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    clear_llm_cache()
    try:
        with pytest.raises(AgentConfigurationError):
            get_llm()
    finally:
        clear_llm_cache()


def test_full_tool_set_matches_documented_architecture() -> None:
    from app.agent.tool_registry import get_full_tool_set

    names = {tool.name for tool in get_full_tool_set()}
    assert names == {
        "planner",
        "chat_fallback",
        "web_search",
        "category_insight",
        "item_search",
        "item_picker",
        "price_compare",
        "shipping_calc",
        "shopping_summary",
        "dispatch_tool",
    }


def test_terminal_tools_are_return_direct() -> None:
    from app.agent.tool_registry import chat_fallback, shopping_summary

    assert chat_fallback.return_direct is True
    assert shopping_summary.return_direct is True


def test_scoped_sub_agent_has_single_terminal_search_tool() -> None:
    from app.agent.tool_registry import get_sub_agent_tool_set

    tools = get_sub_agent_tool_set()
    assert [tool.name for tool in tools] == ["item_search"]
    assert tools[0].return_direct is True
    assert set(tools[0].args) == {"top_k"}


def test_scoped_sub_agent_reuses_parent_plan_without_mutating_it() -> None:
    from app.agent.dispatch_tool import _scoped_parent_plan
    from app.models import QueryPlan

    parent = initial_state(
        query="四平台咖啡杯",
        thread_id="parent-plan-test",
        user_id=None,
    )
    parent["plan"] = QueryPlan(
        original_query=parent["query"],
        category="咖啡杯",
        category_key="coffee_cup",
        budget_cny=300,
        platforms=["amazon", "shopee", "aliexpress", "ebay"],
        hard_constraints=["不要塑料"],
        soft_preferences=["偏好小众手作"],
    )

    scoped = _scoped_parent_plan(parent, platform="ebay", category=None)

    assert scoped is not None
    assert scoped.platforms == ["ebay"]
    assert scoped.category == "咖啡杯"
    assert scoped.budget_cny == 300
    assert scoped.hard_constraints == ["不要塑料"]
    assert parent["plan"].platforms == ["amazon", "shopee", "aliexpress", "ebay"]


def test_stage_permissions_follow_checkpointed_state() -> None:
    state = initial_state(
        query="买咖啡杯",
        thread_id="permission-test",
        user_id=None,
    )
    assert select_tool_names(state) == ["planner", "chat_fallback", "web_search"]

    state["plan"] = {
        "original_query": "买咖啡杯",
        "category": "咖啡杯",
        "platforms": ["amazon"],
        "hard_constraints": [],
        "soft_preferences": [],
    }
    assert "item_search" in select_tool_names(state)
    assert "price_compare" not in select_tool_names(state)

    state["search_outputs"] = {"amazon": {"candidates": []}}
    assert "price_compare" in select_tool_names(state)

    state["plan"]["platforms"] = ["amazon", "shopee"]
    assert "price_compare" not in select_tool_names(state)
    assert "item_search" in select_tool_names(state)
    state["search_outputs"]["shopee"] = {"candidates": []}
    assert "price_compare" in select_tool_names(state)

    state["terminated"] = True
    assert select_tool_names(state) == []


def test_sub_agent_is_leaf_worker_and_stops_after_scoped_search() -> None:
    state = initial_state(
        query="只搜 Amazon 咖啡杯",
        thread_id="sub-permission-test",
        user_id=None,
        allowed_platforms=["amazon"],
        allowed_category="咖啡杯",
        is_sub_agent=True,
    )
    state["plan"] = {
        "original_query": state["query"],
        "category": "咖啡杯",
        "platforms": ["amazon"],
        "hard_constraints": [],
        "soft_preferences": [],
    }

    tools = select_tool_names(state)
    assert "item_search" in tools
    assert "dispatch_tool" not in tools

    state["search_outputs"] = {"amazon": {"candidates": []}}
    assert select_tool_names(state) == []


def test_parallel_mapping_reducer_merges_child_artifacts() -> None:
    merged = merge_mappings(
        {"amazon": {"count": 2}},
        {"shopee": {"count": 3}},
    )
    assert merged == {
        "amazon": {"count": 2},
        "shopee": {"count": 3},
    }


@pytest.mark.asyncio
async def test_sub_agent_platform_scope_is_enforced_by_tool() -> None:
    from app.agent.tool_registry import item_search

    state = initial_state(
        query="买咖啡杯",
        thread_id="scope-test",
        user_id=None,
        allowed_platforms=["amazon"],
        allowed_category="咖啡杯",
        is_sub_agent=True,
    )
    runtime = ToolRuntime(
        state=state,
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="call-scope",
        store=None,
        tools=[],
    )

    rejected = await item_search.coroutine(
        platform="shopee",
        runtime=runtime,
        query="咖啡杯",
        category="咖啡杯",
    )
    assert "拒绝越权平台" in rejected.update["messages"][0].content
    assert "search_outputs" not in rejected.update

    accepted = await item_search.coroutine(
        platform="amazon",
        runtime=runtime,
        query="咖啡杯",
        category="旅行收纳",
    )
    assert "amazon" in accepted.update["search_outputs"]
    output = accepted.update["search_outputs"]["amazon"]
    assert all(
        candidate.attributes["category"] == "咖啡杯"
        for candidate in output.candidates
    )


@pytest.mark.asyncio
async def test_item_search_reuses_existing_platform_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.agent import tool_registry

    state = initial_state(
        query="买旅行收纳",
        thread_id="search-dedup-test",
        user_id=None,
    )
    state["search_outputs"] = {
        "amazon": {
            "platform": "amazon",
            "candidates": [
                {
                    "item_id": "amazon:cached",
                    "same_group_id": "cached",
                    "platform": "amazon",
                    "title": "缓存旅行收纳袋",
                    "category_key": "travel_storage",
                    "price": 99.0,
                    "currency": "CNY",
                }
            ],
            "total_recall": 1,
            "truncated": False,
        }
    }
    runtime = ToolRuntime(
        state=state,
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="call-dedup",
        store=None,
        tools=[],
    )

    async def unexpected_search(**_: object):
        raise AssertionError("已有平台结果时不应再次执行底层检索")

    monkeypatch.setattr(tool_registry, "_item_search", unexpected_search)
    reused = await tool_registry.item_search.coroutine(
        platform="amazon",
        runtime=runtime,
        query="换个说法也不应重复搜",
        category="travel_storage",
    )

    output = reused.update["search_outputs"]["amazon"]
    assert output.candidates[0].item_id == "amazon:cached"
