from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.main_agent import run_agent
from app.agent.state import initial_state
from app.memory.store import store
from app.models import (
    CategoryInsightOutput,
    GroundedClaim,
    ItemPickerOutput,
    PickedItem,
    QueryPlan,
    ShoppingSummaryOutput,
)
from app.tools.planner import plan_query
from app.tools.shopping_summary import shopping_summary
from app.utils.runtime import ensure_session_dir, safe_join


class _FakeStructuredModel:
    def __init__(self, response: object) -> None:
        self.response = response

    async def ainvoke(self, messages: object) -> object:
        return self.response


class _FakeLLM:
    def __init__(self, response: object) -> None:
        self.response = response

    def with_structured_output(
        self,
        schema: object,
        **kwargs: object,
    ) -> _FakeStructuredModel:
        return _FakeStructuredModel(self.response)


def test_safe_join_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        safe_join(tmp_path, "..", "secret.txt")


@pytest.mark.asyncio
async def test_planner_uses_structured_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = QueryPlan(
        original_query="模型暂存值",
        category="咖啡杯",
        budget_cny=300,
        platforms=["amazon"],
        hard_constraints=["不要塑料", "预算不超过 300 CNY"],
        soft_preferences=["偏好小众"],
    )
    monkeypatch.setattr(
        "app.tools.planner.get_llm",
        lambda: _FakeLLM(expected),
    )

    query = "只在亚马逊买咖啡杯，预算300，不要塑料，偏好小众"
    plan = await plan_query(query)

    assert plan.original_query == query
    assert plan.category == "咖啡杯"
    assert plan.platforms == ["amazon"]
    assert plan.budget_cny == 300


@pytest.mark.asyncio
async def test_summary_uses_llm_and_preserves_picks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = QueryPlan(
        original_query="买一个小众咖啡杯",
        category="咖啡杯",
        platforms=["shopee"],
        soft_preferences=["偏好小众"],
    )
    picker = ItemPickerOutput(
        picks=[
            PickedItem(
                item_id="SHP-CUP-01",
                platform="shopee",
                title="手作粗陶带壶嘴咖啡分享杯",
                landed_cny=177.68,
                score=0.92,
                reasons=["款式更小众"],
            )
        ]
    )
    monkeypatch.setattr(
        "app.tools.shopping_summary.get_llm",
        lambda: _FakeLLM(
            {
                "final_text": "# 推荐\n\n手作粗陶咖啡分享杯",
                "learned_preferences": ["偏好小众"],
            }
        ),
    )

    result = await shopping_summary(picker, plan)

    assert result.final_text.startswith("# 推荐")
    assert result.picks == picker.picks
    assert result.learned_preferences == ["偏好小众"]


@pytest.mark.asyncio
async def test_summary_appends_validated_category_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = QueryPlan(
        original_query="买一个咖啡杯",
        category="咖啡杯",
        platforms=["shopee"],
    )
    picker = ItemPickerOutput(picks=[])
    insight = CategoryInsightOutput(
        category="咖啡杯",
        category_key="coffee_cup",
        answer_mode="llm_grounded",
        grounded_answer="- 跨平台应比较到手价。 [K0030]",
        grounded_claims=[
            GroundedClaim(text="跨平台应比较到手价。", citation_ids=["K0030"], support_score=0.8)
        ],
    )
    monkeypatch.setattr(
        "app.tools.shopping_summary.get_llm",
        lambda: _FakeLLM({"final_text": "# 暂无候选", "learned_preferences": []}),
    )

    result = await shopping_summary(picker, plan, insight)

    assert "### 品类依据（项目知识库）" in result.final_text
    assert "[K0030]" in result.final_text
    assert "不代表实时平台市场" in result.final_text


@pytest.mark.asyncio
async def test_main_entry_persists_checkpointed_business_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.agent import graph_runtime

    async def fake_agent(*, query: str, thread_id: str, system_prompt: str, initial):
        state = dict(initial)
        state["plan"] = QueryPlan(
            original_query=query,
            category="咖啡杯",
            platforms=["amazon"],
        )
        state["summary"] = ShoppingSummaryOutput(
            final_text="# Agent 结果",
            picks=[],
            learned_preferences=["偏好陶瓷"],
        )
        state["terminated"] = True
        state["terminal_tool"] = "shopping_summary"
        return graph_runtime.AgentRunResult(final_text="# Agent 结果", state=state)

    monkeypatch.setattr(graph_runtime, "ainvoke_agent", fake_agent)
    old_path = store.path
    store.path = tmp_path / "preferences.json"
    thread_id = "pytest-agent-state"
    try:
        result = await run_agent("找一个陶瓷咖啡杯", thread_id, "pytest-user")
    finally:
        store.path = old_path

    assert result.status == "ok"
    assert result.final == "# Agent 结果"
    payload = json.loads(
        (ensure_session_dir(thread_id) / "result.json").read_text(encoding="utf-8")
    )
    assert payload["runtime"] == "langgraph-agentloop"
    assert payload["terminated"] is True
    assert payload["terminal_tool"] == "shopping_summary"
    assert payload["plan"]["category"] == "咖啡杯"


def test_initial_state_contains_checkpointed_artifacts() -> None:
    state = initial_state(
        query="买杯子",
        thread_id="state-test",
        user_id="u1",
        long_term_preferences=["不要塑料"],
    )
    assert state["plan"] is None
    assert state["search_outputs"] == {}
    assert state["sub_agent_results"] == {}
    assert state["terminated"] is False
