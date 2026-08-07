from __future__ import annotations

from typing import Annotated, Any

from langchain.agents import AgentState

from app.models import (
    CategoryInsightOutput,
    ItemPickerOutput,
    ItemSearchOutput,
    Platform,
    PriceCompareOutput,
    QueryPlan,
    ShippingCalcOutput,
    ShoppingSummaryOutput,
)


def merge_mappings(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge concurrent mapping updates, with the newest value winning per key."""

    return {**(left or {}), **(right or {})}


def take_latest(left: Any, right: Any) -> Any:
    """Reducer for optional artifacts that may be updated by parallel tools."""

    return right if right is not None else left


class ShopPilotState(AgentState):
    """Checkpointed short-term state for one main or homogeneous sub AgentLoop.

    Unlike the former ContextVar-only workspace, every business artifact here is
    part of the LangGraph state snapshot and can therefore be recovered by the
    configured checkpointer for the same thread_id.
    """

    query: str
    thread_id: str
    user_id: str | None
    long_term_preferences: list[str]
    long_term_memory: list[dict[str, Any]]

    plan: Annotated[QueryPlan | None, take_latest]
    insight: Annotated[CategoryInsightOutput | None, take_latest]
    search_outputs: Annotated[dict[str, ItemSearchOutput], merge_mappings]
    compared: Annotated[PriceCompareOutput | None, take_latest]
    shipping: Annotated[ShippingCalcOutput | None, take_latest]
    picker: Annotated[ItemPickerOutput | None, take_latest]
    summary: Annotated[ShoppingSummaryOutput | None, take_latest]

    # Complete child artifacts are preserved without overwriting main artifacts.
    sub_agent_results: Annotated[dict[str, dict[str, Any]], merge_mappings]

    # Cache-aware context governance. Persisted history stays complete; these
    # fields describe the stable model-input summary and current cache epoch.
    context_summary: Annotated[str | None, take_latest]
    context_summary_until: Annotated[int, take_latest]
    context_cache_epoch: Annotated[int, take_latest]
    context_breakpoint_index: Annotated[int, take_latest]
    context_compaction_count: Annotated[int, take_latest]
    context_metrics: Annotated[dict[str, Any], merge_mappings]

    # Tool execution harness. Idempotency entries contain replayable tool state
    # updates; reliability keeps compact last-run metrics per tool.
    tool_idempotency: Annotated[dict[str, dict[str, Any]], merge_mappings]
    tool_reliability: Annotated[dict[str, dict[str, Any]], merge_mappings]

    # Runtime scope and state-machine controls.
    allowed_platforms: list[Platform] | None
    allowed_category: str | None
    is_sub_agent: bool
    terminated: bool
    terminal_tool: str | None


def initial_state(
    *,
    query: str,
    thread_id: str,
    user_id: str | None,
    long_term_preferences: list[str] | None = None,
    long_term_memory: list[dict[str, Any]] | None = None,
    allowed_platforms: list[Platform] | None = None,
    allowed_category: str | None = None,
    is_sub_agent: bool = False,
) -> ShopPilotState:
    """Create a complete initial state so checkpoint snapshots stay predictable."""

    return ShopPilotState(
        messages=[],
        query=query,
        thread_id=thread_id,
        user_id=user_id,
        long_term_preferences=list(long_term_preferences or []),
        long_term_memory=list(long_term_memory or []),
        plan=None,
        insight=None,
        search_outputs={},
        compared=None,
        shipping=None,
        picker=None,
        summary=None,
        sub_agent_results={},
        context_summary=None,
        context_summary_until=0,
        context_cache_epoch=0,
        context_breakpoint_index=0,
        context_compaction_count=0,
        context_metrics={},
        tool_idempotency={},
        tool_reliability={},
        allowed_platforms=allowed_platforms,
        allowed_category=allowed_category,
        is_sub_agent=is_sub_agent,
        terminated=False,
        terminal_tool=None,
    )


def state_payload(state: ShopPilotState | dict[str, Any]) -> dict[str, Any]:
    """Convert checkpointed state into the JSON-compatible task artifact."""

    def dump(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value

    search_outputs = state.get("search_outputs") or {}
    return {
        "query": state.get("query"),
        "thread_id": state.get("thread_id"),
        "user_id": state.get("user_id"),
        "long_term_preferences": state.get("long_term_preferences") or [],
        "long_term_memory": state.get("long_term_memory") or [],
        "plan": dump(state.get("plan")),
        "category_insight": dump(state.get("insight")),
        "search_outputs": [dump(output) for output in search_outputs.values()],
        "price_compare": dump(state.get("compared")),
        "shipping": dump(state.get("shipping")),
        "picker": dump(state.get("picker")),
        "summary": dump(state.get("summary")),
        "sub_agent_results": state.get("sub_agent_results") or {},
        "context_governance": {
            "summary": state.get("context_summary"),
            "summary_until": int(state.get("context_summary_until") or 0),
            "cache_epoch": int(state.get("context_cache_epoch") or 0),
            "breakpoint_index": int(state.get("context_breakpoint_index") or 0),
            "compaction_count": int(state.get("context_compaction_count") or 0),
            "metrics": state.get("context_metrics") or {},
        },
        "tool_harness": {
            "idempotency_entries": len(state.get("tool_idempotency") or {}),
            "reliability": state.get("tool_reliability") or {},
        },
        "allowed_platforms": state.get("allowed_platforms"),
        "allowed_category": state.get("allowed_category"),
        "is_sub_agent": bool(state.get("is_sub_agent")),
        "terminated": bool(state.get("terminated")),
        "terminal_tool": state.get("terminal_tool"),
    }
