from __future__ import annotations

import asyncio
import json
import os
import time
from functools import lru_cache
from typing import Any

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from app.agent.dispatch_tool import SubAgentRun, safe_run_sub_agent
from app.agent.llm import get_llm
from app.agent.settings import tool_result_max_chars
from app.models import CategoryInsightOutput, ItemSearchOutput, Platform, QueryPlan
from app.tools.category_insight import category_insight as _category_insight
from app.tools.item_picker import item_picker as _item_picker
from app.tools.item_search import item_search as _item_search
from app.tools.planner import plan_query as _plan_query
from app.tools.price_compare import price_compare as _price_compare
from app.tools.shipping_calc import shipping_calc as _shipping_calc
from app.tools.shopping_summary import shopping_summary as _shopping_summary


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _compact(value: Any, *, depth: int = 0) -> Any:
    """Produce a bounded JSON-safe preview without cutting JSON mid-token."""

    value = _model_dump(value)
    if depth >= 4:
        return "…"
    if isinstance(value, dict):
        items = list(value.items())
        compacted = {
            str(key): _compact(item, depth=depth + 1)
            for key, item in items[:20]
        }
        if len(items) > 20:
            compacted["_omitted_keys"] = len(items) - 20
        return compacted
    if isinstance(value, (list, tuple)):
        compacted = [_compact(item, depth=depth + 1) for item in value[:8]]
        if len(value) > 8:
            compacted.append({"_omitted_items": len(value) - 8})
        return compacted
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "…"
    return value


def _json_result(value: Any) -> str:
    raw = _model_dump(value)
    text = json.dumps(raw, ensure_ascii=False)
    limit = tool_result_max_chars()
    if len(text) <= limit:
        return text
    preview = {
        "truncated_for_model": True,
        "original_chars": len(text),
        "preview": _compact(raw),
    }
    return json.dumps(preview, ensure_ascii=False)


def _command(
    runtime: ToolRuntime,
    *,
    tool_name: str,
    content: str,
    updates: dict[str, Any] | None = None,
    artifact: Any | None = None,
) -> Command:
    message = ToolMessage(
        content=content,
        tool_call_id=runtime.tool_call_id,
        name=tool_name,
        artifact=artifact,
    )
    return Command(update={**(updates or {}), "messages": [message]})


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content or "")


def _scope_plan(plan: QueryPlan, state: dict[str, Any]) -> QueryPlan:
    updates: dict[str, Any] = {}
    allowed_platforms = state.get("allowed_platforms")
    allowed_category = state.get("allowed_category")
    if allowed_platforms:
        updates["platforms"] = list(allowed_platforms)
    if allowed_category:
        updates["category"] = allowed_category
    return plan.model_copy(update=updates) if updates else plan


def _platform_allowed(platform: Platform, state: dict[str, Any]) -> bool:
    allowed = state.get("allowed_platforms")
    return not allowed or platform in allowed


def _category_for_scope(requested: str, state: dict[str, Any]) -> str:
    allowed = state.get("allowed_category")
    if allowed:
        return str(allowed)
    if requested:
        return requested
    plan = state.get("plan")
    return plan.category if isinstance(plan, QueryPlan) else "旅行收纳"


@tool
async def planner(user_input: str, runtime: ToolRuntime) -> Command:
    """拆解购物需求，提取品类、预算、平台、硬约束和软偏好。复杂购物请求应先调用。"""

    started = time.perf_counter()
    from app.api.monitor import monitor

    await monitor.report_tool_start("planner", {"user_input": user_input[:500]})
    plan = _scope_plan(await _plan_query(user_input), runtime.state)
    await monitor.report_tool_end(
        "planner", int((time.perf_counter() - started) * 1000)
    )
    return _command(
        runtime,
        tool_name="planner",
        content=_json_result(plan),
        updates={"plan": plan},
        artifact=plan.model_dump(mode="json"),
    )


@tool(return_direct=True)
async def chat_fallback(message: str, runtime: ToolRuntime) -> Command:
    """处理与商品搜索无关的简短闲聊，并直接结束本次 AgentLoop。"""

    from app.api.monitor import monitor

    await monitor.report_tool_start("chat_fallback", {"message": message[:500]})
    response = await get_llm().ainvoke(
        [
            ("system", "你是 ShopPilot 购物助手。简洁回答当前非购物消息，不要虚构商品或执行下单。"),
            ("user", message),
        ]
    )
    text = _content_to_text(response.content).strip()
    await monitor.report_tool_end("chat_fallback", 0)
    return _command(
        runtime,
        tool_name="chat_fallback",
        content=text,
        updates={"terminated": True, "terminal_tool": "chat_fallback"},
    )


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """检索实时评测、趋势或商品外部资料。需要配置 TAVILY_API_KEY 和 web 可选依赖。"""

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return (
            "WebSearch 未配置：缺少 TAVILY_API_KEY。"
            "请改用品类知识和商品数据，或明确告知用户实时资料不可用。"
        )
    try:
        from tavily import TavilyClient
    except ImportError:
        return "WebSearch 未安装：请执行 `uv sync --extra web`。"

    client = TavilyClient(api_key=api_key)
    result = await asyncio.to_thread(
        client.search,
        query=query,
        max_results=max(1, min(max_results, 10)),
        search_depth="advanced",
    )
    return _json_result(result)


@tool
async def category_insight(
    category: str,
    runtime: ToolRuntime,
    depth: str = "deep",
) -> Command:
    """查询品类构成、典型材质、价格带和热门款式，为搜索和精排提供知识。"""

    selected = _category_for_scope(category, runtime.state)
    plan = runtime.state.get("plan")
    category_key = plan.category_key if isinstance(plan, QueryPlan) else None
    insight = await _category_insight(
        selected,
        depth=depth,
        query=str(runtime.state.get("query") or ""),
        category_key=category_key,
    )
    return _command(
        runtime,
        tool_name="category_insight",
        content=_json_result(insight),
        updates={"insight": insight},
        artifact=insight.model_dump(mode="json"),
    )


@tool
async def item_search(
    platform: Platform,
    runtime: ToolRuntime,
    query: str = "",
    category: str = "",
    top_k: int = 20,
) -> Command:
    """在一个指定电商平台检索商品。跨多个独立平台时优先使用 dispatch_tool 并行 fork。"""

    state = runtime.state
    if not _platform_allowed(platform, state):
        allowed = state.get("allowed_platforms") or []
        return _command(
            runtime,
            tool_name="item_search",
            content=f"拒绝越权平台 {platform}；当前子 Agent 只允许 {allowed}。",
        )

    existing_raw = (state.get("search_outputs") or {}).get(platform)
    if existing_raw is not None:
        existing = ItemSearchOutput.model_validate(existing_raw)
        if existing.candidates:
            from app.api.monitor import monitor

            await monitor.report_tool_idempotency_hit(
                tool_name="item_search",
                idempotency_key=f"checkpoint-search:{platform}",
                source="checkpoint_search_outputs",
            )
            return _command(
                runtime,
                tool_name="item_search",
                content=_json_result(existing),
                updates={"search_outputs": {platform: existing}},
                artifact=existing.model_dump(mode="json"),
            )

    selected_query = query or str(state.get("query") or "")
    selected_category = _category_for_scope(category, state)
    plan = state.get("plan")
    preferences = list(state.get("long_term_preferences") or [])
    category_key: str | None = None
    budget_cny: float | None = None
    hard_constraints: list[str] = []
    if isinstance(plan, QueryPlan):
        preferences.extend(plan.hard_constraints)
        preferences.extend(plan.soft_preferences)
        category_key = plan.category_key
        budget_cny = plan.budget_cny
        hard_constraints = list(plan.hard_constraints)
    output = await _item_search(
        query=selected_query,
        platform=platform,
        category=selected_category,
        top_k=top_k,
        user_preferences=list(dict.fromkeys(preferences)),
        category_key=category_key,
        budget_cny=budget_cny,
        hard_constraints=hard_constraints,
    )
    return _command(
        runtime,
        tool_name="item_search",
        content=_json_result(output),
        updates={"search_outputs": {platform: output}},
        artifact=output.model_dump(mode="json"),
    )


@tool
async def price_compare(
    runtime: ToolRuntime,
    base_currency: str = "CNY",
    top_n: int = 12,
) -> Command:
    """合并当前 checkpoint 中的候选，统一币种并做跨平台比价。必须先有 ItemSearch 结果。"""

    search_outputs = runtime.state.get("search_outputs") or {}
    candidates = [
        candidate
        for output in search_outputs.values()
        for candidate in ItemSearchOutput.model_validate(output).candidates
    ]
    if not candidates:
        return _command(
            runtime,
            tool_name="price_compare",
            content="当前没有搜索候选。请先调用 item_search 或 dispatch_tool。",
        )
    compared = await _price_compare(
        candidates,
        base_currency=base_currency,
        top_n=top_n,
    )
    return _command(
        runtime,
        tool_name="price_compare",
        content=_json_result(compared),
        updates={"compared": compared},
        artifact=compared.model_dump(mode="json"),
    )


@tool
async def shipping_calc(
    runtime: ToolRuntime,
    destination: str = "CN",
) -> Command:
    """根据最近一次比价结果估算运费、关税、到手价和时效。必须先调用 price_compare。"""

    compared = runtime.state.get("compared")
    if compared is None:
        return _command(
            runtime,
            tool_name="shipping_calc",
            content="当前没有比价结果。请先调用 price_compare。",
        )
    points = compared.ranked if hasattr(compared, "ranked") else compared["ranked"]
    shipping = await _shipping_calc(points, destination=destination)
    return _command(
        runtime,
        tool_name="shipping_calc",
        content=_json_result(shipping),
        updates={"shipping": shipping},
        artifact=shipping.model_dump(mode="json"),
    )


@tool
async def item_picker(
    runtime: ToolRuntime,
    top_n: int = 3,
) -> Command:
    """根据预算、排除项、长期偏好和品类知识，从到手价候选中精排 1-5 件商品。"""

    state = runtime.state
    plan = state.get("plan")
    insight = state.get("insight")
    shipping = state.get("shipping")
    if plan is None:
        return _command(runtime, tool_name="item_picker", content="缺少购物计划。请先调用 planner。")
    if insight is None:
        return _command(runtime, tool_name="item_picker", content="缺少品类知识。请先调用 category_insight。")
    if shipping is None:
        return _command(runtime, tool_name="item_picker", content="缺少到手价结果。请先调用 shipping_calc。")

    plan = QueryPlan.model_validate(plan)
    hard = list(
        dict.fromkeys(
            [*(state.get("long_term_preferences") or []), *plan.hard_constraints]
        )
    )
    soft = list(
        dict.fromkeys(
            [*(state.get("long_term_preferences") or []), *plan.soft_preferences]
        )
    )
    landed = shipping.items if hasattr(shipping, "items") else shipping["items"]
    picker = await _item_picker(
        landed,
        insight=insight,
        hard_constraints=hard,
        soft_preferences=soft,
        budget_cny=plan.budget_cny,
        top_n=top_n,
    )
    return _command(
        runtime,
        tool_name="item_picker",
        content=_json_result(picker),
        updates={"picker": picker},
        artifact=picker.model_dump(mode="json"),
    )


@tool(return_direct=True)
async def shopping_summary(runtime: ToolRuntime) -> Command:
    """生成最终 Markdown 购物清单，写入 checkpoint，并立即结束 AgentLoop。"""

    state = runtime.state
    plan = state.get("plan")
    picker = state.get("picker")
    if plan is None:
        return _command(runtime, tool_name="shopping_summary", content="缺少购物计划。请先调用 planner。")
    if picker is None:
        return _command(runtime, tool_name="shopping_summary", content="缺少精排结果。请先调用 item_picker。")

    insight = state.get("insight")
    summary = await _shopping_summary(
        picker,
        QueryPlan.model_validate(plan),
        CategoryInsightOutput.model_validate(insight) if insight is not None else None,
    )
    return _command(
        runtime,
        tool_name="shopping_summary",
        content=summary.final_text,
        updates={
            "summary": summary,
            "terminated": True,
            "terminal_tool": "shopping_summary",
        },
        artifact=summary.model_dump(mode="json"),
    )


@tool
async def dispatch_tool(
    demands: str,
    runtime: ToolRuntime,
    platform: Platform | None = None,
    category: str | None = None,
) -> Command:
    """按需 fork 同质子 AgentLoop；平台和品类范围由 checkpoint state 强制执行。"""

    from app.api.monitor import monitor

    started = time.perf_counter()
    await monitor.report_tool_start(
        "dispatch_tool",
        {
            "demands": demands[:500],
            "platform": platform,
            "category": category,
        },
    )
    if runtime.state.get("is_sub_agent"):
        await monitor.report_tool_end(
            "dispatch_tool", int((time.perf_counter() - started) * 1000)
        )
        return _command(
            runtime,
            tool_name="dispatch_tool",
            content="子 Agent 是叶子执行器，不允许递归调用 dispatch_tool；请直接完成当前范围内的检索。",
        )

    result = await safe_run_sub_agent(
        demands=demands,
        platform=platform,
        category=category,
        parent_state=runtime.state,
    )
    await monitor.report_tool_end(
        "dispatch_tool", int((time.perf_counter() - started) * 1000)
    )
    if isinstance(result, str):
        return _command(runtime, tool_name="dispatch_tool", content=result)

    assert isinstance(result, SubAgentRun)
    child_state = result.state
    child_search = child_state.get("search_outputs") or {}
    child_insight = child_state.get("insight")
    artifact = result.artifact()
    updates: dict[str, Any] = {
        "search_outputs": child_search,
        "sub_agent_results": {result.sub_thread_id: artifact},
    }
    if child_insight is not None:
        updates["insight"] = child_insight
    return _command(
        runtime,
        tool_name="dispatch_tool",
        content=_json_result(
            {
                "sub_thread_id": result.sub_thread_id,
                "final_answer": result.final_answer,
                "search_platforms": list(child_search),
                "terminated": bool(child_state.get("terminated")),
            }
        ),
        updates=updates,
        artifact=artifact,
    )


@tool("item_search", return_direct=True)
async def scoped_sub_agent_item_search(
    runtime: ToolRuntime,
    top_k: int = 20,
) -> Command:
    """Search the single platform enforced by a scoped leaf sub-agent state."""

    allowed = [str(value) for value in (runtime.state.get("allowed_platforms") or [])]
    if len(allowed) != 1:
        return _command(
            runtime,
            tool_name="item_search",
            content=(
                "scoped leaf 子 Agent 必须且只能绑定一个平台；"
                f"当前 allowed_platforms={allowed}。"
            ),
        )
    return await item_search.coroutine(
        platform=allowed[0],
        runtime=runtime,
        query="",
        category="",
        top_k=top_k,
    )


@lru_cache(maxsize=1)
def get_sub_agent_tool_set():
    """Return the minimal terminal tool set for one-platform leaf workers."""

    return [scoped_sub_agent_item_search]


@lru_cache(maxsize=1)
def get_full_tool_set():
    """Return the registered tool objects for the main AgentLoop."""

    return [
        planner,
        chat_fallback,
        web_search,
        category_insight,
        item_search,
        item_picker,
        price_compare,
        shipping_calc,
        shopping_summary,
        dispatch_tool,
    ]


@lru_cache(maxsize=1)
def get_tools_by_name() -> dict[str, Any]:
    return {tool_object.name: tool_object for tool_object in get_full_tool_set()}


TERMINAL_TOOLS = {"shopping_summary", "chat_fallback"}
