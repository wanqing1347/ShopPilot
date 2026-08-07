from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.messages import AIMessage, AIMessageChunk

from app.agent.checkpoint import get_checkpointer
from app.agent.context_governance import CacheBreakpointMiddleware
from app.agent.llm import get_llm
from app.agent.settings import max_model_steps, max_tool_calls
from app.agent.state import ShopPilotState
from app.agent.tool_reliability import ToolReliabilityMiddleware
from app.api.monitor import monitor


@dataclass
class AgentRunResult:
    final_text: str
    state: ShopPilotState | dict[str, Any]
    resumed: bool = False


def _plan_platforms(state: ShopPilotState | dict[str, Any]) -> list[str]:
    plan = state.get("plan")
    if plan is None:
        return []
    platforms = getattr(plan, "platforms", None)
    if platforms is None and isinstance(plan, dict):
        platforms = plan.get("platforms")
    return [str(platform) for platform in (platforms or []) if platform]


def _missing_search_platforms(state: ShopPilotState | dict[str, Any]) -> list[str]:
    searched = {str(platform) for platform in (state.get("search_outputs") or {}).keys()}
    scoped = [str(platform) for platform in (state.get("allowed_platforms") or []) if platform]
    expected = scoped or _plan_platforms(state)
    return [platform for platform in expected if platform not in searched]


def select_tool_names(state: ShopPilotState | dict[str, Any]) -> list[str]:
    """Apply stage-aware tool permissions while retaining deterministic recovery.

    Multi-platform shopping is not considered ready for comparison until every
    planned/scoped platform has a search result. Sub-agents are leaf workers:
    they may gather insight/search evidence but must not recursively fork another
    dispatch_tool, which avoids self-deduplication waits and fork deadlocks.
    """

    if state.get("terminated"):
        return []
    if state.get("plan") is None:
        return ["planner", "chat_fallback", "web_search"]

    is_sub_agent = bool(state.get("is_sub_agent"))
    missing_platforms = _missing_search_platforms(state)
    if missing_platforms or not state.get("search_outputs"):
        tools = ["planner", "category_insight", "item_search", "web_search"]
        if not is_sub_agent:
            tools.insert(3, "dispatch_tool")
        return tools

    # A child fork only needs to return scoped search/insight artifacts. Running
    # compare -> shipping -> picker -> summary inside the child duplicates work
    # that the parent performs after all child results are merged.
    if is_sub_agent:
        return []

    if state.get("compared") is None:
        return ["category_insight", "item_search", "dispatch_tool", "price_compare", "web_search"]
    if state.get("shipping") is None:
        return ["item_search", "dispatch_tool", "price_compare", "shipping_calc"]
    if state.get("picker") is None:
        return ["category_insight", "price_compare", "shipping_calc", "item_picker"]
    if state.get("summary") is None:
        return ["item_picker", "shopping_summary"]
    return []


@wrap_model_call
async def stage_tool_permissions(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Any],
) -> ModelResponse:
    """Expose only tools allowed by the current checkpointed workflow state."""

    from app.agent.tool_registry import get_tools_by_name

    tools_by_name = get_tools_by_name()
    selected = [
        tools_by_name[name]
        for name in select_tool_names(request.state)
        if name in tools_by_name
    ]
    return await handler(request.override(tools=selected))


async def build_agent(system_prompt: str):
    """Build the LangChain v1 agent backed by persistent LangGraph state."""

    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

    from app.agent.tool_registry import get_full_tool_set

    model = get_llm()
    return create_agent(
        model=model,
        tools=get_full_tool_set(),
        system_prompt=system_prompt,
        state_schema=ShopPilotState,
        checkpointer=await get_checkpointer(),
        middleware=[
            stage_tool_permissions,
            CacheBreakpointMiddleware(model),
            ModelCallLimitMiddleware(
                run_limit=max_model_steps(),
                exit_behavior="end",
            ),
            # Keep the call limiter outside reliability retries: one model-issued
            # tool call consumes one budget unit, regardless of transport retries.
            ToolCallLimitMiddleware(
                run_limit=max_tool_calls(),
                exit_behavior="continue",
            ),
            ToolReliabilityMiddleware(),
        ],
    )


def _langfuse_callbacks() -> list[object]:
    if not (
        os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        and os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    ):
        return []
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return []
    return [CallbackHandler()]


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


def extract_final_text(state: ShopPilotState | dict[str, Any]) -> str:
    summary = state.get("summary")
    if summary is not None:
        if hasattr(summary, "final_text"):
            return str(summary.final_text).strip()
        if isinstance(summary, dict) and summary.get("final_text"):
            return str(summary["final_text"]).strip()

    messages = state.get("messages") or []
    if not messages:
        return "Agent 未返回消息。"
    message = messages[-1]
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return _content_to_text(content).strip() or "Agent 已结束，但最终消息为空。"


def _last_ai_message(update: Any) -> AIMessage | None:
    if not isinstance(update, dict):
        return None
    messages = update.get("messages") or []
    if not messages:
        return None
    candidate = messages[-1]
    return candidate if isinstance(candidate, AIMessage) else None


def _run_config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": max_model_steps() * 2 + 1,
        "callbacks": _langfuse_callbacks(),
        "metadata": {
            "langfuse_session_id": thread_id,
            "langfuse_tags": ["shoppilot", "agentloop"],
        },
    }


async def _stream_updates(agent: Any, graph_input: Any, config: dict[str, Any]) -> None:
    model_step = 0
    token_counts: dict[str, int] = {}
    async for chunk in agent.astream(
        graph_input,
        config=config,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        if not isinstance(chunk, dict):
            continue
        chunk_type = chunk.get("type")
        if chunk_type == "messages":
            data = chunk.get("data")
            if not isinstance(data, (tuple, list)) or len(data) != 2:
                continue
            token, metadata = data
            if not isinstance(token, AIMessageChunk) or not isinstance(metadata, dict):
                continue
            node = str(metadata.get("langgraph_node") or "")
            tags = {str(tag) for tag in metadata.get("tags") or []}
            # Only expose the main Agent model. Structured-output LLM calls inside
            # tools and context-compaction summaries must not leak into the answer.
            if node != "model" or "shoppilot-context-compaction" in tags:
                continue
            delta = token.text
            if not delta:
                continue
            message_id = str(token.id) if token.id else None
            token_key = message_id or f"{node}:anonymous"
            token_index = token_counts.get(token_key, 0) + 1
            token_counts[token_key] = token_index
            await monitor.report_assistant_token(
                delta=delta,
                message_id=message_id,
                node=node,
                token_index=token_index,
            )
            continue

        if chunk_type != "updates":
            continue
        data = chunk.get("data") or {}
        if not isinstance(data, dict):
            continue
        for update in data.values():
            message = _last_ai_message(update)
            if message is None:
                continue
            model_step += 1
            tool_calls = [str(call.get("name")) for call in message.tool_calls]
            await monitor.report_assistant_call(
                step=model_step,
                tool_calls=tool_calls,
                preview=_content_to_text(message.content),
            )


async def ainvoke_agent(
    *,
    query: str,
    thread_id: str,
    system_prompt: str,
    initial: ShopPilotState | dict[str, Any],
) -> AgentRunResult:
    """Start a fresh checkpointed AgentLoop and return its final state."""

    await monitor.report_stage("think", "Agent 正在根据 checkpoint state 选择下一步工具")
    agent = await build_agent(system_prompt)
    config = _run_config(thread_id)
    input_state = dict(initial)
    input_state["messages"] = [{"role": "user", "content": query}]

    await _stream_updates(agent, input_state, config)
    snapshot = await agent.aget_state(config)
    final_state = dict(snapshot.values)
    await monitor.report_stage("reflect", "AgentLoop 已满足终止条件")
    return AgentRunResult(
        final_text=extract_final_text(final_state),
        state=final_state,
        resumed=False,
    )


async def aresume_agent(
    *,
    thread_id: str,
    system_prompt: str,
) -> AgentRunResult:
    """Resume one unfinished thread from its latest persistent checkpoint."""

    agent = await build_agent(system_prompt)
    config = _run_config(thread_id)
    snapshot = await agent.aget_state(config)
    if not snapshot.values:
        raise LookupError(f"checkpoint 不存在：{thread_id}")

    if snapshot.next:
        await monitor.report_stage(
            "resume",
            "从 SQLite checkpoint 恢复未完成的 AgentLoop",
            next_nodes=list(snapshot.next),
        )
        await _stream_updates(agent, None, config)
        snapshot = await agent.aget_state(config)
    else:
        await monitor.report_stage(
            "resume",
            "checkpoint 已处于完成状态，直接返回持久化结果",
        )

    final_state = dict(snapshot.values)
    return AgentRunResult(
        final_text=extract_final_text(final_state),
        state=final_state,
        resumed=True,
    )
