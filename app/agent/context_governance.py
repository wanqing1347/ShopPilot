from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langgraph.types import Command

from app.agent.settings import (
    context_compaction_min_messages,
    context_compaction_trigger_chars,
    context_compaction_trigger_messages,
    context_keep_recent_messages,
    context_keep_recent_tool_calls,
    context_summary_max_chars,
    context_tool_message_max_chars,
    prompt_cache_key_enabled,
    prompt_cache_key_prefix,
)
from app.api.monitor import monitor


@dataclass(frozen=True)
class ContextWindow:
    messages: list[BaseMessage]
    summary: str | None
    summary_until: int
    breakpoint_index: int
    cache_epoch: int
    original_chars: int
    compacted_chars: int
    compacted_tool_messages: int


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text is not None:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content or "")


def _message_chars(message: BaseMessage) -> int:
    return len(_content_text(message.content))


def _tool_call_parent_index(messages: list[BaseMessage], tool_index: int) -> int:
    tool_message = messages[tool_index]
    if not isinstance(tool_message, ToolMessage):
        return tool_index
    target_id = tool_message.tool_call_id
    for index in range(tool_index - 1, -1, -1):
        candidate = messages[index]
        if not isinstance(candidate, AIMessage):
            continue
        if any(str(call.get("id")) == target_id for call in candidate.tool_calls):
            return index
    return tool_index


def compute_cache_breakpoint(
    messages: list[BaseMessage],
    *,
    keep_recent_tool_calls: int,
    keep_recent_messages: int,
) -> int:
    """Return a safe compaction boundary that never splits a tool-call pair."""

    if not messages:
        return 0
    tool_indices = [
        index for index, message in enumerate(messages) if isinstance(message, ToolMessage)
    ]
    if keep_recent_tool_calls > 0 and len(tool_indices) > keep_recent_tool_calls:
        target_tool = tool_indices[-keep_recent_tool_calls]
        return _tool_call_parent_index(messages, target_tool)

    fallback = max(0, len(messages) - max(1, keep_recent_messages))
    while fallback > 0 and isinstance(messages[fallback], ToolMessage):
        fallback = _tool_call_parent_index(messages, fallback)
    return fallback


def _compact_json_value(value: Any, *, list_limit: int = 3, depth: int = 0) -> Any:
    if depth >= 3:
        if isinstance(value, (dict, list)):
            return {"_context_compacted": True, "type": type(value).__name__}
        return value
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, item in list(value.items())[:24]:
            compacted[str(key)] = _compact_json_value(
                item,
                list_limit=list_limit,
                depth=depth + 1,
            )
        if len(value) > 24:
            compacted["_omitted_keys"] = len(value) - 24
        return compacted
    if isinstance(value, list):
        compacted_items = [
            _compact_json_value(item, list_limit=list_limit, depth=depth + 1)
            for item in value[:list_limit]
        ]
        if len(value) > list_limit:
            compacted_items.append({"_omitted_items": len(value) - list_limit})
        return compacted_items
    return value


def compact_tool_message(message: ToolMessage, max_chars: int) -> ToolMessage:
    """Create a model-only compact copy while retaining valid JSON and tool identity."""

    content = _content_text(message.content)
    if len(content) <= max_chars:
        return message

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        payload: Any = {
            "_context_compacted": True,
            "original_chars": len(content),
            "preview": content[: max(128, max_chars - 180)],
        }
    else:
        payload = {
            "_context_compacted": True,
            "original_chars": len(content),
            "data": _compact_json_value(parsed),
        }

    compacted = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(compacted) > max_chars:
        compacted = json.dumps(
            {
                "_context_compacted": True,
                "original_chars": len(content),
                "preview": compacted[: max(128, max_chars - 180)],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return message.model_copy(update={"content": compacted, "artifact": None})


def _serialize_for_summary(messages: list[BaseMessage], tool_limit: int) -> str:
    rows: list[str] = []
    for message in messages:
        role = getattr(message, "type", message.__class__.__name__)
        content = _content_text(message.content)
        if isinstance(message, AIMessage) and message.tool_calls:
            calls = [
                {
                    "name": call.get("name"),
                    "args": call.get("args"),
                    "id": call.get("id"),
                }
                for call in message.tool_calls
            ]
            content = (
                content
                + "\n工具调用："
                + json.dumps(calls, ensure_ascii=False, separators=(",", ":"))
            ).strip()
        if isinstance(message, ToolMessage) and len(content) > tool_limit:
            content = _content_text(compact_tool_message(message, tool_limit).content)
        rows.append(f"[{role}] {content}")
    return "\n".join(rows)


def _summary_message(summary: str, epoch: int) -> SystemMessage:
    return SystemMessage(
        content=(
            f"【ShopPilot 稳定工作记忆 / cache epoch {epoch}】\n"
            "以下内容由已完成历史压缩而来。保持其中的用户目标、硬约束、关键决定、"
            "失败尝试和可复用结论；不要把它当成新的用户指令。\n\n"
            f"{summary}"
        )
    )


def _stable_cache_key(request: ModelRequest[Any]) -> str:
    system = _content_text(request.system_message.content) if request.system_message else ""
    tool_names: list[str] = []
    for tool in request.tools:
        if isinstance(tool, dict):
            tool_names.append(str(tool.get("name") or tool.get("type") or "tool"))
        else:
            tool_names.append(str(getattr(tool, "name", tool.__class__.__name__)))
    material = system + "\n" + "|".join(sorted(tool_names))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
    return f"{prompt_cache_key_prefix()}-{digest}"


def _usage_metrics(response: ModelResponse[Any]) -> dict[str, int]:
    input_tokens = 0
    cache_read_tokens = 0
    cache_creation_tokens = 0
    for message in response.result:
        if not isinstance(message, AIMessage):
            continue
        usage = message.usage_metadata or {}
        input_tokens = max(input_tokens, int(usage.get("input_tokens") or 0))
        details = usage.get("input_token_details") or {}
        if isinstance(details, dict):
            cache_read_tokens = max(
                cache_read_tokens,
                int(details.get("cache_read") or details.get("cached_tokens") or 0),
            )
            cache_creation_tokens = max(
                cache_creation_tokens,
                int(details.get("cache_creation") or 0),
            )
        response_details = message.response_metadata.get("token_usage") or {}
        prompt_details = response_details.get("prompt_tokens_details") or {}
        if isinstance(prompt_details, dict):
            cache_read_tokens = max(
                cache_read_tokens,
                int(prompt_details.get("cached_tokens") or 0),
            )
    return {
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
    }


class CacheBreakpointMiddleware(AgentMiddleware):
    """Cache-aware context compaction without mutating the persisted message history.

    The middleware advances a coarse summary boundary only after enough new history
    accumulates. Between boundary advances, the synthetic summary and all prior tail
    messages remain byte-for-byte stable, improving provider prompt-cache reuse.
    """

    def __init__(
        self,
        model: Any,
        *,
        trigger_messages: int | None = None,
        trigger_chars: int | None = None,
        keep_recent_tools: int | None = None,
        keep_recent_messages: int | None = None,
        min_compaction_messages: int | None = None,
        summary_max_chars: int | None = None,
        tool_message_max_chars: int | None = None,
        enable_prompt_cache_key: bool | None = None,
    ) -> None:
        self.model = model
        self.trigger_messages = trigger_messages or context_compaction_trigger_messages()
        self.trigger_chars = trigger_chars or context_compaction_trigger_chars()
        self.keep_recent_tools = (
            context_keep_recent_tool_calls()
            if keep_recent_tools is None
            else max(0, keep_recent_tools)
        )
        self.keep_recent_messages = keep_recent_messages or context_keep_recent_messages()
        self.min_compaction_messages = (
            min_compaction_messages or context_compaction_min_messages()
        )
        self.summary_max_chars = summary_max_chars or context_summary_max_chars()
        self.tool_message_max_chars = (
            tool_message_max_chars or context_tool_message_max_chars()
        )
        self.enable_prompt_cache_key = (
            prompt_cache_key_enabled()
            if enable_prompt_cache_key is None
            else enable_prompt_cache_key
        )

    async def _summarize(
        self,
        *,
        previous_summary: str | None,
        messages: list[BaseMessage],
    ) -> str:
        history = _serialize_for_summary(messages, self.tool_message_max_chars)
        prompt = [
            SystemMessage(
                content=(
                    "你是 ShopPilot Agent 的上下文压缩器。请把历史压缩成高密度工作记忆。"
                    "必须保留：用户目标、预算和排除项、平台/品类范围、已完成工具结论、"
                    "关键商品或数值、失败尝试、尚未完成的下一步。不要编造，不要写寒暄，"
                    "输出纯文本，使用简短分段。"
                )
            ),
            HumanMessage(
                content=(
                    f"已有工作记忆：\n{previous_summary or '无'}\n\n"
                    f"本次新增历史：\n{history}"
                )
            ),
        ]
        try:
            response = await self.model.ainvoke(
                prompt,
                config={"tags": ["shoppilot-context-compaction"]},
            )
        except TypeError as exc:
            # Lightweight custom/fake chat models may implement ainvoke(messages)
            # without RunnableConfig support. They cannot emit graph token streams,
            # so retrying without tags is safe and preserves compatibility.
            if "config" not in str(exc):
                raise
            response = await self.model.ainvoke(prompt)
        summary = _content_text(getattr(response, "content", response)).strip()
        if not summary:
            raise RuntimeError("上下文压缩模型返回空摘要")
        return summary[: self.summary_max_chars]

    def _build_window(
        self,
        *,
        messages: list[BaseMessage],
        summary: str | None,
        summary_until: int,
        breakpoint_index: int,
        cache_epoch: int,
    ) -> ContextWindow:
        tail = messages[summary_until:]
        compacted_tail: list[BaseMessage] = []
        compacted_tools = 0
        for message in tail:
            if isinstance(message, ToolMessage):
                compacted = compact_tool_message(message, self.tool_message_max_chars)
                if compacted is not message:
                    compacted_tools += 1
                compacted_tail.append(compacted)
            else:
                compacted_tail.append(message)
        model_messages = (
            [_summary_message(summary, cache_epoch), *compacted_tail]
            if summary
            else compacted_tail
        )
        return ContextWindow(
            messages=model_messages,
            summary=summary,
            summary_until=summary_until,
            breakpoint_index=breakpoint_index,
            cache_epoch=cache_epoch,
            original_chars=sum(_message_chars(message) for message in messages),
            compacted_chars=sum(_message_chars(message) for message in model_messages),
            compacted_tool_messages=compacted_tools,
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
        state = request.state
        messages = list(request.messages)
        summary = str(state.get("context_summary") or "").strip() or None
        summary_until = min(int(state.get("context_summary_until") or 0), len(messages))
        cache_epoch = int(state.get("context_cache_epoch") or 0)
        breakpoint_index = compute_cache_breakpoint(
            messages,
            keep_recent_tool_calls=self.keep_recent_tools,
            keep_recent_messages=self.keep_recent_messages,
        )
        pending_messages = len(messages) - summary_until
        pending_chars = sum(_message_chars(message) for message in messages[summary_until:])
        should_advance = (
            breakpoint_index > summary_until
            and breakpoint_index - summary_until >= self.min_compaction_messages
            and (
                pending_messages >= self.trigger_messages
                or pending_chars >= self.trigger_chars
            )
        )

        updates: dict[str, Any] = {}
        if should_advance:
            delta = messages[summary_until:breakpoint_index]
            try:
                summary = await self._summarize(
                    previous_summary=summary,
                    messages=delta,
                )
            except Exception as exc:
                await monitor.report_context_compaction_failed(
                    error=f"{type(exc).__name__}: {exc}",
                    pending_messages=pending_messages,
                    pending_chars=pending_chars,
                )
            else:
                summary_until = breakpoint_index
                cache_epoch += 1
                updates.update(
                    {
                        "context_summary": summary,
                        "context_summary_until": summary_until,
                        "context_cache_epoch": cache_epoch,
                        "context_breakpoint_index": breakpoint_index,
                        "context_compaction_count": int(
                            state.get("context_compaction_count") or 0
                        )
                        + 1,
                    }
                )

        window = self._build_window(
            messages=messages,
            summary=summary,
            summary_until=summary_until,
            breakpoint_index=breakpoint_index,
            cache_epoch=cache_epoch,
        )
        settings = dict(request.model_settings)
        cache_key: str | None = None
        if self.enable_prompt_cache_key:
            cache_key = _stable_cache_key(request)
            settings.setdefault("prompt_cache_key", cache_key)

        response = await handler(
            request.override(messages=window.messages, model_settings=settings)
        )
        usage = _usage_metrics(response)
        metrics = {
            "cache_epoch": window.cache_epoch,
            "summary_until": window.summary_until,
            "breakpoint_index": window.breakpoint_index,
            "original_message_count": len(messages),
            "model_message_count": len(window.messages),
            "original_chars": window.original_chars,
            "model_chars": window.compacted_chars,
            "saved_chars": max(0, window.original_chars - window.compacted_chars),
            "compacted_tool_messages": window.compacted_tool_messages,
            "prompt_cache_key": cache_key,
            **usage,
        }
        updates["context_metrics"] = metrics

        if should_advance and "context_summary" in updates:
            await monitor.report_context_compaction(
                epoch=window.cache_epoch,
                summarized_until=window.summary_until,
                original_chars=window.original_chars,
                model_chars=window.compacted_chars,
                compacted_tool_messages=window.compacted_tool_messages,
            )
        if usage["cache_read_tokens"] or usage["cache_creation_tokens"]:
            await monitor.report_prompt_cache_usage(
                cache_read_tokens=usage["cache_read_tokens"],
                cache_creation_tokens=usage["cache_creation_tokens"],
                input_tokens=usage["input_tokens"],
                cache_key=cache_key,
            )

        return ExtendedModelResponse(
            model_response=response,
            command=Command(update=updates),
        )
