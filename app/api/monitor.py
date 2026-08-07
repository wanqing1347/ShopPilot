from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.api.connection import manager
from app.api.context import get_root_thread_id, get_thread_id


class Monitor:
    async def _emit(
        self,
        event: str,
        message: str,
        data: dict[str, Any],
        *,
        record_history: bool = True,
    ) -> None:
        route_thread_id = get_root_thread_id()
        if route_thread_id is None:
            return
        actor_thread_id = get_thread_id()
        payload = {
            "type": "monitor_event",
            "event": event,
            "message": message,
            "data": {**data, "actor_thread_id": actor_thread_id},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await manager.send_to_thread(
            payload,
            route_thread_id,
            record_history=record_history,
        )

    async def report_stage(self, stage: str, message: str, **data: Any) -> None:
        await self._emit("stage", message, {"stage": stage, **data})

    async def report_assistant_token(
        self,
        *,
        delta: str,
        message_id: str | None,
        node: str,
        token_index: int,
    ) -> None:
        if not delta:
            return
        await self._emit(
            "assistant_token",
            "模型正在流式输出",
            {
                "delta": delta,
                "message_id": message_id,
                "node": node,
                "token_index": token_index,
            },
            # Token deltas are high-volume and should not evict durable lifecycle
            # events from the replay/trace ring buffer.
            record_history=False,
        )

    async def report_assistant_call(
        self,
        *,
        step: int,
        tool_calls: list[str] | None = None,
        preview: str = "",
    ) -> None:
        calls = tool_calls or []
        message = (
            f"Agent 第 {step} 轮决定调用：{', '.join(calls)}"
            if calls
            else f"Agent 第 {step} 轮正在生成回答"
        )
        await self._emit(
            "assistant_call",
            message,
            {"step": step, "tool_calls": calls, "preview": preview[:500]},
        )

    async def report_tool_start(self, tool_name: str, args: dict[str, Any]) -> None:
        await self._emit(
            "tool_start",
            f"正在调用 {tool_name}",
            {"tool_name": tool_name, "args": args},
        )

    async def report_tool_end(self, tool_name: str, duration_ms: int) -> None:
        await self._emit(
            "tool_end",
            f"{tool_name} 完成",
            {"tool_name": tool_name, "duration_ms": duration_ms},
        )

    async def report_knowledge_retrieval(
        self,
        *,
        category_key: str,
        returned_count: int,
        evidence_sufficient: bool,
        embedding_provider: str,
        vector_engine: str,
        duration_ms: int,
    ) -> None:
        await self._emit(
            "knowledge_retrieval",
            f"品类知识检索返回 {returned_count} 条证据",
            {
                "category_key": category_key,
                "returned_count": returned_count,
                "evidence_sufficient": evidence_sufficient,
                "embedding_provider": embedding_provider,
                "vector_engine": vector_engine,
                "duration_ms": duration_ms,
            },
        )

    async def report_knowledge_synthesis(
        self,
        *,
        category_key: str,
        success: bool,
        claim_count: int,
        invalid_claim_count: int,
        attempts: int,
        duration_ms: int,
        fallback_reason: str | None = None,
    ) -> None:
        await self._emit(
            "knowledge_synthesis",
            "品类知识引用生成完成" if success else "品类知识生成已安全降级",
            {
                "category_key": category_key,
                "success": success,
                "claim_count": claim_count,
                "invalid_claim_count": invalid_claim_count,
                "attempts": attempts,
                "duration_ms": duration_ms,
                "fallback_reason": fallback_reason,
            },
        )

    async def report_retrieval_search(
        self,
        *,
        platform: str,
        category_key: str,
        mode: str,
        embedding_provider: str,
        vector_engine: str,
        reranker: str,
        eligible_count: int,
        returned_count: int,
        duration_ms: int,
    ) -> None:
        source_label = "官方 API" if mode == "official_api" else "Hybrid Retrieval"
        await self._emit(
            "retrieval_search",
            f"{platform} {source_label} 返回 {returned_count} 条候选",
            {
                "platform": platform,
                "category_key": category_key,
                "mode": mode,
                "embedding_provider": embedding_provider,
                "vector_engine": vector_engine,
                "reranker": reranker,
                "eligible_count": eligible_count,
                "returned_count": returned_count,
                "duration_ms": duration_ms,
            },
        )

    async def report_tool_attempt(
        self,
        *,
        tool_name: str,
        attempt: int,
        max_attempts: int,
        timeout_sec: float,
        idempotency_key: str,
    ) -> None:
        await self._emit(
            "tool_attempt",
            f"{tool_name} 第 {attempt}/{max_attempts} 次执行",
            {
                "tool_name": tool_name,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "timeout_sec": timeout_sec,
                "idempotency_key": idempotency_key,
            },
        )

    async def report_tool_retry(
        self,
        *,
        tool_name: str,
        attempt: int,
        next_attempt: int,
        category: str,
        delay_sec: float,
        error: str,
    ) -> None:
        await self._emit(
            "tool_retry",
            f"{tool_name} 将在退避后重试",
            {
                "tool_name": tool_name,
                "attempt": attempt,
                "next_attempt": next_attempt,
                "category": category,
                "delay_sec": round(delay_sec, 3),
                "error": error[:1000],
            },
        )

    async def report_tool_success(
        self,
        *,
        tool_name: str,
        attempts: int,
        duration_ms: int,
        source: str,
    ) -> None:
        await self._emit(
            "tool_success",
            f"{tool_name} 执行成功",
            {
                "tool_name": tool_name,
                "attempts": attempts,
                "duration_ms": duration_ms,
                "source": source,
            },
        )

    async def report_tool_failure(
        self,
        *,
        tool_name: str,
        category: str,
        attempts: int,
        duration_ms: int,
        retryable: bool,
        error: str,
    ) -> None:
        await self._emit(
            "tool_failure",
            f"{tool_name} 执行失败：{category}",
            {
                "tool_name": tool_name,
                "category": category,
                "attempts": attempts,
                "duration_ms": duration_ms,
                "retryable": retryable,
                "error": error[:1000],
                "recoverable": True,
            },
        )

    async def report_tool_idempotency_hit(
        self,
        *,
        tool_name: str,
        idempotency_key: str,
        source: str,
    ) -> None:
        await self._emit(
            "tool_idempotency_hit",
            f"{tool_name} 复用幂等结果",
            {
                "tool_name": tool_name,
                "idempotency_key": idempotency_key,
                "source": source,
            },
        )

    async def report_tool_circuit_open(
        self,
        *,
        tool_name: str,
        failures: int,
        reset_sec: float,
        category: str,
    ) -> None:
        await self._emit(
            "tool_circuit_open",
            f"{tool_name} 熔断器已打开",
            {
                "tool_name": tool_name,
                "failures": failures,
                "reset_sec": reset_sec,
                "category": category,
            },
        )

    async def report_tool_circuit_rejected(
        self,
        *,
        tool_name: str,
        retry_after_sec: float,
    ) -> None:
        await self._emit(
            "tool_circuit_rejected",
            f"{tool_name} 被熔断器快速拒绝",
            {
                "tool_name": tool_name,
                "retry_after_sec": round(retry_after_sec, 3),
                "recoverable": True,
            },
        )

    async def report_tool_circuit_recovered(self, *, tool_name: str) -> None:
        await self._emit(
            "tool_circuit_recovered",
            f"{tool_name} 熔断器已恢复",
            {"tool_name": tool_name},
        )

    async def report_fork(self, sub_thread_id: str, demands: str) -> None:
        await self._emit(
            "fork",
            "派发同质子 AgentLoop",
            {"sub_thread_id": sub_thread_id, "demands": demands[:300]},
        )

    async def report_fork_queued(
        self,
        *,
        fingerprint: str,
        platform: str | None,
        category: str | None,
        global_position: int,
        task_position: int,
    ) -> None:
        await self._emit(
            "fork_queued",
            "子 Agent 已进入并发队列",
            {
                "fingerprint": fingerprint,
                "platform": platform,
                "category": category,
                "global_position": global_position,
                "task_position": task_position,
            },
        )

    async def report_fork_dequeued(
        self,
        *,
        fingerprint: str,
        platform: str | None,
        category: str | None,
        wait_ms: int,
    ) -> None:
        await self._emit(
            "fork_dequeued",
            "子 Agent 获得并发额度",
            {
                "fingerprint": fingerprint,
                "platform": platform,
                "category": category,
                "wait_ms": wait_ms,
            },
        )

    async def report_fork_deduplicated(
        self,
        *,
        fingerprint: str,
        platform: str | None,
        category: str | None,
        source: str,
    ) -> None:
        await self._emit(
            "fork_deduplicated",
            "复用等价子任务，跳过重复 fork",
            {
                "fingerprint": fingerprint,
                "platform": platform,
                "category": category,
                "source": source,
            },
        )

    async def report_fork_rejected(self, reason: str, message: str) -> None:
        await self._emit(
            "fork_rejected",
            message,
            {"reason": reason, "recoverable": True},
        )

    async def report_context_compaction(
        self,
        *,
        epoch: int,
        summarized_until: int,
        original_chars: int,
        model_chars: int,
        compacted_tool_messages: int,
    ) -> None:
        await self._emit(
            "context_compaction",
            f"上下文进入 cache epoch {epoch}",
            {
                "epoch": epoch,
                "summarized_until": summarized_until,
                "original_chars": original_chars,
                "model_chars": model_chars,
                "saved_chars": max(0, original_chars - model_chars),
                "compacted_tool_messages": compacted_tool_messages,
            },
        )

    async def report_context_compaction_failed(
        self,
        *,
        error: str,
        pending_messages: int,
        pending_chars: int,
    ) -> None:
        await self._emit(
            "context_compaction_failed",
            "上下文压缩失败，本轮保留完整历史继续执行",
            {
                "error": error,
                "pending_messages": pending_messages,
                "pending_chars": pending_chars,
            },
        )

    async def report_prompt_cache_usage(
        self,
        *,
        cache_read_tokens: int,
        cache_creation_tokens: int,
        input_tokens: int,
        cache_key: str | None,
    ) -> None:
        await self._emit(
            "prompt_cache",
            f"Prompt Cache 命中 {cache_read_tokens} tokens",
            {
                "cache_read_tokens": cache_read_tokens,
                "cache_creation_tokens": cache_creation_tokens,
                "input_tokens": input_tokens,
                "cache_key": cache_key,
            },
        )

    async def report_memory_retrieved(self, *, count: int, query: str) -> None:
        await self._emit(
            "memory_retrieved",
            f"召回 {count} 条长期记忆",
            {"count": count, "query": query[:300]},
        )

    async def report_memory_updated(
        self,
        *,
        upserted: int,
        superseded: int,
        unchanged: int,
    ) -> None:
        await self._emit(
            "memory_updated",
            f"长期记忆更新 {upserted} 条",
            {
                "upserted": upserted,
                "superseded": superseded,
                "unchanged": unchanged,
            },
        )

    async def report_task_result(self, final_answer: str, files: list[str]) -> None:
        await self._emit(
            "task_result",
            "任务完成",
            {"final_answer": final_answer, "files": files},
        )

    async def report_task_cancelled(self) -> None:
        await self._emit("task_cancelled", "任务已取消", {})

    async def report_error(self, error_type: str, message: str) -> None:
        await self._emit("error", message, {"error_type": error_type})


monitor = Monitor()
