from __future__ import annotations

import asyncio
import hashlib
import json
import random
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Awaitable, Callable, cast

import httpx
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools import ToolException
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import ValidationError

from app.agent.settings import (
    tool_circuit_failure_threshold,
    tool_circuit_reset_sec,
    tool_idempotency_ttl_sec,
    tool_max_retries,
    tool_retry_initial_delay_sec,
    tool_retry_jitter,
    tool_retry_max_delay_sec,
    tool_timeout_compute_sec,
    tool_timeout_llm_sec,
    tool_timeout_search_sec,
    tool_timeout_sub_agent_sec,
    tool_timeout_web_sec,
)
from app.api.monitor import monitor

ToolResult = ToolMessage | Command[Any]
ToolHandler = Callable[[ToolCallRequest], Awaitable[ToolResult]]


class ToolErrorCategory(StrEnum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    BUSINESS = "business"
    INTERNAL = "internal"
    CIRCUIT_OPEN = "circuit_open"


_RETRYABLE_CATEGORIES = {
    ToolErrorCategory.RATE_LIMIT,
    ToolErrorCategory.TIMEOUT,
    ToolErrorCategory.NETWORK,
}


class ToolBusinessError(RuntimeError):
    """Non-retryable domain failure raised by a tool adapter."""


class ToolPermanentError(RuntimeError):
    """Non-retryable configuration or permission failure."""


class ToolExecutionFailed(RuntimeError):
    def __init__(
        self,
        *,
        error: Exception,
        category: ToolErrorCategory,
        attempts: int,
    ) -> None:
        super().__init__(str(error))
        self.error = error
        self.category = category
        self.attempts = attempts


@dataclass(frozen=True)
class CircuitDecision:
    allowed: bool
    half_open: bool = False
    retry_after_sec: float = 0.0


@dataclass
class CircuitState:
    consecutive_failures: int = 0
    open_until: float = 0.0
    half_open_in_flight: bool = False


@dataclass(frozen=True)
class ToolPolicy:
    timeout_sec: float
    cache_ttl_sec: float


@dataclass(frozen=True)
class ToolExecutionOutcome:
    result: ToolResult
    attempts: int


_TOOL_STATE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "planner": ("query", "allowed_platforms", "allowed_category"),
    "chat_fallback": (),
    "web_search": (),
    "category_insight": ("allowed_category",),
    "item_search": (
        "query",
        "plan",
        "long_term_preferences",
        "allowed_platforms",
        "allowed_category",
    ),
    "price_compare": ("search_outputs",),
    "shipping_calc": ("compared",),
    "item_picker": ("plan", "insight", "shipping", "long_term_preferences"),
    "shopping_summary": ("plan", "picker"),
    # dispatch_tool already owns a stronger in-flight/TTL deduplicator because it
    # creates child threads. It still receives timeout, retry, and circuit handling.
    "dispatch_tool": (),
}


class ToolExecutionRegistry:
    """Process-local circuit state and in-flight idempotency coordination."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._circuits: dict[str, CircuitState] = {}
        self._inflight: dict[
            tuple[str, str], asyncio.Task[ToolExecutionOutcome]
        ] = {}

    async def circuit_decision(self, tool_name: str) -> CircuitDecision:
        now = time.monotonic()
        async with self._lock:
            state = self._circuits.setdefault(tool_name, CircuitState())
            if state.open_until > now:
                return CircuitDecision(
                    allowed=False,
                    retry_after_sec=max(0.0, state.open_until - now),
                )
            if state.open_until > 0.0:
                if state.half_open_in_flight:
                    return CircuitDecision(allowed=False, retry_after_sec=0.1)
                state.half_open_in_flight = True
                return CircuitDecision(allowed=True, half_open=True)
            return CircuitDecision(allowed=True)

    async def record_success(self, tool_name: str) -> bool:
        """Reset the circuit and return whether it recovered from open/half-open."""

        async with self._lock:
            state = self._circuits.setdefault(tool_name, CircuitState())
            recovered = bool(state.open_until or state.half_open_in_flight)
            state.consecutive_failures = 0
            state.open_until = 0.0
            state.half_open_in_flight = False
            return recovered

    async def record_transient_failure(
        self,
        tool_name: str,
        *,
        threshold: int,
        reset_sec: float,
    ) -> tuple[bool, int]:
        """Record one exhausted transient call and return (opened, failures)."""

        async with self._lock:
            state = self._circuits.setdefault(tool_name, CircuitState())
            state.consecutive_failures += 1
            state.half_open_in_flight = False
            opened = state.consecutive_failures >= threshold
            if opened:
                state.open_until = time.monotonic() + reset_sec
            return opened, state.consecutive_failures

    async def release_half_open(self, tool_name: str) -> None:
        async with self._lock:
            state = self._circuits.setdefault(tool_name, CircuitState())
            state.half_open_in_flight = False

    async def get_or_create(
        self,
        *,
        thread_id: str,
        idempotency_key: str,
        operation: Callable[[], Awaitable[ToolExecutionOutcome]],
    ) -> tuple[asyncio.Task[ToolExecutionOutcome], str]:
        key = (thread_id, idempotency_key)
        async with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                return existing, "inflight"
            task = asyncio.create_task(
                operation(),
                name=f"tool-{thread_id}-{idempotency_key[:10]}",
            )
            self._inflight[key] = task
            task.add_done_callback(
                lambda completed: asyncio.create_task(
                    self._remove_inflight(key, completed)
                )
            )
            return task, "executed"

    async def _remove_inflight(
        self,
        key: tuple[str, str],
        completed: asyncio.Task[ToolExecutionOutcome],
    ) -> None:
        async with self._lock:
            if self._inflight.get(key) is completed:
                self._inflight.pop(key, None)

    async def reset(self) -> None:
        async with self._lock:
            tasks = list(self._inflight.values())
            self._inflight.clear()
            self._circuits.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


_registry = ToolExecutionRegistry()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_timestamp(value: datetime) -> str:
    return value.isoformat()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def tool_idempotency_key(request: ToolCallRequest) -> str:
    tool_name = str(request.tool_call.get("name") or "unknown_tool")
    state = request.state if isinstance(request.state, dict) else {}
    dependencies = {
        key: state.get(key)
        for key in _TOOL_STATE_DEPENDENCIES.get(tool_name, ())
    }
    material = {
        "tool": tool_name,
        "args": request.tool_call.get("args") or {},
        "state": dependencies,
    }
    digest = hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    return f"{tool_name}:{digest}"


def _tool_policy(tool_name: str) -> ToolPolicy:
    default_ttl = tool_idempotency_ttl_sec()
    if tool_name in {"planner", "chat_fallback", "shopping_summary"}:
        return ToolPolicy(tool_timeout_llm_sec(), default_ttl)
    if tool_name == "web_search":
        return ToolPolicy(tool_timeout_web_sec(), min(default_ttl, 300.0))
    if tool_name in {"category_insight", "item_search"}:
        return ToolPolicy(tool_timeout_search_sec(), default_ttl)
    if tool_name == "dispatch_tool":
        return ToolPolicy(tool_timeout_sub_agent_sec(), 0.0)
    return ToolPolicy(tool_timeout_compute_sec(), default_ttl)


def classify_tool_error(exc: Exception) -> ToolErrorCategory:
    if isinstance(
        exc,
        (
            ToolBusinessError,
            ToolPermanentError,
            ToolException,
            ValidationError,
            ValueError,
            TypeError,
            KeyError,
            PermissionError,
        ),
    ):
        return ToolErrorCategory.BUSINESS
    if isinstance(
        exc,
        (
            asyncio.TimeoutError,
            TimeoutError,
            httpx.TimeoutException,
        ),
    ):
        return ToolErrorCategory.TIMEOUT
    if isinstance(
        exc,
        (
            httpx.NetworkError,
            ConnectionError,
            ConnectionResetError,
            ConnectionAbortedError,
            socket.gaierror,
        ),
    ):
        return ToolErrorCategory.NETWORK

    try:
        import openai
    except ImportError:  # pragma: no cover - project dependency includes openai
        openai = None  # type: ignore[assignment]
    if openai is not None:
        if isinstance(exc, openai.RateLimitError):
            return ToolErrorCategory.RATE_LIMIT
        if isinstance(exc, openai.APITimeoutError):
            return ToolErrorCategory.TIMEOUT
        if isinstance(exc, openai.APIConnectionError):
            return ToolErrorCategory.NETWORK

    response = getattr(exc, "response", None)
    status_code = getattr(exc, "status_code", None) or getattr(
        response, "status_code", None
    )
    if status_code == 429:
        return ToolErrorCategory.RATE_LIMIT
    if status_code in {408, 504}:
        return ToolErrorCategory.TIMEOUT
    if isinstance(status_code, int) and status_code >= 500:
        return ToolErrorCategory.NETWORK
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return ToolErrorCategory.BUSINESS

    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "rate" in name and "limit" in name or "too many requests" in message:
        return ToolErrorCategory.RATE_LIMIT
    if "timeout" in name or "timed out" in message:
        return ToolErrorCategory.TIMEOUT
    if any(token in name for token in ("connection", "network", "transport")):
        return ToolErrorCategory.NETWORK
    return ToolErrorCategory.INTERNAL


def _retry_delay(attempt: int, *, initial: float, maximum: float, jitter: bool) -> float:
    delay = min(maximum, initial * (2 ** max(0, attempt - 1)))
    if jitter and delay > 0:
        delay *= random.uniform(0.75, 1.25)
    return max(0.0, delay)


def _tool_message_from_result(result: ToolResult) -> ToolMessage | None:
    if isinstance(result, ToolMessage):
        return result
    update = result.update
    if not isinstance(update, dict):
        return None
    for message in reversed(list(update.get("messages") or [])):
        if isinstance(message, ToolMessage):
            return message
    return None


def _payload_from_result(result: ToolResult) -> dict[str, Any] | None:
    message = _tool_message_from_result(result)
    if message is None or message.status == "error":
        return None
    updates: dict[str, Any] = {}
    if isinstance(result, Command) and isinstance(result.update, dict):
        updates = {
            str(key): value
            for key, value in result.update.items()
            if key not in {"messages", "tool_idempotency", "tool_reliability"}
        }
    return {
        "content": message.content,
        "name": message.name,
        "artifact": message.artifact,
        "status": message.status,
        "updates": updates,
    }


def _command_from_payload(
    payload: dict[str, Any],
    *,
    tool_name: str,
    tool_call_id: str,
    extra_updates: dict[str, Any] | None = None,
) -> Command[Any]:
    message = ToolMessage(
        content=payload.get("content", ""),
        tool_call_id=tool_call_id,
        name=str(payload.get("name") or tool_name),
        artifact=payload.get("artifact"),
        status=cast(Any, payload.get("status") or "success"),
    )
    return Command(
        update={
            **dict(payload.get("updates") or {}),
            **(extra_updates or {}),
            "messages": [message],
        }
    )


def _clone_result(
    result: ToolResult,
    *,
    tool_name: str,
    tool_call_id: str,
    extra_updates: dict[str, Any],
) -> Command[Any]:
    if isinstance(result, ToolMessage):
        message = result.model_copy(
            update={"tool_call_id": tool_call_id, "name": result.name or tool_name}
        )
        return Command(update={**extra_updates, "messages": [message]})

    update = dict(result.update) if isinstance(result.update, dict) else {}
    rebound_messages: list[Any] = []
    for message in list(update.get("messages") or []):
        if isinstance(message, ToolMessage):
            rebound_messages.append(
                message.model_copy(
                    update={
                        "tool_call_id": tool_call_id,
                        "name": message.name or tool_name,
                    }
                )
            )
        else:
            rebound_messages.append(message)
    update.update(extra_updates)
    if rebound_messages:
        update["messages"] = rebound_messages
    return Command(
        graph=result.graph,
        update=update,
        resume=result.resume,
        goto=result.goto,
    )


def _cache_entry_valid(entry: Any, now: datetime) -> bool:
    if not isinstance(entry, dict):
        return False
    expires_at = entry.get("expires_at")
    if not expires_at:
        return True
    try:
        parsed = datetime.fromisoformat(str(expires_at))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed > now


def _reliability_update(
    *,
    tool_name: str,
    status: str,
    category: ToolErrorCategory | None,
    attempts: int,
    timeout_sec: float,
    duration_ms: int,
    idempotency_key: str,
    source: str,
) -> dict[str, Any]:
    return {
        "tool_reliability": {
            tool_name: {
                "status": status,
                "category": category.value if category else None,
                "attempts": attempts,
                "timeout_sec": timeout_sec,
                "duration_ms": duration_ms,
                "idempotency_key": idempotency_key,
                "source": source,
                "updated_at": _iso_timestamp(_utc_now()),
            }
        }
    }


def _failure_command(
    *,
    tool_name: str,
    tool_call_id: str,
    category: ToolErrorCategory,
    attempts: int,
    timeout_sec: float,
    idempotency_key: str,
    message: str,
    retry_after_sec: float = 0.0,
) -> Command[Any]:
    payload = {
        "tool_error": {
            "tool": tool_name,
            "category": category.value,
            "retryable": category in _RETRYABLE_CATEGORIES,
            "attempts": attempts,
            "message": message[:1000],
            "retry_after_sec": round(retry_after_sec, 3),
        }
    }
    tool_message = ToolMessage(
        content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        tool_call_id=tool_call_id,
        name=tool_name,
        status="error",
    )
    return Command(
        update={
            "messages": [tool_message],
            **_reliability_update(
                tool_name=tool_name,
                status="failed",
                category=category,
                attempts=attempts,
                timeout_sec=timeout_sec,
                duration_ms=0,
                idempotency_key=idempotency_key,
                source="error",
            ),
        }
    )


class ToolReliabilityMiddleware(AgentMiddleware):
    """Unified timeout, retry, circuit breaker, and idempotency middleware."""

    def __init__(
        self,
        *,
        registry: ToolExecutionRegistry | None = None,
        max_retries: int | None = None,
        initial_delay_sec: float | None = None,
        max_delay_sec: float | None = None,
        jitter: bool | None = None,
        circuit_failure_threshold: int | None = None,
        circuit_reset_sec: float | None = None,
    ) -> None:
        self.registry = registry or _registry
        self.max_retries = tool_max_retries() if max_retries is None else max(0, max_retries)
        self.initial_delay_sec = (
            tool_retry_initial_delay_sec()
            if initial_delay_sec is None
            else max(0.0, initial_delay_sec)
        )
        self.max_delay_sec = (
            tool_retry_max_delay_sec()
            if max_delay_sec is None
            else max(0.0, max_delay_sec)
        )
        self.jitter = tool_retry_jitter() if jitter is None else jitter
        self.circuit_failure_threshold = (
            tool_circuit_failure_threshold()
            if circuit_failure_threshold is None
            else max(1, circuit_failure_threshold)
        )
        self.circuit_reset_sec = (
            tool_circuit_reset_sec()
            if circuit_reset_sec is None
            else max(0.01, circuit_reset_sec)
        )

    async def _execute_with_retry(
        self,
        *,
        request: ToolCallRequest,
        handler: ToolHandler,
        tool_name: str,
        timeout_sec: float,
        idempotency_key: str,
    ) -> ToolExecutionOutcome:
        total_attempts = self.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            await monitor.report_tool_attempt(
                tool_name=tool_name,
                attempt=attempt,
                max_attempts=total_attempts,
                timeout_sec=timeout_sec,
                idempotency_key=idempotency_key,
            )
            try:
                result = await asyncio.wait_for(handler(request), timeout=timeout_sec)
                return ToolExecutionOutcome(result=result, attempts=attempt)
            except Exception as exc:
                category = classify_tool_error(exc)
                retryable = category in _RETRYABLE_CATEGORIES
                if not retryable or attempt >= total_attempts:
                    raise ToolExecutionFailed(
                        error=exc,
                        category=category,
                        attempts=attempt,
                    ) from exc
                delay = _retry_delay(
                    attempt,
                    initial=self.initial_delay_sec,
                    maximum=self.max_delay_sec,
                    jitter=self.jitter,
                )
                await monitor.report_tool_retry(
                    tool_name=tool_name,
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    category=category.value,
                    delay_sec=delay,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if delay > 0:
                    await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: ToolHandler,
    ) -> ToolResult:
        tool_name = str(request.tool_call.get("name") or "unknown_tool")
        tool_call_id = str(request.tool_call.get("id") or "missing-tool-call-id")
        policy = _tool_policy(tool_name)
        idempotency_key = tool_idempotency_key(request)
        state = request.state if isinstance(request.state, dict) else {}
        now = _utc_now()

        cache = state.get("tool_idempotency") or {}
        cached = cache.get(idempotency_key) if isinstance(cache, dict) else None
        if _cache_entry_valid(cached, now):
            payload = cached.get("payload") if isinstance(cached, dict) else None
            if isinstance(payload, dict):
                await monitor.report_tool_idempotency_hit(
                    tool_name=tool_name,
                    idempotency_key=idempotency_key,
                    source="checkpoint",
                )
                return _command_from_payload(
                    payload,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    extra_updates=_reliability_update(
                        tool_name=tool_name,
                        status="success",
                        category=None,
                        attempts=0,
                        timeout_sec=policy.timeout_sec,
                        duration_ms=0,
                        idempotency_key=idempotency_key,
                        source="checkpoint_cache",
                    ),
                )

        decision = await self.registry.circuit_decision(tool_name)
        if not decision.allowed:
            await monitor.report_tool_circuit_rejected(
                tool_name=tool_name,
                retry_after_sec=decision.retry_after_sec,
            )
            return _failure_command(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                category=ToolErrorCategory.CIRCUIT_OPEN,
                attempts=0,
                timeout_sec=policy.timeout_sec,
                idempotency_key=idempotency_key,
                message="工具熔断器处于打开状态，请使用已有结果或稍后重试。",
                retry_after_sec=decision.retry_after_sec,
            )

        thread_id = str(state.get("thread_id") or "unknown-thread")
        started = time.perf_counter()
        task, source = await self.registry.get_or_create(
            thread_id=thread_id,
            idempotency_key=idempotency_key,
            operation=lambda: self._execute_with_retry(
                request=request,
                handler=handler,
                tool_name=tool_name,
                timeout_sec=policy.timeout_sec,
                idempotency_key=idempotency_key,
            ),
        )
        if source == "inflight":
            await monitor.report_tool_idempotency_hit(
                tool_name=tool_name,
                idempotency_key=idempotency_key,
                source="inflight",
            )

        try:
            outcome = await asyncio.shield(task)
            raw_result = outcome.result
        except ToolExecutionFailed as failure:
            duration_ms = int((time.perf_counter() - started) * 1000)
            retryable = failure.category in _RETRYABLE_CATEGORIES
            opened = False
            failures = 0
            if source == "executed" and retryable:
                opened, failures = await self.registry.record_transient_failure(
                    tool_name,
                    threshold=self.circuit_failure_threshold,
                    reset_sec=self.circuit_reset_sec,
                )
                if opened:
                    await monitor.report_tool_circuit_open(
                        tool_name=tool_name,
                        failures=failures,
                        reset_sec=self.circuit_reset_sec,
                        category=failure.category.value,
                    )
            elif source == "executed" and decision.half_open:
                # A non-transient business/internal response proves that the
                # dependency is reachable, so close the transport circuit.
                recovered = await self.registry.record_success(tool_name)
                if recovered:
                    await monitor.report_tool_circuit_recovered(tool_name=tool_name)

            await monitor.report_tool_failure(
                tool_name=tool_name,
                category=failure.category.value,
                attempts=failure.attempts,
                duration_ms=duration_ms,
                retryable=retryable,
                error=f"{type(failure.error).__name__}: {failure.error}",
            )
            return _failure_command(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                category=failure.category,
                attempts=failure.attempts,
                timeout_sec=policy.timeout_sec,
                idempotency_key=idempotency_key,
                message=f"{type(failure.error).__name__}: {failure.error}",
            )
        except Exception as exc:
            # Defensive boundary for failures raised outside the retry operation.
            category = classify_tool_error(exc)
            await monitor.report_tool_failure(
                tool_name=tool_name,
                category=category.value,
                attempts=1,
                duration_ms=int((time.perf_counter() - started) * 1000),
                retryable=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            return _failure_command(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                category=category,
                attempts=1,
                timeout_sec=policy.timeout_sec,
                idempotency_key=idempotency_key,
                message=f"{type(exc).__name__}: {exc}",
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        if source == "executed":
            recovered = await self.registry.record_success(tool_name)
            if recovered:
                await monitor.report_tool_circuit_recovered(tool_name=tool_name)
        await monitor.report_tool_success(
            tool_name=tool_name,
            attempts=outcome.attempts,
            duration_ms=duration_ms,
            source=source,
        )

        extra_updates = _reliability_update(
            tool_name=tool_name,
            status="success",
            category=None,
            attempts=outcome.attempts,
            timeout_sec=policy.timeout_sec,
            duration_ms=duration_ms,
            idempotency_key=idempotency_key,
            source=source,
        )
        payload = _payload_from_result(raw_result)
        if payload is not None and policy.cache_ttl_sec > 0:
            created_at = _utc_now()
            expires_at = datetime.fromtimestamp(
                created_at.timestamp() + policy.cache_ttl_sec,
                tz=timezone.utc,
            )
            extra_updates["tool_idempotency"] = {
                idempotency_key: {
                    "tool_name": tool_name,
                    "created_at": _iso_timestamp(created_at),
                    "expires_at": _iso_timestamp(expires_at),
                    "payload": payload,
                }
            }
        return _clone_result(
            raw_result,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            extra_updates=extra_updates,
        )


async def reset_tool_reliability() -> None:
    """Reset process-local circuits and in-flight calls (tests/service shutdown)."""

    await _registry.reset()
