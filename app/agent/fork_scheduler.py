from __future__ import annotations

import asyncio
import hashlib
import re
import time
import unicodedata
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from app.agent.settings import (
    fork_dedup_ttl_sec,
    fork_queue_timeout_sec,
    max_concurrent_forks,
    max_concurrent_forks_per_task,
    max_fork_queue_size,
    max_forks_per_task,
)
from app.api.monitor import monitor
from app.models import Platform

T = TypeVar("T")


class ForkBudgetExceeded(RuntimeError):
    """Raised when one root task has exhausted its unique fork-attempt budget."""


class ForkQueueTimeout(RuntimeError):
    """Raised when a fork cannot obtain concurrency permits before its deadline."""


class ForkQueueFull(RuntimeError):
    """Raised when the process-wide fork queue has reached its configured cap."""


@dataclass(frozen=True)
class ForkRequestKey:
    root_thread_id: str
    fingerprint: str


@dataclass(frozen=True)
class ForkSchedulerConfig:
    global_concurrency: int
    per_task_concurrency: int
    per_task_budget: int
    max_queue_size: int
    queue_timeout_sec: float
    dedup_ttl_sec: float


@dataclass
class _CachedResult:
    expires_at: float
    value: Any


def current_fork_scheduler_config() -> ForkSchedulerConfig:
    return ForkSchedulerConfig(
        global_concurrency=max_concurrent_forks(),
        per_task_concurrency=max_concurrent_forks_per_task(),
        per_task_budget=max_forks_per_task(),
        max_queue_size=max_fork_queue_size(),
        queue_timeout_sec=fork_queue_timeout_sec(),
        dedup_ttl_sec=fork_dedup_ttl_sec(),
    )


def _normalize_text(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def build_fork_fingerprint(
    *,
    demands: str,
    platform: Platform | None,
    category: str | None,
) -> str:
    """Create a stable key for exact-equivalent platform/category sub tasks."""

    payload = "\x1f".join(
        (
            _normalize_text(demands),
            _normalize_text(platform),
            _normalize_text(category),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class ForkScheduler:
    """Process-local scheduler for homogeneous sub-Agent execution.

    It enforces a global semaphore, a per-root-task semaphore, a total unique
    fork budget, FIFO semaphore queuing, and exact-equivalent request coalescing.
    Successful results remain reusable for a short TTL within the same root task.
    """

    def __init__(self, config: ForkSchedulerConfig) -> None:
        self.config = config
        self._global_semaphore = asyncio.Semaphore(config.global_concurrency)
        self._root_semaphores: dict[str, asyncio.Semaphore] = {}
        self._started_per_root: dict[str, int] = defaultdict(int)
        self._inflight: dict[ForkRequestKey, asyncio.Task[Any]] = {}
        self._completed: dict[ForkRequestKey, _CachedResult] = {}
        self._waiting_global = 0
        self._waiting_per_root: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    def _purge_expired_locked(self, now: float) -> None:
        expired = [
            key for key, cached in self._completed.items() if cached.expires_at <= now
        ]
        for key in expired:
            self._completed.pop(key, None)

    async def run(
        self,
        *,
        root_thread_id: str,
        demands: str,
        platform: Platform | None,
        category: str | None,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        fingerprint = build_fork_fingerprint(
            demands=demands,
            platform=platform,
            category=category,
        )
        key = ForkRequestKey(root_thread_id, fingerprint)
        reused_from = ""

        async with self._lock:
            now = time.monotonic()
            self._purge_expired_locked(now)
            cached = self._completed.get(key)
            if cached is not None:
                result = cached.value
                reused_from = "completed"
                task = None
            else:
                task = self._inflight.get(key)
                if task is not None:
                    result = None
                    reused_from = "inflight"
                else:
                    started = self._started_per_root[root_thread_id]
                    if started >= self.config.per_task_budget:
                        raise ForkBudgetExceeded(
                            "主任务 fork 预算已用尽："
                            f"{started}/{self.config.per_task_budget}"
                        )
                    process_capacity = (
                        self.config.global_concurrency + self.config.max_queue_size
                    )
                    if len(self._inflight) >= process_capacity:
                        raise ForkQueueFull(
                            "进程级 fork 队列已满："
                            f"{len(self._inflight)}/{process_capacity}"
                        )
                    self._started_per_root[root_thread_id] = started + 1
                    root_semaphore = self._root_semaphores.setdefault(
                        root_thread_id,
                        asyncio.Semaphore(self.config.per_task_concurrency),
                    )
                    task = asyncio.create_task(
                        self._execute(
                            key=key,
                            root_thread_id=root_thread_id,
                            platform=platform,
                            category=category,
                            root_semaphore=root_semaphore,
                            operation=operation,
                        ),
                        name=f"shoppilot-fork-{root_thread_id}-{fingerprint[:8]}",
                    )
                    self._inflight[key] = task
                    result = None

        if reused_from:
            await monitor.report_fork_deduplicated(
                fingerprint=fingerprint,
                platform=platform,
                category=category,
                source=reused_from,
            )
        if task is None:
            return result
        # A cancelled duplicate caller must not cancel the shared child execution.
        return await asyncio.shield(task)

    async def _execute(
        self,
        *,
        key: ForkRequestKey,
        root_thread_id: str,
        platform: Platform | None,
        category: str | None,
        root_semaphore: asyncio.Semaphore,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        root_acquired = False
        global_acquired = False
        queued = root_semaphore.locked() or self._global_semaphore.locked()
        queue_started = time.perf_counter()

        if queued:
            async with self._lock:
                self._waiting_global += 1
                self._waiting_per_root[root_thread_id] += 1
                global_position = self._waiting_global
                task_position = self._waiting_per_root[root_thread_id]
            await monitor.report_fork_queued(
                fingerprint=key.fingerprint,
                platform=platform,
                category=category,
                global_position=global_position,
                task_position=task_position,
            )

        try:
            deadline = asyncio.get_running_loop().time() + self.config.queue_timeout_sec
            await asyncio.wait_for(
                root_semaphore.acquire(),
                timeout=max(0.001, deadline - asyncio.get_running_loop().time()),
            )
            root_acquired = True
            await asyncio.wait_for(
                self._global_semaphore.acquire(),
                timeout=max(0.001, deadline - asyncio.get_running_loop().time()),
            )
            global_acquired = True
        except BaseException as exc:
            if global_acquired:
                self._global_semaphore.release()
                global_acquired = False
            if root_acquired:
                root_semaphore.release()
                root_acquired = False
            async with self._lock:
                current = asyncio.current_task()
                if self._inflight.get(key) is current:
                    self._inflight.pop(key, None)
            if isinstance(exc, asyncio.TimeoutError):
                raise ForkQueueTimeout(
                    "子 Agent 排队超过上限 "
                    f"{self.config.queue_timeout_sec:g}s"
                ) from exc
            raise
        finally:
            if queued:
                async with self._lock:
                    self._waiting_global = max(0, self._waiting_global - 1)
                    remaining = max(0, self._waiting_per_root[root_thread_id] - 1)
                    if remaining:
                        self._waiting_per_root[root_thread_id] = remaining
                    else:
                        self._waiting_per_root.pop(root_thread_id, None)

        if queued:
            await monitor.report_fork_dequeued(
                fingerprint=key.fingerprint,
                platform=platform,
                category=category,
                wait_ms=int((time.perf_counter() - queue_started) * 1000),
            )

        succeeded = False
        value: T
        try:
            value = await operation()
            succeeded = True
            return value
        finally:
            if global_acquired:
                self._global_semaphore.release()
            if root_acquired:
                root_semaphore.release()
            async with self._lock:
                current = asyncio.current_task()
                if self._inflight.get(key) is current:
                    self._inflight.pop(key, None)
                if succeeded and self.config.dedup_ttl_sec > 0:
                    self._completed[key] = _CachedResult(
                        expires_at=time.monotonic() + self.config.dedup_ttl_sec,
                        value=value,
                    )

    async def reset_root(self, root_thread_id: str) -> None:
        """Release counters/cache and cancel orphaned children for one main run."""

        async with self._lock:
            tasks = [
                task
                for key, task in self._inflight.items()
                if key.root_thread_id == root_thread_id
            ]
            for key in [
                key for key in self._inflight if key.root_thread_id == root_thread_id
            ]:
                self._inflight.pop(key, None)
            for key in [
                key for key in self._completed if key.root_thread_id == root_thread_id
            ]:
                self._completed.pop(key, None)
            self._started_per_root.pop(root_thread_id, None)
            self._root_semaphores.pop(root_thread_id, None)
            self._waiting_per_root.pop(root_thread_id, None)

        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def shutdown(self) -> None:
        async with self._lock:
            tasks = list(set(self._inflight.values()))
            self._inflight.clear()
            self._completed.clear()
            self._started_per_root.clear()
            self._root_semaphores.clear()
            self._waiting_per_root.clear()
            self._waiting_global = 0
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


_scheduler: ForkScheduler | None = None
_scheduler_loop: asyncio.AbstractEventLoop | None = None
_scheduler_config: ForkSchedulerConfig | None = None


def get_fork_scheduler() -> ForkScheduler:
    global _scheduler, _scheduler_loop, _scheduler_config

    loop = asyncio.get_running_loop()
    config = current_fork_scheduler_config()
    if (
        _scheduler is None
        or _scheduler_loop is not loop
        or _scheduler_config != config
    ):
        _scheduler = ForkScheduler(config)
        _scheduler_loop = loop
        _scheduler_config = config
    return _scheduler


async def reset_fork_scope(root_thread_id: str) -> None:
    scheduler = _scheduler
    if scheduler is not None:
        await scheduler.reset_root(root_thread_id)


async def close_fork_scheduler() -> None:
    global _scheduler, _scheduler_loop, _scheduler_config

    scheduler = _scheduler
    _scheduler = None
    _scheduler_loop = None
    _scheduler_config = None
    if scheduler is not None:
        await scheduler.shutdown()
