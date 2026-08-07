from __future__ import annotations

import asyncio
from collections import defaultdict

import pytest

from app.agent.fork_scheduler import (
    ForkBudgetExceeded,
    ForkQueueFull,
    ForkQueueTimeout,
    ForkScheduler,
    ForkSchedulerConfig,
)


def _config(
    *,
    global_concurrency: int = 2,
    per_task_concurrency: int = 2,
    per_task_budget: int = 8,
    max_queue_size: int = 16,
    queue_timeout_sec: float = 1.0,
    dedup_ttl_sec: float = 30.0,
) -> ForkSchedulerConfig:
    return ForkSchedulerConfig(
        global_concurrency=global_concurrency,
        per_task_concurrency=per_task_concurrency,
        per_task_budget=per_task_budget,
        max_queue_size=max_queue_size,
        queue_timeout_sec=queue_timeout_sec,
        dedup_ttl_sec=dedup_ttl_sec,
    )


@pytest.mark.asyncio
async def test_equivalent_forks_share_one_inflight_execution() -> None:
    scheduler = ForkScheduler(_config())
    calls = 0

    async def operation() -> dict[str, str]:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return {"result": "amazon-cup"}

    first, second = await asyncio.gather(
        scheduler.run(
            root_thread_id="root-dedup",
            demands="搜索 咖啡杯",
            platform="amazon",
            category="咖啡杯",
            operation=operation,
        ),
        scheduler.run(
            root_thread_id="root-dedup",
            demands="  搜索   咖啡杯  ",
            platform="amazon",
            category="咖啡杯",
            operation=operation,
        ),
    )

    assert calls == 1
    assert first == second == {"result": "amazon-cup"}

    cached = await scheduler.run(
        root_thread_id="root-dedup",
        demands="搜索 咖啡杯",
        platform="amazon",
        category="咖啡杯",
        operation=operation,
    )
    assert calls == 1
    assert cached == first
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_global_and_per_root_concurrency_limits() -> None:
    scheduler = ForkScheduler(
        _config(global_concurrency=2, per_task_concurrency=1)
    )
    lock = asyncio.Lock()
    active_global = 0
    max_global = 0
    active_by_root: dict[str, int] = defaultdict(int)
    max_by_root: dict[str, int] = defaultdict(int)

    def operation_for(root: str):
        async def operation() -> str:
            nonlocal active_global, max_global
            async with lock:
                active_global += 1
                active_by_root[root] += 1
                max_global = max(max_global, active_global)
                max_by_root[root] = max(max_by_root[root], active_by_root[root])
            await asyncio.sleep(0.05)
            async with lock:
                active_global -= 1
                active_by_root[root] -= 1
            return root

        return operation

    await asyncio.gather(
        scheduler.run(
            root_thread_id="root-a",
            demands="a-1",
            platform="amazon",
            category="咖啡杯",
            operation=operation_for("root-a"),
        ),
        scheduler.run(
            root_thread_id="root-a",
            demands="a-2",
            platform="shopee",
            category="咖啡杯",
            operation=operation_for("root-a"),
        ),
        scheduler.run(
            root_thread_id="root-b",
            demands="b-1",
            platform="amazon",
            category="旅行收纳",
            operation=operation_for("root-b"),
        ),
        scheduler.run(
            root_thread_id="root-b",
            demands="b-2",
            platform="ebay",
            category="旅行收纳",
            operation=operation_for("root-b"),
        ),
    )

    assert max_global <= 2
    assert max_by_root == {"root-a": 1, "root-b": 1}
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_unique_fork_budget_and_reset() -> None:
    scheduler = ForkScheduler(_config(per_task_budget=2))

    async def operation() -> str:
        return "ok"

    for index in range(2):
        assert (
            await scheduler.run(
                root_thread_id="root-budget",
                demands=f"task-{index}",
                platform="amazon",
                category="咖啡杯",
                operation=operation,
            )
            == "ok"
        )

    with pytest.raises(ForkBudgetExceeded):
        await scheduler.run(
            root_thread_id="root-budget",
            demands="task-3",
            platform="amazon",
            category="咖啡杯",
            operation=operation,
        )

    await scheduler.reset_root("root-budget")
    assert (
        await scheduler.run(
            root_thread_id="root-budget",
            demands="task-after-reset",
            platform="amazon",
            category="咖啡杯",
            operation=operation,
        )
        == "ok"
    )
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_fork_queue_timeout_releases_permits() -> None:
    scheduler = ForkScheduler(
        _config(
            global_concurrency=1,
            per_task_concurrency=1,
            queue_timeout_sec=0.03,
        )
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_operation() -> str:
        started.set()
        await release.wait()
        return "slow"

    first = asyncio.create_task(
        scheduler.run(
            root_thread_id="root-timeout",
            demands="slow",
            platform="amazon",
            category="咖啡杯",
            operation=slow_operation,
        )
    )
    await started.wait()

    with pytest.raises(ForkQueueTimeout):
        await scheduler.run(
            root_thread_id="root-timeout",
            demands="queued",
            platform="shopee",
            category="咖啡杯",
            operation=lambda: asyncio.sleep(0, result="queued"),
        )

    release.set()
    assert await first == "slow"

    # The failed in-flight key must be removable and retryable.
    assert (
        await scheduler.run(
            root_thread_id="root-timeout",
            demands="queued",
            platform="shopee",
            category="咖啡杯",
            operation=lambda: asyncio.sleep(0, result="retried"),
        )
        == "retried"
    )

    # A timed-out waiter must not leak either semaphore permit.
    assert (
        await scheduler.run(
            root_thread_id="root-timeout",
            demands="after-timeout",
            platform="ebay",
            category="咖啡杯",
            operation=lambda: asyncio.sleep(0, result="after"),
        )
        == "after"
    )
    await scheduler.shutdown()


@pytest.mark.asyncio
async def test_process_queue_has_a_hard_capacity() -> None:
    scheduler = ForkScheduler(
        _config(
            global_concurrency=1,
            per_task_concurrency=3,
            max_queue_size=1,
            queue_timeout_sec=1.0,
        )
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_operation() -> str:
        started.set()
        await release.wait()
        return "slow"

    first = asyncio.create_task(
        scheduler.run(
            root_thread_id="root-queue-full",
            demands="running",
            platform="amazon",
            category="咖啡杯",
            operation=slow_operation,
        )
    )
    await started.wait()
    second = asyncio.create_task(
        scheduler.run(
            root_thread_id="root-queue-full",
            demands="waiting",
            platform="shopee",
            category="咖啡杯",
            operation=lambda: asyncio.sleep(0, result="waiting"),
        )
    )
    await asyncio.sleep(0.01)

    with pytest.raises(ForkQueueFull):
        await scheduler.run(
            root_thread_id="root-queue-full",
            demands="rejected",
            platform="ebay",
            category="咖啡杯",
            operation=lambda: asyncio.sleep(0, result="rejected"),
        )

    release.set()
    assert await first == "slow"
    assert await second == "waiting"
    await scheduler.shutdown()
