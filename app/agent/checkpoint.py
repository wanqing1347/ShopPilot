from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agent.settings import (
    checkpoint_auto_setup,
    checkpoint_backend,
    checkpoint_cleanup_batch_size,
    checkpoint_cleanup_scan_limit,
    checkpoint_db_file,
    checkpoint_postgres_dsn,
    checkpoint_retention_days,
)
from app.utils.runtime import PROJECT_ROOT

CheckpointBackend = Literal["memory", "sqlite", "postgres"]

_lock: asyncio.Lock | None = None
_lock_loop: asyncio.AbstractEventLoop | None = None
_checkpointer: BaseCheckpointSaver[Any] | None = None
_sqlite_connection: Any | None = None
_backend_context: Any | None = None
_active_backend: CheckpointBackend | None = None
_active_location: str | None = None


@dataclass(frozen=True)
class CheckpointCleanupReport:
    backend: str
    cutoff: str | None
    retention_days: int
    scanned_checkpoints: int
    unique_threads: int
    stale_candidates: int
    deleted_threads: list[str]
    skipped_active_threads: list[str]
    disabled: bool = False

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def _current_lock() -> asyncio.Lock:
    global _lock, _lock_loop

    loop = asyncio.get_running_loop()
    if _lock is None or _lock_loop is not loop:
        _lock = asyncio.Lock()
        _lock_loop = loop
    return _lock


def resolve_checkpoint_path() -> Path:
    configured = Path(checkpoint_db_file()).expanduser()
    path = configured if configured.is_absolute() else PROJECT_ROOT / configured
    return path.resolve()


def _masked_postgres_location(dsn: str) -> str:
    try:
        parsed = urlsplit(dsn)
        host = parsed.hostname or "configured-host"
        port = f":{parsed.port}" if parsed.port else ""
        database = parsed.path or "/configured-db"
        return f"{parsed.scheme or 'postgresql'}://{host}{port}{database}"
    except ValueError:
        return "postgresql://configured"


def _backend_location(backend: CheckpointBackend) -> str:
    if backend == "memory":
        return "process-memory"
    if backend == "sqlite":
        return str(resolve_checkpoint_path())
    dsn = checkpoint_postgres_dsn()
    if not dsn:
        raise ValueError(
            "SHOPPILOT_CHECKPOINT_BACKEND=postgres 时必须配置 "
            "SHOPPILOT_CHECKPOINT_POSTGRES_DSN"
        )
    return _masked_postgres_location(dsn)


def checkpoint_description() -> dict[str, Any]:
    backend = cast(CheckpointBackend, checkpoint_backend())
    return {
        "backend": backend,
        "location": _backend_location(backend),
        "retention_days": checkpoint_retention_days(),
    }


async def get_checkpointer() -> BaseCheckpointSaver[Any]:
    """Return the process-wide async LangGraph checkpointer.

    SQLite is the local default. PostgreSQL is supported through the official
    AsyncPostgresSaver for production deployments. Tests can select memory.
    """

    global _checkpointer, _sqlite_connection, _backend_context
    global _active_backend, _active_location

    backend = cast(CheckpointBackend, checkpoint_backend())
    location = _backend_location(backend)
    if (
        _checkpointer is not None
        and _active_backend == backend
        and _active_location == location
    ):
        return _checkpointer

    async with _current_lock():
        if (
            _checkpointer is not None
            and _active_backend == backend
            and _active_location == location
        ):
            return _checkpointer

        if _checkpointer is not None:
            await _close_unlocked()

        if backend == "memory":
            from langgraph.checkpoint.memory import InMemorySaver

            _checkpointer = InMemorySaver()
        elif backend == "sqlite":
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            db_path = resolve_checkpoint_path()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = await aiosqlite.connect(str(db_path))
            await connection.execute("PRAGMA journal_mode=WAL")
            await connection.execute("PRAGMA synchronous=NORMAL")
            await connection.execute("PRAGMA busy_timeout=5000")
            await connection.commit()

            saver = AsyncSqliteSaver(connection)
            if checkpoint_auto_setup():
                await saver.setup()
            _sqlite_connection = connection
            _checkpointer = saver
        else:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            except ImportError as exc:
                raise RuntimeError(
                    "PostgreSQL checkpoint 后端未安装；请执行 "
                    "`pip install -e \".[postgres]\"`"
                ) from exc

            context = AsyncPostgresSaver.from_conn_string(
                checkpoint_postgres_dsn()
            )
            saver = await context.__aenter__()
            try:
                if checkpoint_auto_setup():
                    await saver.setup()
            except Exception:
                await context.__aexit__(None, None, None)
                raise
            _backend_context = context
            _checkpointer = saver

        _active_backend = backend
        _active_location = location
        assert _checkpointer is not None
        return _checkpointer


async def _close_unlocked() -> None:
    global _checkpointer, _sqlite_connection, _backend_context
    global _active_backend, _active_location

    connection = _sqlite_connection
    context = _backend_context
    _checkpointer = None
    _sqlite_connection = None
    _backend_context = None
    _active_backend = None
    _active_location = None

    if connection is not None:
        await connection.close()
    if context is not None:
        await context.__aexit__(None, None, None)


async def close_checkpointer() -> None:
    async with _current_lock():
        await _close_unlocked()


async def delete_thread_checkpoint(thread_id: str) -> bool:
    saver = await get_checkpointer()
    config = {"configurable": {"thread_id": thread_id}}
    existed = await saver.aget_tuple(config) is not None
    if existed:
        await saver.adelete_thread(thread_id)
    return existed


async def read_thread_checkpoint(thread_id: str) -> dict[str, Any] | None:
    saver = await get_checkpointer()
    checkpoint_tuple = await saver.aget_tuple(
        {"configurable": {"thread_id": thread_id}}
    )
    if checkpoint_tuple is None:
        return None
    values = checkpoint_tuple.checkpoint.get("channel_values") or {}
    return dict(values) if isinstance(values, dict) else None


def _checkpoint_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


async def cleanup_expired_checkpoints(
    *,
    active_thread_ids: set[str] | None = None,
    now: datetime | None = None,
) -> CheckpointCleanupReport:
    """Delete stale threads in bounded batches using the saver abstraction.

    `alist()` returns newest checkpoints first for official savers. Only the first
    checkpoint seen for each thread is evaluated, so a thread is removed only
    when its latest scanned checkpoint is older than the retention cutoff.
    """

    retention_days = checkpoint_retention_days()
    backend = checkpoint_backend()
    if retention_days <= 0:
        return CheckpointCleanupReport(
            backend=backend,
            cutoff=None,
            retention_days=retention_days,
            scanned_checkpoints=0,
            unique_threads=0,
            stale_candidates=0,
            deleted_threads=[],
            skipped_active_threads=[],
            disabled=True,
        )

    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cutoff = current - timedelta(days=retention_days)
    active = active_thread_ids or set()
    saver = await get_checkpointer()
    seen_threads: set[str] = set()
    stale: list[str] = []
    skipped: list[str] = []
    scanned = 0

    async for item in saver.alist(
        None,
        limit=checkpoint_cleanup_scan_limit(),
    ):
        scanned += 1
        configurable = item.config.get("configurable", {})
        thread_id = str(configurable.get("thread_id", "")).strip()
        if not thread_id or thread_id in seen_threads:
            continue
        seen_threads.add(thread_id)

        timestamp = _checkpoint_timestamp(item.checkpoint.get("ts"))
        if timestamp is None or timestamp >= cutoff:
            continue
        if thread_id in active:
            skipped.append(thread_id)
        else:
            stale.append(thread_id)

    deleted: list[str] = []
    for thread_id in stale[: checkpoint_cleanup_batch_size()]:
        await saver.adelete_thread(thread_id)
        deleted.append(thread_id)

    return CheckpointCleanupReport(
        backend=backend,
        cutoff=cutoff.isoformat(),
        retention_days=retention_days,
        scanned_checkpoints=scanned,
        unique_threads=len(seen_threads),
        stale_candidates=len(stale),
        deleted_threads=deleted,
        skipped_active_threads=skipped,
    )
