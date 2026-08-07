from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from langgraph.checkpoint.base import CheckpointTuple

from app.agent import checkpoint


class _FakeSaver:
    def __init__(self, items: list[CheckpointTuple]) -> None:
        self.items = items
        self.deleted: list[str] = []

    async def alist(self, config, *, filter=None, before=None, limit=None):
        for item in self.items[:limit]:
            yield item

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)


def _tuple(thread_id: str, timestamp: datetime) -> CheckpointTuple:
    return CheckpointTuple(
        config={"configurable": {"thread_id": thread_id}},
        checkpoint={
            "v": 4,
            "id": f"checkpoint-{thread_id}-{timestamp.timestamp()}",
            "ts": timestamp.isoformat(),
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "updated_channels": None,
        },
        metadata={"source": "loop", "step": 1, "parents": {}},
    )


@pytest.mark.asyncio
async def test_cleanup_deletes_only_expired_inactive_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 8, 5, 6, 0, tzinfo=timezone.utc)
    saver = _FakeSaver(
        [
            _tuple("fresh", now - timedelta(days=2)),
            _tuple("active-old", now - timedelta(days=60)),
            _tuple("stale-a", now - timedelta(days=45)),
            _tuple("stale-a", now - timedelta(days=90)),
            _tuple("stale-b", now - timedelta(days=31)),
        ]
    )

    async def fake_get_checkpointer():
        return saver

    monkeypatch.setattr(checkpoint, "get_checkpointer", fake_get_checkpointer)
    monkeypatch.setenv("SHOPPILOT_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("SHOPPILOT_CHECKPOINT_RETENTION_DAYS", "30")
    monkeypatch.setenv("SHOPPILOT_CHECKPOINT_CLEANUP_BATCH_SIZE", "10")
    monkeypatch.setenv("SHOPPILOT_CHECKPOINT_CLEANUP_SCAN_LIMIT", "100")

    report = await checkpoint.cleanup_expired_checkpoints(
        active_thread_ids={"active-old"},
        now=now,
    )

    assert saver.deleted == ["stale-a", "stale-b"]
    assert report.deleted_threads == ["stale-a", "stale-b"]
    assert report.skipped_active_threads == ["active-old"]
    assert report.unique_threads == 4
    assert report.scanned_checkpoints == 5
    assert report.stale_candidates == 2


@pytest.mark.asyncio
async def test_cleanup_respects_batch_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 8, 5, 6, 0, tzinfo=timezone.utc)
    saver = _FakeSaver(
        [
            _tuple("stale-a", now - timedelta(days=40)),
            _tuple("stale-b", now - timedelta(days=41)),
        ]
    )

    async def fake_get_checkpointer():
        return saver

    monkeypatch.setattr(checkpoint, "get_checkpointer", fake_get_checkpointer)
    monkeypatch.setenv("SHOPPILOT_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("SHOPPILOT_CHECKPOINT_RETENTION_DAYS", "30")
    monkeypatch.setenv("SHOPPILOT_CHECKPOINT_CLEANUP_BATCH_SIZE", "1")

    report = await checkpoint.cleanup_expired_checkpoints(now=now)

    assert saver.deleted == ["stale-a"]
    assert report.stale_candidates == 2
    assert report.deleted_threads == ["stale-a"]


@pytest.mark.asyncio
async def test_retention_zero_disables_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHOPPILOT_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("SHOPPILOT_CHECKPOINT_RETENTION_DAYS", "0")

    report = await checkpoint.cleanup_expired_checkpoints()

    assert report.disabled is True
    assert report.deleted_threads == []


def test_postgres_description_masks_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHOPPILOT_CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv(
        "SHOPPILOT_CHECKPOINT_POSTGRES_DSN",
        "postgresql://secret-user:secret-pass@db.example.com:5432/shoppilot",
    )

    description = checkpoint.checkpoint_description()

    assert description["backend"] == "postgres"
    assert description["location"] == "postgresql://db.example.com:5432/shoppilot"
    assert "secret" not in str(description)
