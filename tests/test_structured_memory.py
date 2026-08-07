from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.memory.store import PreferenceStore


@pytest.mark.asyncio
async def test_legacy_string_store_migrates_to_versioned_entries(tmp_path: Path) -> None:
    path = tmp_path / "preferences.json"
    path.write_text(
        json.dumps(
            {"legacy-user": ["不要塑料", "偏好陶瓷"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = PreferenceStore(path)

    result = await store.read_relevant(
        user_id="legacy-user",
        query="找一个不含塑料的陶瓷咖啡杯",
    )

    assert {entry.content for entry in result.entries} == {"不要塑料", "偏好陶瓷"}
    migrated = json.loads(path.read_text(encoding="utf-8"))
    assert migrated["version"] == 2
    assert migrated["users"]["legacy-user"][0]["source_sessions"] == [
        "legacy-import"
    ]
    assert all("confidence" in entry for entry in migrated["users"]["legacy-user"])


@pytest.mark.asyncio
async def test_repeated_evidence_increases_confidence_but_same_session_is_idempotent(
    tmp_path: Path,
) -> None:
    store = PreferenceStore(tmp_path / "preferences.json")

    first = await store.write_many(
        "u1",
        ["偏好陶瓷材质"],
        source_session="thread-1",
        category="咖啡杯",
    )
    first_entry = first.upserted[0]

    duplicate = await store.write_many(
        "u1",
        ["偏好陶瓷材质"],
        source_session="thread-1",
        category="咖啡杯",
    )
    duplicate_entry = duplicate.upserted[0]
    assert duplicate_entry.mention_count == 1
    assert duplicate_entry.confidence == first_entry.confidence
    assert duplicate.unchanged_ids == [first_entry.id]

    reinforced = await store.write_many(
        "u1",
        ["更喜欢陶瓷"],
        source_session="thread-2",
        category="咖啡杯",
    )
    reinforced_entry = reinforced.upserted[0]
    assert reinforced_entry.id == first_entry.id
    assert reinforced_entry.mention_count == 2
    assert reinforced_entry.confidence > first_entry.confidence
    assert reinforced_entry.source_sessions == ["thread-1", "thread-2"]


@pytest.mark.asyncio
async def test_new_opposite_memory_supersedes_old_and_can_be_manually_resolved(
    tmp_path: Path,
) -> None:
    store = PreferenceStore(tmp_path / "preferences.json")
    preferred = (
        await store.write_many(
            "u2",
            ["偏好塑料"],
            source_session="thread-a",
        )
    ).upserted[0]
    blocked_report = await store.write_many(
        "u2",
        ["不要塑料"],
        source_session="thread-b",
    )
    blocked = blocked_report.upserted[0]

    entries = await store.list_entries("u2", include_inactive=True)
    by_id = {entry.id: entry for entry in entries}
    assert by_id[preferred.id].status == "superseded"
    assert by_id[blocked.id].status == "active"
    assert preferred.id in blocked.supersedes
    assert blocked_report.superseded_ids == [preferred.id]

    retrieved = await store.read_relevant("u2", "买一个塑料收纳盒")
    assert [entry.id for entry in retrieved.entries] == [blocked.id]

    selected = await store.resolve_entry("u2", preferred.id)
    assert selected is not None
    assert selected.status == "active"
    assert selected.confidence >= 0.9
    entries = await store.list_entries("u2", include_inactive=True)
    by_id = {entry.id: entry for entry in entries}
    assert by_id[preferred.id].status == "active"
    assert by_id[blocked.id].status == "superseded"


@pytest.mark.asyncio
async def test_relevance_filters_unrelated_category_history_but_keeps_global_profile(
    tmp_path: Path,
) -> None:
    store = PreferenceStore(tmp_path / "preferences.json")
    await store.write_many(
        "u3",
        ["买过咖啡杯"],
        source_session="coffee-history",
        category="咖啡杯",
    )
    await store.write_many(
        "u3",
        ["偏好轻量"],
        source_session="global-style",
        category="旅行收纳",
    )

    laptop = await store.read_relevant("u3", "想买一台适合出差的笔记本电脑")
    assert [entry.content for entry in laptop.entries] == ["偏好轻量"]

    coffee = await store.read_relevant("u3", "再买一个咖啡杯")
    assert {entry.content for entry in coffee.entries} == {"买过咖啡杯", "偏好轻量"}


@pytest.mark.asyncio
async def test_delete_memory_removes_only_selected_entry(tmp_path: Path) -> None:
    store = PreferenceStore(tmp_path / "preferences.json")
    report = await store.write_many(
        "u4",
        ["偏好小众", "不要塑料"],
        source_session="thread-delete",
    )
    deleted_id = report.upserted[0].id

    assert await store.delete_entry("u4", deleted_id) is True
    assert await store.delete_entry("u4", deleted_id) is False
    remaining = await store.list_entries("u4")
    assert len(remaining) == 1
    assert remaining[0].id != deleted_id
