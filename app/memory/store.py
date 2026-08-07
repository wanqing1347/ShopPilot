from __future__ import annotations

import asyncio
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from app.agent.settings import (
    memory_confidence_increment,
    memory_file,
    memory_max_entries_per_user,
    memory_min_relevance,
    memory_retrieval_limit,
)
from app.memory.models import (
    MemoryDocument,
    MemoryEntry,
    MemoryKind,
    MemorySearchResult,
    MemoryScope,
    MemoryWriteReport,
    utc_now,
)
from app.utils.runtime import PROJECT_ROOT

_NEGATIVE_MARKERS = (
    "不要",
    "不喜欢",
    "不想要",
    "不接受",
    "不能接受",
    "避免",
    "排除",
    "拒绝",
    "禁用",
    "不含",
    "无",
)
_PREFERENCE_MARKERS = (
    "偏好",
    "喜欢",
    "倾向",
    "优先",
    "希望",
    "更喜欢",
    "常选",
)
_HISTORY_MARKERS = (
    "买过",
    "购买过",
    "曾买",
    "曾经买",
    "已购",
    "用过",
    "拥有",
)
_GENERIC_WORDS = (
    "用户",
    "我",
    "比较",
    "更加",
    "商品",
    "产品",
    "款式",
    "材质",
    "风格",
    "类型",
    "品牌",
)
_OPPOSING_KINDS: dict[MemoryKind, set[MemoryKind]] = {
    "preference": {"blacklist"},
    "blacklist": {"preference"},
    "history": set(),
}
_KIND_BASE_CONFIDENCE: dict[MemoryKind, float] = {
    "preference": 0.72,
    "blacklist": 0.82,
    "history": 0.65,
}


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).strip().lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _classify_kind(text: str) -> MemoryKind:
    normalized = unicodedata.normalize("NFKC", text).lower()
    if any(marker in normalized for marker in _NEGATIVE_MARKERS):
        return "blacklist"
    if any(marker in normalized for marker in _HISTORY_MARKERS):
        return "history"
    return "preference"


def _infer_scope(text: str, category: str | None, kind: MemoryKind) -> MemoryScope:
    if not category:
        return "global"
    normalized_text = _normalize_text(text)
    normalized_category = _normalize_text(category)
    if kind == "history" or (normalized_category and normalized_category in normalized_text):
        return "category"
    return "global"


def _canonical_subject(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    for marker in sorted(
        (*_NEGATIVE_MARKERS, *_PREFERENCE_MARKERS, *_HISTORY_MARKERS),
        key=len,
        reverse=True,
    ):
        normalized = normalized.replace(marker, "")
    for word in _GENERIC_WORDS:
        normalized = normalized.replace(word, "")
    subject = _normalize_text(normalized)
    return subject or _normalize_text(text)


def _memory_key(
    *,
    content: str,
    category: str | None,
    scope: MemoryScope,
) -> str:
    subject = _canonical_subject(content)
    if scope == "category":
        return f"category:{_normalize_text(category or 'unknown')}:{subject}"
    return f"global:{subject}"


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    latin = set(re.findall(r"[a-z0-9]+", normalized))
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", normalized)
    chinese: set[str] = set()
    for chunk in chinese_chunks:
        if len(chunk) == 1:
            chinese.add(chunk)
            continue
        chinese.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
        if len(chunk) >= 3:
            chinese.update(chunk[index : index + 3] for index in range(len(chunk) - 2))
    return latin | chinese


def _parse_datetime(value: datetime | str | None) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = utc_now()
    else:
        parsed = utc_now()
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _relevance_score(entry: MemoryEntry, query: str, now: datetime) -> float:
    query_normalized = _normalize_text(query)
    content_normalized = entry.normalized_content
    query_tokens = _tokens(query)
    entry_tokens = _tokens(f"{entry.content} {entry.category or ''}")

    overlap = 0.0
    if query_tokens and entry_tokens:
        overlap = len(query_tokens & entry_tokens) / max(1, len(entry_tokens))
    substring = 1.0 if content_normalized and content_normalized in query_normalized else 0.0
    reverse_substring = 0.8 if query_normalized and query_normalized in content_normalized else 0.0
    semantic = max(overlap, substring, reverse_substring)

    category_match = 0.0
    if entry.category:
        normalized_category = _normalize_text(entry.category)
        if normalized_category and normalized_category in query_normalized:
            category_match = 1.0
        elif _tokens(entry.category) & query_tokens:
            category_match = 0.7

    age_days = max(0.0, (now - _parse_datetime(entry.updated_at)).total_seconds() / 86400)
    recency = math.exp(-age_days / 180.0)
    scope_score = 1.0 if entry.scope == "global" else category_match
    kind_bonus = 1.0 if entry.kind == "blacklist" else 0.5 if entry.kind == "preference" else 0.0

    return (
        semantic * 0.42
        + entry.confidence * 0.25
        + category_match * 0.13
        + scope_score * 0.10
        + recency * 0.07
        + kind_bonus * 0.03
    )


class PreferenceStore:
    """Versioned, structured, user-scoped long-term memory store.

    The JSON backend remains intentionally local and inspectable. It supports legacy
    `user_id -> [string]` files, relevance retrieval, evidence reinforcement and
    deterministic conflict resolution. A future vector or database backend can keep
    this public interface.
    """

    def __init__(self, path: Path | None = None) -> None:
        configured = memory_file()
        self.path = path or (PROJECT_ROOT / configured)
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _current_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    def _legacy_entry(self, text: str, *, timestamp: datetime) -> MemoryEntry | None:
        content = text.strip()
        if not content:
            return None
        kind = _classify_kind(content)
        scope: MemoryScope = "global"
        return MemoryEntry(
            key=_memory_key(content=content, category=None, scope=scope),
            kind=kind,
            scope=scope,
            content=content,
            normalized_content=_normalize_text(content),
            confidence=0.5,
            source_sessions=["legacy-import"],
            created_at=timestamp,
            updated_at=timestamp,
        )

    def _read_document(self) -> tuple[MemoryDocument, bool]:
        if not self.path.exists():
            return MemoryDocument(), False
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return MemoryDocument(), False

        if isinstance(raw, dict) and raw.get("version") == 2 and isinstance(raw.get("users"), dict):
            try:
                return MemoryDocument.model_validate(raw), False
            except ValueError:
                return MemoryDocument(), False

        # Backward compatibility with the original `user_id -> [string]` format.
        if isinstance(raw, dict):
            try:
                timestamp = datetime.fromtimestamp(
                    self.path.stat().st_mtime,
                    tz=timezone.utc,
                )
            except OSError:
                timestamp = utc_now()
            users: dict[str, list[MemoryEntry]] = {}
            for user_id, values in raw.items():
                if not isinstance(values, list):
                    continue
                entries = [
                    entry
                    for value in values
                    if isinstance(value, str)
                    if (entry := self._legacy_entry(value, timestamp=timestamp)) is not None
                ]
                if entries:
                    users[str(user_id)] = entries
            return MemoryDocument(users=users), True
        return MemoryDocument(), False

    def _write_document(self, document: MemoryDocument) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @staticmethod
    def render_for_prompt(entries: Iterable[MemoryEntry]) -> list[str]:
        kind_names = {
            "preference": "偏好",
            "blacklist": "排除项",
            "history": "历史",
        }
        rendered: list[str] = []
        for entry in entries:
            scope = "全局" if entry.scope == "global" else f"品类:{entry.category or '未知'}"
            rendered.append(
                f"[{kind_names[entry.kind]}|{scope}|置信度{entry.confidence:.2f}] {entry.content}"
            )
        return rendered

    async def read_relevant(
        self,
        user_id: str | None,
        query: str = "",
        *,
        limit: int | None = None,
    ) -> MemorySearchResult:
        if not user_id:
            return MemorySearchResult(user_id=None, query=query)

        async with self._current_lock():
            document, migrated = self._read_document()
            entries = [
                entry
                for entry in document.users.get(user_id, [])
                if entry.status == "active"
            ]
            now = utc_now()
            requested_limit = limit or memory_retrieval_limit()
            if query.strip():
                scored = [(_relevance_score(entry, query, now), entry) for entry in entries]
                selected = [
                    entry
                    for score, entry in sorted(
                        scored,
                        key=lambda pair: (
                            pair[0],
                            pair[1].confidence,
                            _parse_datetime(pair[1].updated_at),
                        ),
                        reverse=True,
                    )
                    if score >= memory_min_relevance()
                ][:requested_limit]
            else:
                selected = sorted(
                    entries,
                    key=lambda entry: (
                        entry.confidence,
                        _parse_datetime(entry.updated_at),
                    ),
                    reverse=True,
                )[:requested_limit]

            if selected:
                selected_ids = {entry.id for entry in selected}
                for entry in document.users.get(user_id, []):
                    if entry.id in selected_ids:
                        entry.last_accessed_at = now
                        entry.access_count += 1
                self._write_document(document)
            elif migrated:
                self._write_document(document)

            return MemorySearchResult(user_id=user_id, query=query, entries=selected)

    async def write_many(
        self,
        user_id: str | None,
        texts: list[str],
        *,
        source_session: str | None = None,
        category: str | None = None,
    ) -> MemoryWriteReport:
        if not user_id or not texts:
            return MemoryWriteReport(user_id=user_id)

        async with self._current_lock():
            document, _ = self._read_document()
            current = document.users.setdefault(user_id, [])
            report = MemoryWriteReport(user_id=user_id)
            seen_input: set[tuple[str, MemoryKind]] = set()
            now = utc_now()

            for raw_text in texts:
                content = raw_text.strip()
                if not content:
                    continue
                kind = _classify_kind(content)
                scope = _infer_scope(content, category, kind)
                key = _memory_key(content=content, category=category, scope=scope)
                input_key = (key, kind)
                if input_key in seen_input:
                    continue
                seen_input.add(input_key)

                same_kind = next(
                    (
                        entry
                        for entry in sorted(
                            current,
                            key=lambda item: _parse_datetime(item.updated_at),
                            reverse=True,
                        )
                        if entry.key == key and entry.kind == kind
                    ),
                    None,
                )

                if same_kind is None:
                    target = MemoryEntry(
                        key=key,
                        kind=kind,
                        scope=scope,
                        content=content,
                        normalized_content=_normalize_text(content),
                        category=category if scope == "category" else None,
                        confidence=_KIND_BASE_CONFIDENCE[kind],
                        source_sessions=[source_session] if source_session else [],
                        created_at=now,
                        updated_at=now,
                    )
                    current.append(target)
                else:
                    target = same_kind
                    if source_session and source_session in target.source_sessions:
                        report.unchanged_ids.append(target.id)
                    else:
                        target.mention_count += 1
                        target.confidence = min(
                            0.98,
                            target.confidence + memory_confidence_increment(),
                        )
                        if source_session:
                            target.source_sessions = [
                                *target.source_sessions,
                                source_session,
                            ][-20:]
                    target.content = content
                    target.normalized_content = _normalize_text(content)
                    target.scope = scope
                    target.category = category if scope == "category" else None
                    target.updated_at = now
                    target.status = "active"

                opposing = [
                    entry
                    for entry in current
                    if entry.id != target.id
                    and entry.key == key
                    and entry.status == "active"
                    and entry.kind in _OPPOSING_KINDS[kind]
                ]
                if opposing:
                    target.conflict_group = key
                    target.supersedes = list(
                        dict.fromkeys([*target.supersedes, *(entry.id for entry in opposing)])
                    )
                    for entry in opposing:
                        entry.status = "superseded"
                        entry.conflict_group = key
                        entry.updated_at = now
                        report.superseded_ids.append(entry.id)

                report.upserted.append(target.model_copy(deep=True))

            current.sort(
                key=lambda entry: (
                    entry.status == "active",
                    _parse_datetime(entry.updated_at),
                ),
                reverse=True,
            )
            document.users[user_id] = current[: memory_max_entries_per_user()]
            self._write_document(document)
            return report

    async def list_entries(
        self,
        user_id: str,
        *,
        include_inactive: bool = True,
    ) -> list[MemoryEntry]:
        async with self._current_lock():
            document, migrated = self._read_document()
            if migrated:
                self._write_document(document)
            entries = document.users.get(user_id, [])
            selected = entries if include_inactive else [entry for entry in entries if entry.status == "active"]
            return [
                entry.model_copy(deep=True)
                for entry in sorted(
                    selected,
                    key=lambda item: _parse_datetime(item.updated_at),
                    reverse=True,
                )
            ]

    async def delete_entry(self, user_id: str, memory_id: str) -> bool:
        async with self._current_lock():
            document, _ = self._read_document()
            entries = document.users.get(user_id, [])
            remaining = [entry for entry in entries if entry.id != memory_id]
            if len(remaining) == len(entries):
                return False
            if remaining:
                document.users[user_id] = remaining
            else:
                document.users.pop(user_id, None)
            self._write_document(document)
            return True

    async def resolve_entry(self, user_id: str, memory_id: str) -> MemoryEntry | None:
        """Make one memory authoritative and supersede its opposite active entries."""

        async with self._current_lock():
            document, _ = self._read_document()
            entries = document.users.get(user_id, [])
            selected = next((entry for entry in entries if entry.id == memory_id), None)
            if selected is None:
                return None
            now = utc_now()
            selected.status = "active"
            selected.updated_at = now
            selected.confidence = max(selected.confidence, 0.9)
            selected.conflict_group = selected.conflict_group or selected.key
            superseded: list[str] = []
            for entry in entries:
                if (
                    entry.id != selected.id
                    and entry.key == selected.key
                    and entry.kind in _OPPOSING_KINDS[selected.kind]
                    and entry.status == "active"
                ):
                    entry.status = "superseded"
                    entry.conflict_group = selected.conflict_group
                    entry.updated_at = now
                    superseded.append(entry.id)
            selected.supersedes = list(dict.fromkeys([*selected.supersedes, *superseded]))
            self._write_document(document)
            return selected.model_copy(deep=True)


store = PreferenceStore()
