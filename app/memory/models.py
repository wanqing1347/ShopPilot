from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

MemoryKind = Literal["preference", "blacklist", "history"]
MemoryScope = Literal["global", "category"]
MemoryStatus = Literal["active", "superseded", "conflicted"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    key: str
    kind: MemoryKind
    scope: MemoryScope
    content: str
    normalized_content: str
    category: str | None = None
    status: MemoryStatus = "active"
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    mention_count: int = Field(default=1, ge=1)
    source_sessions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_accessed_at: datetime | None = None
    access_count: int = Field(default=0, ge=0)
    conflict_group: str | None = None
    supersedes: list[str] = Field(default_factory=list)


class MemoryDocument(BaseModel):
    version: int = 2
    users: dict[str, list[MemoryEntry]] = Field(default_factory=dict)


class MemorySearchResult(BaseModel):
    user_id: str | None
    query: str
    entries: list[MemoryEntry] = Field(default_factory=list)

    @property
    def prompt_texts(self) -> list[str]:
        return [entry.content for entry in self.entries]


class MemoryWriteReport(BaseModel):
    user_id: str | None
    upserted: list[MemoryEntry] = Field(default_factory=list)
    superseded_ids: list[str] = Field(default_factory=list)
    unchanged_ids: list[str] = Field(default_factory=list)
