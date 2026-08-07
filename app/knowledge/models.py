from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnowledgeDocument:
    doc_id: str
    category_key: str
    category_path: tuple[str, ...]
    title: str
    content: str
    source: str
    updated_at: str | None = None

    @property
    def category(self) -> str:
        return self.category_path[-1] if self.category_path else self.category_key

    @property
    def text(self) -> str:
        return " ".join(
            part
            for part in [
                self.category,
                self.category_key,
                " ".join(self.category_path),
                self.title,
                self.content,
            ]
            if part
        )


@dataclass(frozen=True)
class KnowledgeHit:
    document: KnowledgeDocument
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    bm25_rank: int | None = None
    vector_rank: int | None = None


@dataclass(frozen=True)
class KnowledgeSearchResult:
    hits: list[KnowledgeHit]
    total_candidates: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
