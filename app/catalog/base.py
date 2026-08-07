from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.models import Candidate, Platform


@dataclass(frozen=True)
class CatalogSearchRequest:
    query: str
    platform: Platform
    category: str
    top_k: int = 20
    user_preferences: tuple[str, ...] = ()
    category_key: str | None = None
    budget_cny: float | None = None
    hard_constraints: tuple[str, ...] = ()


@dataclass
class CatalogSearchResult:
    candidates: list[Candidate]
    total_candidates: int
    provider: str
    live: bool
    diagnostics: dict[str, object] = field(default_factory=dict)
    fallback_reason: str | None = None


class CatalogProvider(Protocol):
    name: str

    async def search(self, request: CatalogSearchRequest) -> CatalogSearchResult:
        """Search one source and normalize the result into Candidate objects."""
