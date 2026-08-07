from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

from app.agent.settings import public_demo_data_file
from app.catalog.base import CatalogSearchRequest, CatalogSearchResult
from app.catalog.public_demo_taxonomy import CATEGORY_ALIASES
from app.models import Candidate
from app.recall.hybrid import exclusions_from_constraints
from app.recall.tokenizer import normalize_text, tokenize
from app.utils.runtime import PROJECT_ROOT


class PublicDemoProviderError(RuntimeError):
    """Raised when the local public-demo catalog cannot be loaded."""


def _snapshot_path() -> Path:
    configured = Path(public_demo_data_file()).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (PROJECT_ROOT / configured).resolve()


@lru_cache(maxsize=4)
def _load_snapshot(path_text: str, modified_ns: int) -> tuple[Candidate, ...]:
    del modified_ns
    path = Path(path_text)
    rows: list[Candidate] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                candidate = Candidate.model_validate_json(line)
            except Exception as exc:
                raise PublicDemoProviderError(
                    f"公开演示商品数据格式错误: {path}:{line_no}: {exc}"
                ) from exc
            if candidate.platform != "public_demo":
                raise PublicDemoProviderError(
                    f"公开演示快照包含错误平台: {candidate.item_id} -> {candidate.platform}"
                )
            rows.append(candidate)
    if not rows:
        raise PublicDemoProviderError(f"公开演示商品快照为空: {path}")
    return tuple(rows)


def clear_public_demo_cache() -> None:
    _load_snapshot.cache_clear()


def _resolved_category(request: CatalogSearchRequest) -> str | None:
    if request.category_key in CATEGORY_ALIASES:
        return request.category_key
    source = normalize_text(
        " ".join(
            [
                request.query,
                request.category,
                *(request.user_preferences or ()),
            ]
        )
    )
    for category_key, aliases in CATEGORY_ALIASES.items():
        if any(normalize_text(alias) in source for alias in aliases):
            return category_key
    return None


def _attribute_text(candidate: Candidate, key: str) -> str:
    value = candidate.attributes.get(key)
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value or "")


def _candidate_text(candidate: Candidate) -> str:
    return " ".join(
        [
            candidate.title,
            candidate.title_en or "",
            candidate.description,
            candidate.brand or "",
            " ".join(candidate.category_path),
            _attribute_text(candidate, "category"),
            _attribute_text(candidate, "source_category"),
            _attribute_text(candidate, "category_aliases"),
            _attribute_text(candidate, "tags"),
            _attribute_text(candidate, "genre"),
            _attribute_text(candidate, "genre_zh"),
        ]
    )


def _eligible(candidate: Candidate, request: CatalogSearchRequest) -> bool:
    if not candidate.is_available:
        return False
    price = candidate.landed_price_cny or candidate.price_cny
    if request.budget_cny is not None and price is not None and price > request.budget_cny:
        return False
    excluded = exclusions_from_constraints(list(request.hard_constraints))
    if excluded:
        haystack = normalize_text(_candidate_text(candidate))
        if any(normalize_text(term) in haystack for term in excluded):
            return False
    return True


def _rank(candidates: list[Candidate], query_text: str) -> list[tuple[float, Candidate]]:
    query_tokens = set(tokenize(query_text))
    tokenized = [set(tokenize(_candidate_text(candidate))) for candidate in candidates]
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(tokens)
    total = max(1, len(candidates))
    ranked: list[tuple[float, Candidate]] = []
    for candidate, product_tokens in zip(candidates, tokenized, strict=True):
        matched = query_tokens & product_tokens
        lexical_score = 0.0
        if matched:
            lexical_score = sum(
                math.log((total + 1) / (document_frequency[token] + 1)) + 1.0
                for token in matched
            ) / max(1.0, math.sqrt(len(query_tokens)))
        rating_score = float(candidate.rating or 0.0) / 5.0
        source_quality = 0.05 if candidate.provider in {"dummyjson_snapshot", "books_to_scrape_snapshot"} else 0.0
        score = lexical_score * 0.88 + rating_score * 0.10 + source_quality
        ranked.append((score, candidate))
    ranked.sort(
        key=lambda pair: (
            -pair[0],
            -(pair[1].rating or 0.0),
            pair[1].price_cny or float("inf"),
            pair[1].item_id,
        )
    )
    return ranked


class PublicDemoCatalogProvider:
    """Search a diversified local snapshot built from mock-commerce sources."""

    name = "public_demo_catalog_snapshot"

    async def search(self, request: CatalogSearchRequest) -> CatalogSearchResult:
        if request.platform != "public_demo":
            raise PublicDemoProviderError(
                "PublicDemoCatalogProvider 只支持 public_demo 平台"
            )
        path = _snapshot_path()
        if not path.exists():
            raise PublicDemoProviderError(
                "缺少公开演示商品快照。请运行 scripts/build_public_demo_catalog.py；"
                f"当前路径: {path}"
            )
        snapshot = list(_load_snapshot(str(path), path.stat().st_mtime_ns))
        category_key = _resolved_category(request)
        candidates = snapshot
        if category_key:
            candidates = [
                candidate for candidate in candidates if candidate.category_key == category_key
            ]
        candidates = [candidate for candidate in candidates if _eligible(candidate, request)]
        query_text = " ".join(
            [
                request.query,
                request.category,
                *(request.user_preferences or ()),
            ]
        )
        ranked = _rank(candidates, query_text)
        selected = [candidate for _, candidate in ranked[: request.top_k]]
        source_counts = Counter(
            str(candidate.attributes.get("catalog_source") or candidate.provider)
            for candidate in snapshot
        )
        category_counts = Counter(candidate.category_key or "unknown" for candidate in snapshot)
        return CatalogSearchResult(
            candidates=selected,
            total_candidates=len(candidates),
            provider=self.name,
            live=False,
            diagnostics={
                "catalog_provider": self.name,
                "catalog_live": False,
                "catalog_snapshot": True,
                "snapshot_path": str(path),
                "snapshot_count": len(snapshot),
                "source_count": len(source_counts),
                "source_counts": dict(sorted(source_counts.items())),
                "distinct_category_keys": len(category_counts),
                "eligible_count": len(candidates),
                "returned_count": len(selected),
                "resolved_category_key": category_key,
                "mode": "public_demo_multi_source_snapshot",
                "embedding_provider": "none",
                "vector_engine": "token_idf",
                "reranker_applied": "token_idf_plus_rating",
                "source_notice": (
                    "Mock-commerce product data for retrieval testing and interview demos; "
                    "not live marketplace inventory or checkout offers."
                ),
            },
        )
