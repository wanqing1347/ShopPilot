from __future__ import annotations

import time

from app.api.monitor import monitor
from app.catalog import CatalogSearchRequest, search_catalog
from app.models import ItemSearchOutput, Platform


async def item_search(
    query: str,
    platform: Platform,
    category: str,
    top_k: int = 20,
    user_preferences: list[str] | None = None,
    *,
    category_key: str | None = None,
    budget_cny: float | None = None,
    hard_constraints: list[str] | None = None,
) -> ItemSearchOutput:
    """Search one platform through its configured live or synthetic catalog provider."""

    top_k = max(1, min(top_k, 50))
    preferences = user_preferences or []
    await monitor.report_tool_start(
        "item_search",
        {
            "query": query,
            "platform": platform,
            "category": category,
            "category_key": category_key,
            "budget_cny": budget_cny,
            "top_k": top_k,
        },
    )
    started = time.perf_counter()
    result = await search_catalog(
        CatalogSearchRequest(
            query=query,
            platform=platform,
            category=category,
            top_k=top_k,
            user_preferences=tuple(preferences),
            category_key=category_key,
            budget_cny=budget_cny,
            hard_constraints=tuple(hard_constraints or ()),
        )
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    diagnostics = {
        **result.diagnostics,
        "catalog_provider": result.provider,
        "catalog_live": result.live,
        "catalog_fallback_reason": result.fallback_reason,
    }
    await monitor.report_retrieval_search(
        platform=platform,
        category_key=str(diagnostics.get("resolved_category_key") or category_key or ""),
        mode=str(diagnostics.get("mode") or "hybrid"),
        embedding_provider=str(diagnostics.get("embedding_provider") or "unknown"),
        vector_engine=str(diagnostics.get("vector_engine") or "unknown"),
        reranker=str(diagnostics.get("reranker_applied") or "rules"),
        eligible_count=int(diagnostics.get("eligible_count") or 0),
        returned_count=len(result.candidates),
        duration_ms=duration_ms,
    )
    await monitor.report_tool_end("item_search", duration_ms)
    return ItemSearchOutput(
        platform=platform,
        candidates=result.candidates,
        total_recall=result.total_candidates,
        truncated=result.total_candidates > top_k,
        retrieval=diagnostics,
    )
