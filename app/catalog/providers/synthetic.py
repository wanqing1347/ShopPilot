from __future__ import annotations

import asyncio

from app.agent.settings import dataset_dir
from app.catalog.base import CatalogSearchRequest, CatalogSearchResult
from app.recall.catalog import resolve_dataset_root
from app.recall.hybrid import (
    SearchFilters,
    exclusions_from_constraints,
    get_hybrid_retriever,
)


class SyntheticCatalogProvider:
    name = "offline_snapshot"
    supported_platforms = {"amazon", "walmart", "ebay"}

    async def search(self, request: CatalogSearchRequest) -> CatalogSearchResult:
        if request.platform not in self.supported_platforms:
            return CatalogSearchResult(
                candidates=[],
                total_candidates=0,
                provider=self.name,
                live=False,
                diagnostics={
                    "catalog_provider": self.name,
                    "catalog_live": False,
                    "mode": "offline_snapshot",
                    "eligible_count": 0,
                    "returned_count": 0,
                    "offline_partition_available": False,
                    "unsupported_platform": request.platform,
                    "resolved_category_key": request.category_key,
                },
            )

        configured_dataset_dir = resolve_dataset_root(dataset_dir())
        provider_name = self.name
        retriever = await asyncio.to_thread(get_hybrid_retriever, configured_dataset_dir)
        result = await asyncio.to_thread(
            retriever.search,
            query=request.query,
            filters=SearchFilters(
                platform=request.platform,
                category=request.category,
                category_key=request.category_key,
                max_landed_price_cny=request.budget_cny,
                excluded_terms=exclusions_from_constraints(
                    list(request.hard_constraints)
                ),
            ),
            top_k=request.top_k,
            user_preferences=list(request.user_preferences),
        )
        diagnostics = {
            **result.diagnostics,
            "catalog_provider": provider_name,
            "catalog_live": False,
            "catalog_snapshot": True,
            "top_scores": [
                {
                    "item_id": hit.candidate.item_id,
                    "final": hit.final_score,
                    "rrf": hit.rrf_score,
                    "rerank": hit.rerank_score,
                    "learned": hit.learned_score,
                    "bm25_rank": hit.bm25_rank,
                    "vector_rank": hit.vector_rank,
                }
                for hit in result.hits[:5]
            ],
        }
        return CatalogSearchResult(
            candidates=[hit.candidate for hit in result.hits],
            total_candidates=result.total_candidates,
            provider=provider_name,
            live=False,
            diagnostics=diagnostics,
        )
