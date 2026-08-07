from __future__ import annotations

from functools import lru_cache

from app.agent.settings import public_demo_enabled
from app.catalog.base import CatalogSearchRequest, CatalogSearchResult
from app.catalog.providers.public_demo import (
    PublicDemoCatalogProvider,
    PublicDemoProviderError,
    clear_public_demo_cache,
)
from app.catalog.providers.synthetic import SyntheticCatalogProvider


@lru_cache(maxsize=1)
def _public_demo_provider() -> PublicDemoCatalogProvider:
    return PublicDemoCatalogProvider()


@lru_cache(maxsize=1)
def _synthetic_provider() -> SyntheticCatalogProvider:
    return SyntheticCatalogProvider()


def clear_catalog_provider_cache() -> None:
    _public_demo_provider.cache_clear()
    _synthetic_provider.cache_clear()
    clear_public_demo_cache()


async def search_catalog(request: CatalogSearchRequest) -> CatalogSearchResult:
    """Route requests to the two currently usable local data sources.

    The four marketplace labels in the synthetic dataset represent offline,
    large-marketplace-style product partitions. They are suitable for demos and
    evaluation, but are never presented as live official-platform results.

    ``public_demo`` is an optional local mock-commerce catalog for extra product
    categories when the user explicitly asks for a demo catalog.
    """

    if request.platform == "public_demo" and public_demo_enabled():
        try:
            return await _public_demo_provider().search(request)
        except PublicDemoProviderError as exc:
            fallback = await _synthetic_provider().search(request)
            fallback.fallback_reason = f"{type(exc).__name__}: {str(exc)[:300]}"
            fallback.diagnostics.update(
                {
                    "catalog_fallback": True,
                    "catalog_fallback_reason": fallback.fallback_reason,
                    "catalog_provider_chain": [
                        "public_demo_catalog_snapshot",
                        "synthetic_hybrid",
                    ],
                }
            )
            return fallback

    result = await _synthetic_provider().search(request)
    result.diagnostics.setdefault("catalog_fallback", False)
    return result
