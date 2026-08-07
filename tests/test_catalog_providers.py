from __future__ import annotations

import pytest

from app.catalog.base import CatalogSearchRequest
from app.catalog.router import clear_catalog_provider_cache, search_catalog


@pytest.mark.asyncio
async def test_synthetic_marketplace_partition_remains_available() -> None:
    clear_catalog_provider_cache()
    result = await search_catalog(
        CatalogSearchRequest(
            query="陶瓷咖啡杯",
            platform="ebay",
            category="咖啡杯",
            category_key="coffee_cup",
            top_k=3,
        )
    )

    assert result.provider == "synthetic_hybrid"
    assert result.live is False
    assert result.candidates
    assert all(item.platform == "ebay" for item in result.candidates)
    assert all(item.verification_status == "synthetic" for item in result.candidates)
    assert result.diagnostics["catalog_fallback"] is False


@pytest.mark.asyncio
async def test_unknown_live_credentials_are_not_part_of_runtime() -> None:
    """The runtime no longer depends on unavailable official API credentials."""

    clear_catalog_provider_cache()
    result = await search_catalog(
        CatalogSearchRequest(
            query="机械键盘",
            platform="amazon",
            category="键盘",
            category_key="keyboard",
            top_k=2,
        )
    )

    assert result.provider == "synthetic_hybrid"
    assert result.fallback_reason is None
    assert "catalog_provider_chain" not in result.diagnostics
