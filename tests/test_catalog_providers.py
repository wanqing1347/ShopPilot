from __future__ import annotations

import pytest

from app.catalog.base import CatalogSearchRequest
from app.catalog.router import clear_catalog_provider_cache, search_catalog


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["amazon", "walmart", "ebay"])
async def test_online_platforms_search_cached_snapshot(platform: str) -> None:
    clear_catalog_provider_cache()
    result = await search_catalog(
        CatalogSearchRequest(
            query="running shoes",
            platform=platform,
            category="鞋靴",
            category_key="footwear",
            top_k=3,
        )
    )

    assert result.provider == "offline_snapshot"
    assert result.live is False
    assert result.candidates
    assert all(item.platform == platform for item in result.candidates)
    assert all(item.data_origin == "offline_snapshot" for item in result.candidates)
    assert all(item.verification_status == "cached" for item in result.candidates)
    assert result.diagnostics["catalog_snapshot"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["amazon", "walmart"])
async def test_additional_offline_snapshot_platforms_search(platform: str) -> None:
    clear_catalog_provider_cache()
    result = await search_catalog(
        CatalogSearchRequest(
            query="wireless headphones",
            platform=platform,
            category="耳机",
            category_key="headphones",
            top_k=3,
        )
    )

    assert result.provider == "offline_snapshot"
    assert result.live is False
    assert result.candidates
    assert all(item.platform == platform for item in result.candidates)
    assert all(item.verification_status == "cached" for item in result.candidates)
    assert result.diagnostics["catalog_fallback"] is False


@pytest.mark.asyncio
async def test_offline_catalog_does_not_require_live_credentials() -> None:
    """The offline catalog remains usable without live-provider credentials."""

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

    assert result.provider == "offline_snapshot"
    assert result.fallback_reason is None
    assert "catalog_provider_chain" not in result.diagnostics
