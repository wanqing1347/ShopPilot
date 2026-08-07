from __future__ import annotations

import pytest

from app.catalog.base import CatalogSearchRequest
from app.catalog.providers.public_demo import clear_public_demo_cache
from app.catalog.router import clear_catalog_provider_cache, search_catalog
from app.tools.price_compare import price_compare
from app.tools.shipping_calc import shipping_calc


@pytest.fixture(autouse=True)
def reset_public_demo(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHOPPILOT_PUBLIC_DEMO_ENABLED", "true")
    monkeypatch.setenv(
        "SHOPPILOT_PUBLIC_DEMO_DATA_FILE",
        "data/public_demo/products.jsonl",
    )
    clear_public_demo_cache()
    clear_catalog_provider_cache()
    yield
    clear_public_demo_cache()
    clear_catalog_provider_cache()


@pytest.mark.asyncio
async def test_public_demo_snapshot_searches_footwear() -> None:
    result = await search_catalog(
        CatalogSearchRequest(
            query="想在公开演示商城找运动鞋",
            platform="public_demo",
            category="鞋子",
            category_key="footwear",
            top_k=5,
        )
    )

    assert result.live is False
    assert result.provider == "public_demo_catalog_snapshot"
    assert result.total_candidates >= 20
    assert result.candidates
    assert all(item.platform == "public_demo" for item in result.candidates)
    assert all(item.category_key == "footwear" for item in result.candidates)
    assert all(item.verification_status == "public_demo" for item in result.candidates)
    assert all(item.attributes["mock_commerce_data"] is True for item in result.candidates)
    assert result.diagnostics["snapshot_count"] == 1000
    assert result.diagnostics["source_count"] >= 10
    assert result.diagnostics["distinct_category_keys"] >= 25


@pytest.mark.asyncio
async def test_public_demo_applies_budget_and_preserves_sources() -> None:
    result = await search_catalog(
        CatalogSearchRequest(
            query="儿童运动鞋 kids light-up sneakers",
            platform="public_demo",
            category="鞋子",
            category_key="footwear",
            top_k=20,
            budget_cny=300,
        )
    )

    assert result.candidates
    assert all((item.landed_price_cny or item.price_cny or 0) <= 300 for item in result.candidates)
    assert all(item.source_url for item in result.candidates)
    assert any(
        item.item_id == "public-demo:web-scraping-dev:10"
        for item in result.candidates
    )
    assert result.diagnostics["catalog_snapshot"] is True
    assert "not live marketplace" in result.diagnostics["source_notice"]


@pytest.mark.asyncio
async def test_public_demo_recognizes_smartphones_and_headphones() -> None:
    smartphones = await search_catalog(
        CatalogSearchRequest(
            query="智能手机",
            platform="public_demo",
            category="手机",
            category_key="smartphones",
            top_k=5,
        )
    )
    headphones = await search_catalog(
        CatalogSearchRequest(
            query="蓝牙耳机",
            platform="public_demo",
            category="耳机",
            category_key="headphones",
            top_k=5,
        )
    )

    assert smartphones.candidates
    assert all(item.category_key == "smartphones" for item in smartphones.candidates)
    assert headphones.candidates
    assert all(item.category_key == "headphones" for item in headphones.candidates)


@pytest.mark.asyncio
async def test_public_demo_books_keep_genres_searchable() -> None:
    result = await search_catalog(
        CatalogSearchRequest(
            query="悬疑小说 mystery",
            platform="public_demo",
            category="图书",
            category_key="books",
            top_k=10,
        )
    )

    assert result.candidates
    assert all(item.category_key == "books" for item in result.candidates)
    assert any(
        str(item.attributes.get("genre", "")).lower() in {"mystery", "suspense", "crime", "thriller"}
        for item in result.candidates
    )


@pytest.mark.asyncio
async def test_public_demo_recognizes_tools_and_collectibles() -> None:
    tools = await search_catalog(
        CatalogSearchRequest(
            query="五金工具 钳子 螺丝刀",
            platform="public_demo",
            category="工具",
            category_key="tools",
            top_k=5,
        )
    )
    collectibles = await search_catalog(
        CatalogSearchRequest(
            query="宝可梦收藏玩具",
            platform="public_demo",
            category="收藏玩具",
            category_key="collectibles",
            top_k=5,
        )
    )

    assert tools.candidates
    assert all(item.category_key == "tools" for item in tools.candidates)
    assert any(
        item.attributes.get("catalog_source") == "practice-software-testing"
        for item in tools.candidates
    )
    assert collectibles.candidates
    assert all(item.category_key == "collectibles" for item in collectibles.candidates)
    assert all(
        item.attributes.get("catalog_source") == "scrapeme"
        for item in collectibles.candidates
    )


@pytest.mark.asyncio
async def test_public_demo_category_distribution_is_capped() -> None:
    result = await search_catalog(
        CatalogSearchRequest(
            query="食品",
            platform="public_demo",
            category="食品",
            category_key="groceries",
            top_k=1,
        )
    )

    counts = result.diagnostics["source_counts"]
    assert counts["open-prices"] >= 100
    assert counts["books-to-scrape"] <= 200
    assert result.total_candidates == 200


@pytest.mark.asyncio
async def test_public_demo_has_zero_mock_shipping_and_duty() -> None:
    result = await search_catalog(
        CatalogSearchRequest(
            query="巧克力食品",
            platform="public_demo",
            category="食品",
            category_key="groceries",
            top_k=1,
        )
    )
    compared = await price_compare(result.candidates, top_n=1)
    landed = await shipping_calc(compared.ranked, destination="CN")

    assert landed.items[0].shipping_cny == 0.0
    assert landed.items[0].duty_cny == 0.0
    assert landed.items[0].landed_cny == landed.items[0].price_cny
