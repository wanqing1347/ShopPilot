from __future__ import annotations

import json
from collections import Counter

import pytest

from app.models import Candidate
from app.recall.catalog import (
    catalog_for_platform,
    clear_catalog_cache,
    dataset_root,
    load_catalog,
)
from app.tools.item_search import item_search


def test_schema_v2_dataset_loads_directly_as_candidates() -> None:
    clear_catalog_cache()
    products = load_catalog()

    assert len(products) == 1200
    assert all(isinstance(product, Candidate) for product in products)
    assert all(product.schema_version == 2 for product in products)
    assert all(product.same_group_id for product in products)
    assert all(product.price >= 0 and product.currency for product in products)
    assert all(product.attributes["category_key"] == product.category_key for product in products)
    assert all(product.attributes["category_path"] == product.category_path for product in products)

    summary = json.loads(
        (dataset_root() / "dataset_summary.json").read_text(encoding="utf-8")
    )
    assert summary["schema_version"] == 2
    assert summary["compatible_project"] == "shoppilot-agent>=0.8.0"
    assert summary["counts"]["platform_items"] == 1200


def test_coffee_cup_has_four_platforms_and_complete_groups() -> None:
    coffee = [product for product in load_catalog() if product.category_key == "coffee_cup"]

    assert len(coffee) == 200
    assert Counter(product.platform for product in coffee) == {
        "amazon": 50,
        "shopee": 50,
        "aliexpress": 50,
        "ebay": 50,
    }
    grouped: dict[str, set[str]] = {}
    for product in coffee:
        grouped.setdefault(product.same_group_id, set()).add(product.platform)
    assert len(grouped) == 50
    assert all(platforms == {"amazon", "shopee", "aliexpress", "ebay"} for platforms in grouped.values())


@pytest.mark.asyncio
async def test_item_search_reads_coffee_candidates_without_mapping() -> None:
    output = await item_search(
        query="预算 300 元，想买小众手作粗陶咖啡杯，最好带壶嘴",
        platform="amazon",
        category="咖啡杯",
        top_k=10,
        user_preferences=["偏好小众手作", "偏好粗陶"],
    )

    available_amazon = [
        product
        for product in load_catalog()
        if product.category_key == "coffee_cup"
        and product.platform == "amazon"
        and product.is_available
    ]
    assert output.total_recall == len(available_amazon)
    assert output.candidates
    assert all(candidate.platform == "amazon" for candidate in output.candidates)
    assert all(candidate.category_key == "coffee_cup" for candidate in output.candidates)
    assert all(candidate.attributes["category"] == "咖啡杯" for candidate in output.candidates)
    assert all(candidate.same_group_id.startswith("G-COF-") for candidate in output.candidates)
    assert all(candidate in catalog_for_platform("amazon") for candidate in output.candidates)
