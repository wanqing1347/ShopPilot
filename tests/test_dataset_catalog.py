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

    assert products
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
    assert summary["dataset_name"] == "shoppilot_offline_catalog"
    assert summary["total"] == len(products)
    assert summary["verification_status"] == "cached"
    assert summary["platform_counts"] == dict(
        Counter(product.platform for product in products)
    )


def test_offline_catalog_covers_all_snapshot_platforms() -> None:
    expected_platforms = {"amazon", "walmart", "ebay"}
    assert set(product.platform for product in load_catalog()) == expected_platforms


@pytest.mark.asyncio
async def test_item_search_reads_offline_candidates_without_mapping() -> None:
    output = await item_search(
        query="running shoes",
        platform="amazon",
        category="鞋靴",
        top_k=10,
        category_key="footwear",
    )

    available_amazon = [
        product
        for product in load_catalog()
        if product.category_key == "footwear"
        and product.platform == "amazon"
        and product.is_available
    ]
    assert output.total_recall == len(available_amazon)
    assert output.candidates
    assert all(candidate.platform == "amazon" for candidate in output.candidates)
    assert all(candidate.category_key == "footwear" for candidate in output.candidates)
    assert all(candidate.attributes["category"] == "鞋靴" for candidate in output.candidates)
    assert all(candidate in catalog_for_platform("amazon") for candidate in output.candidates)
