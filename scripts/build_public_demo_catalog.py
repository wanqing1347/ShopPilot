from __future__ import annotations

import argparse
import html
import json
import math
import re
import tempfile
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import httpx

from app.catalog.public_demo_taxonomy import (
    BOOK_GENRE_ZH,
    CATEGORY_LABEL_ZH,
    aliases_for,
    infer_category_key,
)
from app.models import Candidate
from app.utils.currency import FX_RATES, to_base
from app.utils.runtime import PROJECT_ROOT
from scrape_public_demo import scrape as scrape_web_scraping_dev

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "public_demo" / "products.jsonl"
USER_AGENT = "ShopPilotCatalogBuilder/1.1 (educational public-demo catalog)"

DUMMYJSON_URL = "https://dummyjson.com/products?limit=0"
PLATZI_URL = "https://api.escuelajs.co/api/v1/products?offset=0&limit=1000"
AUTOMATION_EXERCISE_URL = "https://automationexercise.com/api/productsList"
PRACTICE_TOOLS_URL = "https://api.practicesoftwaretesting.com/products"
FAKESTORE_URL = "https://fakestoreapi.com/products"
MOCK_SHOP_URL = "https://mock.shop/api"
VENDURE_URL = "https://readonlydemo.vendure.io/shop-api"
OPEN_PRICES_URL = "https://prices.openfoodfacts.org/api/v1/prices"
SCRAPEME_BASE_URL = "https://scrapeme.live/"
SCRAPEME_ROBOTS_URL = urljoin(SCRAPEME_BASE_URL, "robots.txt")
BOOKS_BASE_URL = "https://books.toscrape.com/"
BOOKS_HOME_URL = urljoin(BOOKS_BASE_URL, "index.html")

RATING_MAP = {"One": 1.0, "Two": 2.0, "Three": 3.0, "Four": 4.0, "Five": 5.0}
CORE_SOURCE_NAMES = {
    "dummyjson",
    "platzi",
    "web-scraping-dev",
    "automation-exercise",
    "practice-software-testing",
    "fakestoreapi",
    "mock-shop",
    "vendure-demo",
}
SOURCE_POLICIES: dict[str, dict[str, object]] = {
    "dummyjson": {"type": "fake_api", "purpose": "testing and prototyping"},
    "platzi": {"type": "fake_api", "purpose": "testing and learning"},
    "web-scraping-dev": {
        "type": "mock_commerce_fixture",
        "robots_respected": True,
        "request_delay_seconds": 2.0,
    },
    "automation-exercise": {"type": "automation_practice_api"},
    "practice-software-testing": {"type": "software_testing_demo_api"},
    "fakestoreapi": {"type": "fake_api"},
    "mock-shop": {"type": "mock_store_graphql_api"},
    "vendure-demo": {"type": "read_only_demo_graphql_api"},
    "open-prices": {
        "type": "open_price_observations",
        "license": "ODbL",
        "not_current_inventory": True,
    },
    "scrapeme": {"type": "mock_commerce_fixture", "robots_respected": True},
    "books-to-scrape": {"type": "mock_book_catalog"},
}


class CatalogBuildError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_text(value: object, *, limit: int = 500) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split()).strip()
    return text[:limit]


def safe_float(value: object) -> float | None:
    try:
        result = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    if result is None or not math.isfinite(result):
        return None
    return result


def safe_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def stable_slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:120] or "item"


def category_metadata(category_key: str, source_category: str) -> tuple[str, list[str]]:
    category_zh = CATEGORY_LABEL_ZH.get(category_key, "其他商品")
    terms = aliases_for(category_key, source_category)
    return category_zh, terms


def common_candidate(
    *,
    source: str,
    source_id: str,
    provider: str,
    title: str,
    description: str,
    category_key: str,
    source_category: str,
    price: float,
    currency: str,
    retrieved_at: str,
    brand: str | None = None,
    rating: float | None = None,
    review_count: int | None = None,
    stock: int | None = None,
    is_available: bool = True,
    image_url: str | None = None,
    source_url: str | None = None,
    tags: list[str] | None = None,
    original_price: float | None = None,
    source_type: str = "public_demo_or_fake_api",
    mock_commerce_data: bool = True,
    quality_grade: str = "B",
    extra_attributes: dict[str, Any] | None = None,
) -> Candidate:
    category_zh, terms = category_metadata(category_key, source_category)
    currency = currency.upper()
    if currency not in FX_RATES:
        raise CatalogBuildError(f"未配置演示汇率: {currency}")
    price_cny = round(to_base(price, currency), 2)
    original_price_cny = (
        round(to_base(original_price, currency), 2)
        if original_price is not None and original_price >= price
        else None
    )
    searchable_terms = list(dict.fromkeys([*terms, *(tags or [])]))
    enriched_description = clean_text(
        f"{description} 品类关键词: {' '.join(searchable_terms)}",
        limit=750,
    )
    attributes: dict[str, Any] = {
        "category": category_zh,
        "category_key": category_key,
        "category_path": ["Public demo catalog", category_zh],
        "source_category": source_category,
        "category_aliases": searchable_terms,
        "tags": tags or [],
        "catalog_source": source,
        "source_type": source_type,
        "mock_commerce_data": mock_commerce_data,
        "public_test_or_open_data": True,
    }
    if extra_attributes:
        attributes.update(extra_attributes)
    identifier = f"{source}:{source_id}"
    return Candidate(
        item_id=f"public-demo:{identifier}",
        same_group_id=f"PUBLIC-DEMO:{identifier.upper()}",
        platform="public_demo",
        title=title,
        title_en=title,
        description=enriched_description,
        brand=brand,
        category_key=category_key,
        category_path=["Public demo catalog", category_zh],
        price=round(price, 2),
        currency=currency,
        price_cny=price_cny,
        original_price_cny=original_price_cny,
        shipping_cny=0.0,
        landed_price_cny=price_cny,
        rating=rating,
        review_count=review_count,
        stock=stock,
        is_available=is_available,
        image_url=image_url,
        attributes=attributes,
        ingested_at=retrieved_at,
        quality_grade=quality_grade,
        data_origin="public_demo_catalog",
        provider=provider,
        source_url=source_url,
        retrieved_at=retrieved_at,
        verification_status="public_demo",
    )


def request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    params: dict[str, object] | None = None,
    payload: dict[str, object] | None = None,
    attempts: int = 3,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.request(method, url, params=params, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
    raise CatalogBuildError(f"读取公开测试数据失败: {url}: {last_error}") from last_error


def fetch_dummyjson(client: httpx.Client, retrieved_at: str) -> list[Candidate]:
    payload = request_json(client, "GET", DUMMYJSON_URL)
    rows: list[Candidate] = []
    for item in payload.get("products", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        title = clean_text(item.get("title"), limit=180)
        price = safe_float(item.get("price"))
        source_id = str(item.get("id") or "").strip()
        source_category = clean_text(item.get("category"), limit=80).lower()
        if not title or price is None or not source_id:
            continue
        category_key = infer_category_key(source_category, title)
        discount = safe_float(item.get("discountPercentage"))
        original_price = (
            price / max(0.01, 1.0 - discount / 100.0)
            if discount is not None and 0 < discount < 100
            else None
        )
        reviews = item.get("reviews") if isinstance(item.get("reviews"), list) else []
        rows.append(
            common_candidate(
                source="dummyjson",
                source_id=source_id,
                provider="dummyjson_snapshot",
                title=title,
                description=clean_text(item.get("description")),
                category_key=category_key,
                source_category=source_category,
                price=price,
                original_price=original_price,
                currency="USD",
                retrieved_at=retrieved_at,
                brand=clean_text(item.get("brand"), limit=100) or None,
                rating=safe_float(item.get("rating")),
                review_count=len(reviews) or None,
                stock=safe_int(item.get("stock")),
                is_available=str(item.get("availabilityStatus") or "").lower() != "out of stock",
                image_url=clean_text(item.get("thumbnail"), limit=500) or None,
                source_url=f"https://dummyjson.com/products/{source_id}",
                tags=[clean_text(tag, limit=80) for tag in item.get("tags", []) if clean_text(tag, limit=80)],
                extra_attributes={
                    "sku": clean_text(item.get("sku"), limit=80) or None,
                    "weight": safe_float(item.get("weight")),
                    "warranty_information": clean_text(item.get("warrantyInformation"), limit=160) or None,
                    "shipping_information": clean_text(item.get("shippingInformation"), limit=160) or None,
                    "return_policy": clean_text(item.get("returnPolicy"), limit=160) or None,
                },
            )
        )
    return rows


def normalize_platzi_category(value: object) -> str:
    normalized = clean_text(value, limit=120).lower()
    rules = (
        ("clothes", "clothes"),
        ("electronics", "electronics"),
        ("furniture", "furniture"),
        ("shoes", "shoes"),
        ("food", "groceries"),
        ("beverage", "groceries"),
        ("station", "miscellaneous"),
        ("miscellaneous", "miscellaneous"),
    )
    for token, category in rules:
        if token in normalized:
            return category
    return "miscellaneous"


def fetch_platzi(client: httpx.Client, retrieved_at: str) -> list[Candidate]:
    payload = request_json(client, "GET", PLATZI_URL)
    rows: list[Candidate] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("id") or "").strip()
        title = clean_text(item.get("title"), limit=180)
        price = safe_float(item.get("price"))
        category = item.get("category") if isinstance(item.get("category"), dict) else {}
        source_category = normalize_platzi_category(category.get("name"))
        if not source_id or not title or price is None:
            continue
        category_key = infer_category_key(source_category, title)
        images = item.get("images") if isinstance(item.get("images"), list) else []
        image_url = next(
            (
                clean_text(value, limit=500)
                for value in images
                if clean_text(value, limit=500).startswith("http")
            ),
            None,
        )
        rows.append(
            common_candidate(
                source="platzi",
                source_id=source_id,
                provider="platzi_fake_store_snapshot",
                title=title,
                description=clean_text(item.get("description")),
                category_key=category_key,
                source_category=source_category,
                price=price,
                currency="USD",
                retrieved_at=retrieved_at,
                image_url=image_url,
                source_url=f"https://api.escuelajs.co/api/v1/products/{source_id}",
                tags=[source_category],
                extra_attributes={"category_id": category.get("id")},
            )
        )
    return rows


def fetch_web_scraping_dev(retrieved_at: str) -> list[Candidate]:
    with tempfile.TemporaryDirectory(prefix="shoppilot-web-scraping-dev-") as temp_dir:
        temp_output = Path(temp_dir) / "products.jsonl"
        source_rows = scrape_web_scraping_dev(limit=25, pages=5, output=temp_output)
    rows: list[Candidate] = []
    for item in source_rows:
        source_id = item.item_id.rsplit(":", 1)[-1]
        source_category = clean_text(item.attributes.get("category"), limit=80).lower()
        category_key = infer_category_key(source_category, item.title)
        rows.append(
            common_candidate(
                source="web-scraping-dev",
                source_id=source_id,
                provider="web_scraping_dev_snapshot",
                title=item.title,
                description=item.description,
                category_key=category_key,
                source_category=source_category,
                price=float(item.price),
                currency=item.currency,
                retrieved_at=retrieved_at,
                brand=item.brand,
                rating=item.rating,
                review_count=item.review_count,
                stock=item.stock,
                is_available=item.is_available,
                image_url=item.image_url,
                source_url=item.source_url,
                tags=[source_category],
                source_type="mock_commerce_fixture",
                extra_attributes={"robots_respected": True, "request_delay_seconds": 2.0},
            )
        )
    return rows


def fetch_automation_exercise(client: httpx.Client, retrieved_at: str) -> list[Candidate]:
    payload = request_json(client, "GET", AUTOMATION_EXERCISE_URL)
    rows: list[Candidate] = []
    for item in payload.get("products", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("id") or "").strip()
        title = clean_text(item.get("name"), limit=180)
        price_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(item.get("price") or ""))
        category = item.get("category") if isinstance(item.get("category"), dict) else {}
        usertype = category.get("usertype") if isinstance(category.get("usertype"), dict) else {}
        source_category = clean_text(
            f"{usertype.get('usertype', '')} {category.get('category', '')}", limit=120
        ).lower()
        if not source_id or not title or not price_match:
            continue
        category_key = infer_category_key(source_category, title)
        rows.append(
            common_candidate(
                source="automation-exercise",
                source_id=source_id,
                provider="automation_exercise_api_snapshot",
                title=title,
                description="Automation Exercise practice-store product.",
                category_key=category_key,
                source_category=source_category,
                price=float(price_match.group(1)),
                currency="INR",
                retrieved_at=retrieved_at,
                brand=clean_text(item.get("brand"), limit=100) or None,
                source_url=f"https://automationexercise.com/product_details/{source_id}",
                tags=[source_category, "automation practice"],
                source_type="automation_practice_api",
            )
        )
    return rows


def fetch_practice_tools(client: httpx.Client, retrieved_at: str) -> list[Candidate]:
    first = request_json(client, "GET", PRACTICE_TOOLS_URL, params={"page": 1})
    if not isinstance(first, dict):
        return []
    last_page = max(1, safe_int(first.get("last_page")) or 1)
    payloads = [first]
    for page in range(2, last_page + 1):
        payloads.append(request_json(client, "GET", PRACTICE_TOOLS_URL, params={"page": page}))
    rows: list[Candidate] = []
    for payload in payloads:
        for item in payload.get("data", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("id") or "").strip()
            title = clean_text(item.get("name"), limit=180)
            price = safe_float(item.get("price"))
            category = item.get("category") if isinstance(item.get("category"), dict) else {}
            source_category = clean_text(category.get("name"), limit=100).lower() or "tools"
            brand_payload = item.get("brand") if isinstance(item.get("brand"), dict) else {}
            image_payload = item.get("product_image") if isinstance(item.get("product_image"), dict) else {}
            file_name = clean_text(image_payload.get("file_name"), limit=200)
            if not source_id or not title or price is None:
                continue
            rows.append(
                common_candidate(
                    source="practice-software-testing",
                    source_id=source_id,
                    provider="practice_software_testing_api_snapshot",
                    title=title,
                    description=clean_text(item.get("description"), limit=600),
                    category_key=infer_category_key(source_category, title),
                    source_category=source_category,
                    price=price,
                    currency="USD",
                    retrieved_at=retrieved_at,
                    brand=clean_text(brand_payload.get("name"), limit=100) or None,
                    stock=None,
                    is_available=bool(item.get("in_stock")),
                    image_url=(
                        f"https://api.practicesoftwaretesting.com/images/products/{file_name}"
                        if file_name
                        else None
                    ),
                    source_url=f"https://practicesoftwaretesting.com/product/{source_id}",
                    tags=[source_category, "tools", "software testing"],
                    source_type="software_testing_demo_api",
                    extra_attributes={
                        "co2_rating": item.get("co2_rating"),
                        "eco_friendly": bool(item.get("is_eco_friendly")),
                        "rental": bool(item.get("is_rental")),
                    },
                )
            )
    return rows


def fetch_fakestore(client: httpx.Client, retrieved_at: str) -> list[Candidate]:
    payload = request_json(client, "GET", FAKESTORE_URL)
    rows: list[Candidate] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("id") or "").strip()
        title = clean_text(item.get("title"), limit=180)
        price = safe_float(item.get("price"))
        source_category = clean_text(item.get("category"), limit=100).lower()
        rating_payload = item.get("rating") if isinstance(item.get("rating"), dict) else {}
        if not source_id or not title or price is None:
            continue
        rows.append(
            common_candidate(
                source="fakestoreapi",
                source_id=source_id,
                provider="fakestoreapi_snapshot",
                title=title,
                description=clean_text(item.get("description"), limit=600),
                category_key=infer_category_key(source_category, title),
                source_category=source_category,
                price=price,
                currency="USD",
                retrieved_at=retrieved_at,
                rating=safe_float(rating_payload.get("rate")),
                review_count=safe_int(rating_payload.get("count")),
                image_url=clean_text(item.get("image"), limit=500) or None,
                source_url=f"https://fakestoreapi.com/products/{source_id}",
                tags=[source_category, "fake store"],
                source_type="fake_api",
            )
        )
    return rows


def fetch_mock_shop(client: httpx.Client, retrieved_at: str) -> list[Candidate]:
    query = """
    query Catalog($first: Int!, $after: String) {
      products(first: $first, after: $after) {
        edges { node {
          id title description handle vendor productType tags
          featuredImage { url }
          priceRange { minVariantPrice { amount currencyCode } }
        } }
        pageInfo { hasNextPage endCursor }
      }
    }
    """
    after: str | None = None
    rows: list[Candidate] = []
    while True:
        payload = request_json(
            client,
            "POST",
            MOCK_SHOP_URL,
            payload={"query": query, "variables": {"first": 100, "after": after}},
        )
        connection = ((payload.get("data") or {}).get("products") or {}) if isinstance(payload, dict) else {}
        for edge in connection.get("edges", []):
            item = edge.get("node") if isinstance(edge, dict) else None
            if not isinstance(item, dict):
                continue
            source_id = stable_slug(str(item.get("id") or item.get("handle") or item.get("title") or ""))
            title = clean_text(item.get("title"), limit=180)
            price_payload = ((item.get("priceRange") or {}).get("minVariantPrice") or {})
            price = safe_float(price_payload.get("amount"))
            currency = clean_text(price_payload.get("currencyCode"), limit=10).upper()
            source_category = clean_text(item.get("productType"), limit=100).lower()
            tags = [clean_text(tag, limit=80) for tag in item.get("tags", []) if clean_text(tag, limit=80)]
            if not source_category:
                source_category = "apparel"
            if not source_id or not title or price is None or currency not in FX_RATES:
                continue
            image = item.get("featuredImage") if isinstance(item.get("featuredImage"), dict) else {}
            handle = clean_text(item.get("handle"), limit=200)
            rows.append(
                common_candidate(
                    source="mock-shop",
                    source_id=source_id,
                    provider="shopify_mock_shop_snapshot",
                    title=title,
                    description=clean_text(item.get("description"), limit=600),
                    category_key=infer_category_key(source_category, title),
                    source_category=source_category,
                    price=price,
                    currency=currency,
                    retrieved_at=retrieved_at,
                    brand=clean_text(item.get("vendor"), limit=100) or None,
                    image_url=clean_text(image.get("url"), limit=500) or None,
                    source_url=f"https://mock.shop/products/{handle}" if handle else "https://mock.shop/",
                    tags=[*tags, "shopify mock shop"],
                    source_type="mock_store_graphql_api",
                )
            )
        page_info = connection.get("pageInfo") if isinstance(connection.get("pageInfo"), dict) else {}
        if not page_info.get("hasNextPage"):
            break
        after = str(page_info.get("endCursor") or "") or None
        if not after:
            break
    return rows


def fetch_vendure(client: httpx.Client, retrieved_at: str) -> list[Candidate]:
    query = """
    query Catalog {
      products(options: { take: 100 }) {
        totalItems
        items {
          id name slug description
          featuredAsset { preview }
          variants { priceWithTax currencyCode stockLevel }
          collections { name slug }
        }
      }
    }
    """
    payload = request_json(client, "POST", VENDURE_URL, payload={"query": query})
    connection = ((payload.get("data") or {}).get("products") or {}) if isinstance(payload, dict) else {}
    rows: list[Candidate] = []
    for item in connection.get("items", []):
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("id") or "").strip()
        title = clean_text(item.get("name"), limit=180)
        variants = item.get("variants") if isinstance(item.get("variants"), list) else []
        valid_variants = [variant for variant in variants if isinstance(variant, dict)]
        if not source_id or not title or not valid_variants:
            continue
        variant = min(valid_variants, key=lambda value: safe_float(value.get("priceWithTax")) or float("inf"))
        cents = safe_float(variant.get("priceWithTax"))
        currency = clean_text(variant.get("currencyCode"), limit=10).upper()
        if cents is None or currency not in FX_RATES:
            continue
        collections = item.get("collections") if isinstance(item.get("collections"), list) else []
        collection_names = [
            clean_text(value.get("name"), limit=100)
            for value in collections
            if isinstance(value, dict) and clean_text(value.get("name"), limit=100)
        ]
        source_category = " ".join(collection_names).lower() or "electronics"
        asset = item.get("featuredAsset") if isinstance(item.get("featuredAsset"), dict) else {}
        slug = clean_text(item.get("slug"), limit=200)
        rows.append(
            common_candidate(
                source="vendure-demo",
                source_id=source_id,
                provider="vendure_readonly_demo_snapshot",
                title=title,
                description=clean_text(item.get("description"), limit=600),
                category_key=infer_category_key(source_category, title),
                source_category=source_category,
                price=cents / 100.0,
                currency=currency,
                retrieved_at=retrieved_at,
                stock=safe_int(variant.get("stockLevel")),
                image_url=clean_text(asset.get("preview"), limit=500) or None,
                source_url=f"https://demo.vendure.io/products/{slug}" if slug else "https://demo.vendure.io/",
                tags=[*collection_names, "vendure demo"],
                source_type="read_only_demo_graphql_api",
            )
        )
    return rows


def humanize_open_food_tag(value: str) -> str:
    tag = value.split(":", 1)[-1]
    return clean_text(tag.replace("-", " "), limit=120)


def fetch_open_prices(
    client: httpx.Client,
    retrieved_at: str,
    *,
    pages: int = 5,
    page_size: int = 100,
) -> list[Candidate]:
    latest_by_code: dict[str, dict[str, Any]] = {}
    for page in range(1, pages + 1):
        payload = request_json(
            client,
            "GET",
            OPEN_PRICES_URL,
            params={"page": page, "size": page_size, "ordering": "-date"},
        )
        for item in payload.get("items", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            product = item.get("product") if isinstance(item.get("product"), dict) else {}
            code = clean_text(item.get("product_code") or product.get("code"), limit=80)
            name = clean_text(item.get("product_name") or product.get("product_name"), limit=220)
            price = safe_float(item.get("price"))
            currency = clean_text(item.get("currency"), limit=10).upper()
            if not code or not name or price is None or price <= 0 or currency not in FX_RATES:
                continue
            latest_by_code.setdefault(code, item)
    rows: list[Candidate] = []
    for code, item in latest_by_code.items():
        product = item.get("product") if isinstance(item.get("product"), dict) else {}
        categories = [
            humanize_open_food_tag(str(tag))
            for tag in product.get("categories_tags", [])
            if clean_text(tag, limit=160)
        ]
        source_category = categories[-1].lower() if categories else "groceries"
        price_id = str(item.get("id") or code)
        name = clean_text(item.get("product_name") or product.get("product_name"), limit=220)
        brands = clean_text(product.get("brands"), limit=160) or None
        rows.append(
            common_candidate(
                source="open-prices",
                source_id=code,
                provider="open_food_facts_open_prices_snapshot",
                title=name,
                description=(
                    "Open Prices historical food-price observation linked to an Open Food Facts product. "
                    f"Observed date: {clean_text(item.get('date'), limit=30)}."
                ),
                category_key="groceries",
                source_category=source_category,
                price=float(item["price"]),
                currency=str(item["currency"]),
                retrieved_at=retrieved_at,
                brand=brands,
                image_url=clean_text(product.get("image_url"), limit=500) or None,
                source_url=f"https://prices.openfoodfacts.org/prices/{price_id}",
                tags=[*categories, "open food facts", "open prices", "食品价格"],
                source_type="open_price_observation",
                mock_commerce_data=False,
                quality_grade="A",
                extra_attributes={
                    "open_data": True,
                    "license": "ODbL",
                    "product_code": code,
                    "observed_date": item.get("date"),
                    "price_per": item.get("price_per"),
                    "price_is_discounted": item.get("price_is_discounted"),
                    "availability_unknown": True,
                    "not_current_inventory": True,
                },
            )
        )
    rows.sort(key=lambda row: (str(row.attributes.get("source_category")), row.title, row.item_id))
    return rows


def parse_robots(text: str) -> tuple[list[str], float | None]:
    disallowed: list[str] = []
    crawl_delay: float | None = None
    applies = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            applies = value == "*"
        elif applies and key == "disallow" and value:
            disallowed.append(value)
        elif applies and key == "crawl-delay":
            crawl_delay = safe_float(value)
    return disallowed, crawl_delay


def assert_robots_allowed(url: str, disallowed: Iterable[str]) -> None:
    path = urlparse(url).path or "/"
    for prefix in disallowed:
        if prefix == "/" or path.startswith(prefix):
            raise CatalogBuildError(f"robots.txt 不允许访问: {url} (Disallow: {prefix})")


def parse_scrapeme_page(page_html: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    blocks = re.findall(
        r'<li\s+class=["\']([^"\']*\bproduct\b[^"\']*)["\']>(.*?)</li>',
        page_html,
        flags=re.I | re.S,
    )
    for class_names, block in blocks:
        url_match = re.search(r'<a\s+href=["\']([^"\']+)["\'][^>]*woocommerce-LoopProduct-link', block, flags=re.I)
        title_match = re.search(r'<h2[^>]*woocommerce-loop-product__title[^>]*>(.*?)</h2>', block, flags=re.I | re.S)
        price_match = re.search(
            r'woocommerce-Price-currencySymbol[^>]*>\s*(?:£|&pound;|&#163;)\s*</span>\s*([0-9]+(?:\.[0-9]+)?)',
            block,
            flags=re.I | re.S,
        )
        image_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', block, flags=re.I)
        if not url_match or not title_match or not price_match:
            continue
        class_tokens = class_names.split()
        categories = [token.removeprefix("product_cat-") for token in class_tokens if token.startswith("product_cat-")]
        tags = [token.removeprefix("product_tag-") for token in class_tokens if token.startswith("product_tag-")]
        rows.append(
            {
                "url": url_match.group(1),
                "title": clean_text(title_match.group(1), limit=180),
                "price": float(price_match.group(1)),
                "image_url": image_match.group(1) if image_match else None,
                "categories": categories,
                "tags": tags,
                "is_available": "outofstock" not in class_tokens,
            }
        )
    return rows


def fetch_scrapeme(
    client: httpx.Client,
    retrieved_at: str,
    *,
    max_items: int = 180,
    delay_seconds: float = 0.25,
) -> list[Candidate]:
    robots = client.get(SCRAPEME_ROBOTS_URL)
    robots.raise_for_status()
    disallowed, declared_delay = parse_robots(robots.text)
    delay = max(delay_seconds, declared_delay or 0.0)
    rows: list[Candidate] = []
    page = 1
    last_request = 0.0
    while len(rows) < max_items:
        page_url = urljoin(SCRAPEME_BASE_URL, "shop/") if page == 1 else urljoin(SCRAPEME_BASE_URL, f"shop/page/{page}/")
        assert_robots_allowed(page_url, disallowed)
        wait = delay - (time.monotonic() - last_request)
        if wait > 0:
            time.sleep(wait)
        response = client.get(page_url)
        last_request = time.monotonic()
        if response.status_code == 404:
            break
        response.raise_for_status()
        parsed = parse_scrapeme_page(response.text)
        if not parsed:
            break
        for item in parsed:
            source_url = str(item["url"])
            source_id = stable_slug(urlparse(source_url).path.rstrip("/").split("/")[-1])
            source_categories = [clean_text(value, limit=80) for value in item["categories"]]
            source_category = "pokemon" if "pokemon" in source_categories else "collectibles"
            rows.append(
                common_candidate(
                    source="scrapeme",
                    source_id=source_id,
                    provider="scrapeme_practice_store_snapshot",
                    title=str(item["title"]),
                    description="Mock-commerce collectible product for retrieval testing.",
                    category_key="collectibles",
                    source_category=source_category,
                    price=float(item["price"]),
                    currency="GBP",
                    retrieved_at=retrieved_at,
                    is_available=bool(item["is_available"]),
                    image_url=str(item["image_url"]) if item["image_url"] else None,
                    source_url=source_url,
                    tags=[*source_categories, *item["tags"], "collectibles", "收藏玩具"],
                    source_type="mock_commerce_fixture",
                    extra_attributes={
                        "robots_respected": True,
                        "request_delay_seconds": delay,
                    },
                )
            )
            if len(rows) >= max_items:
                break
        page += 1
    return rows


def parse_book_categories(home_html: str) -> list[tuple[str, str]]:
    matches = re.findall(
        r'<a\s+href=["\'](catalogue/category/books/[^"\']+/index\.html)["\']>\s*([^<]+?)\s*</a>',
        home_html,
        flags=re.I | re.S,
    )
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for href, name in matches:
        category = clean_text(name, limit=100)
        url = urljoin(BOOKS_BASE_URL, href)
        if url not in seen:
            seen.add(url)
            rows.append((category, url))
    return rows


def parse_book_page(page_html: str, page_url: str, category: str) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    for block in re.findall(r'<article\s+class=["\']product_pod["\']>(.*?)</article>', page_html, flags=re.I | re.S):
        title_match = re.search(
            r'<h3>\s*<a\s+href=["\']([^"\']+)["\'][^>]*title=["\']([^"\']+)["\']',
            block,
            flags=re.I | re.S,
        )
        price_match = re.search(
            r'<p\s+class=["\']price_color["\']>\s*(?:&pound;|£)\s*([0-9.]+)',
            block,
            flags=re.I,
        )
        rating_match = re.search(
            r'<p\s+class=["\']star-rating\s+(One|Two|Three|Four|Five)["\']',
            block,
            flags=re.I,
        )
        image_match = re.search(r'<img\s+src=["\']([^"\']+)["\']', block, flags=re.I)
        if not title_match or not price_match:
            continue
        href, raw_title = title_match.groups()
        products.append(
            {
                "title": clean_text(raw_title, limit=220),
                "price": float(price_match.group(1)),
                "rating": RATING_MAP.get((rating_match.group(1) if rating_match else "").title()),
                "image_url": urljoin(page_url, image_match.group(1)) if image_match else None,
                "source_url": urljoin(page_url, href),
                "category": category,
                "is_available": "In stock" in clean_text(block, limit=2000),
            }
        )
    return products


def fetch_books(
    client: httpx.Client,
    retrieved_at: str,
    *,
    per_category: int = 3,
    delay_seconds: float = 0.08,
) -> dict[str, list[Candidate]]:
    home = client.get(BOOKS_HOME_URL)
    home.raise_for_status()
    categories = parse_book_categories(home.text)
    if not categories:
        raise CatalogBuildError("Books to Scrape 未发现品类链接")
    buckets: dict[str, list[Candidate]] = defaultdict(list)
    last_request = 0.0
    seen_urls: set[str] = set()
    for category, page_url in categories:
        wait = delay_seconds - (time.monotonic() - last_request)
        if wait > 0:
            time.sleep(wait)
        response = client.get(page_url)
        last_request = time.monotonic()
        response.raise_for_status()
        for item in parse_book_page(response.text, str(response.url), category)[:per_category]:
            source_url = str(item["source_url"])
            if source_url in seen_urls:
                continue
            seen_urls.add(source_url)
            source_id = urlparse(source_url).path.rstrip("/").split("/")[-2]
            genre_normalized = " ".join(category.lower().split())
            genre_zh = BOOK_GENRE_ZH.get(genre_normalized, category)
            buckets[category].append(
                common_candidate(
                    source="books-to-scrape",
                    source_id=stable_slug(source_id),
                    provider="books_to_scrape_snapshot",
                    title=str(item["title"]),
                    description=f"公开测试图书目录样例。Genre: {category}; 中文题材: {genre_zh}。",
                    category_key="books",
                    source_category=category,
                    price=float(item["price"]),
                    currency="GBP",
                    retrieved_at=retrieved_at,
                    brand="Books to Scrape",
                    rating=safe_float(item["rating"]),
                    is_available=bool(item["is_available"]),
                    image_url=str(item["image_url"]) if item["image_url"] else None,
                    source_url=source_url,
                    tags=[category, genre_zh, "books", "图书"],
                    source_type="mock_book_catalog",
                    extra_attributes={"genre": category, "genre_zh": genre_zh},
                )
            )
    return buckets


def bucket_by_category(rows: Iterable[Candidate]) -> dict[str, list[Candidate]]:
    buckets: dict[str, list[Candidate]] = defaultdict(list)
    for row in rows:
        buckets[row.category_key].append(row)
    return buckets


def balanced_take(buckets: dict[str, list[Candidate]], count: int) -> list[Candidate]:
    queues = {key: deque(value) for key, value in sorted(buckets.items()) if value}
    selected: list[Candidate] = []
    while queues and len(selected) < count:
        exhausted: list[str] = []
        for key in list(queues):
            queue = queues[key]
            if queue and len(selected) < count:
                selected.append(queue.popleft())
            if not queue:
                exhausted.append(key)
        for key in exhausted:
            queues.pop(key, None)
    return selected


def deduplicate(rows: Iterable[Candidate]) -> list[Candidate]:
    result: list[Candidate] = []
    seen_ids: set[str] = set()
    seen_source_urls: set[str] = set()
    for row in rows:
        if row.item_id in seen_ids:
            continue
        if row.source_url and row.source_url in seen_source_urls:
            continue
        seen_ids.add(row.item_id)
        if row.source_url:
            seen_source_urls.add(row.source_url)
        result.append(row)
    return result


def fill_to_target(
    selected: list[Candidate],
    pools: dict[str, list[Candidate]],
    *,
    target: int,
    category_cap: int,
) -> list[Candidate]:
    selected = deduplicate(selected)
    selected_ids = {row.item_id for row in selected}
    category_counts = Counter(row.category_key for row in selected)
    queues = {
        source: deque(row for row in rows if row.item_id not in selected_ids)
        for source, rows in pools.items()
    }
    while len(selected) < target and any(queues.values()):
        progress = False
        for source in sorted(queues):
            queue = queues[source]
            if not queue or len(selected) >= target:
                continue
            candidate: Candidate | None = None
            for _ in range(len(queue)):
                value = queue.popleft()
                if category_counts[value.category_key] < category_cap:
                    candidate = value
                    break
                queue.append(value)
            if candidate is None:
                continue
            if candidate.item_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.item_id)
            category_counts[candidate.category_key] += 1
            progress = True
        if not progress:
            break
    return selected


def write_snapshot(
    rows: list[Candidate],
    output: Path,
    *,
    target: int,
    retrieved_at: str,
    source_caps: dict[str, int],
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(row.model_dump_json() for row in rows) + "\n", encoding="utf-8")
    temporary.replace(output)
    source_counts = Counter(str(row.attributes.get("catalog_source")) for row in rows)
    category_counts = Counter(row.category_key or "unknown" for row in rows)
    source_category_counts = Counter(str(row.attributes.get("source_category")) for row in rows)
    summary: dict[str, Any] = {
        "retrieved_at": retrieved_at,
        "target_count": target,
        "count": len(rows),
        "distinct_category_keys": len(category_counts),
        "distinct_source_categories": len(source_category_counts),
        "source_counts": dict(sorted(source_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "source_category_counts": dict(sorted(source_category_counts.items())),
        "source_caps": source_caps,
        "source_policies": SOURCE_POLICIES,
        "output": str(output),
        "notice": (
            "Public test, mock-commerce and open-price sample data only; "
            "not live marketplace inventory or checkout prices."
        ),
    }
    output.with_name("snapshot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build(*, target: int, output: Path) -> tuple[list[Candidate], dict[str, Any]]:
    retrieved_at = utc_now()
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    }
    with httpx.Client(headers=headers, timeout=45.0, follow_redirects=True) as client:
        core_rows = deduplicate(
            [
                *fetch_dummyjson(client, retrieved_at),
                *fetch_platzi(client, retrieved_at),
                *fetch_web_scraping_dev(retrieved_at),
                *fetch_automation_exercise(client, retrieved_at),
                *fetch_practice_tools(client, retrieved_at),
                *fetch_fakestore(client, retrieved_at),
                *fetch_mock_shop(client, retrieved_at),
                *fetch_vendure(client, retrieved_at),
            ]
        )
        open_price_rows = fetch_open_prices(client, retrieved_at, pages=3, page_size=100)
        scrapeme_rows = fetch_scrapeme(client, retrieved_at, max_items=220)
        book_buckets = fetch_books(client, retrieved_at, per_category=4)
        book_rows = balanced_take(book_buckets, 200)

    # These caps prevent the largest public datasets from dominating the catalog.
    scale = target / 1000.0
    source_caps = {
        "open-prices": max(50, round(150 * scale)),
        "scrapeme": max(40, round(150 * scale)),
        "books-to-scrape": max(40, round(120 * scale)),
    }
    selected = [
        *core_rows,
        *balanced_take(bucket_by_category(open_price_rows), source_caps["open-prices"]),
        *balanced_take(bucket_by_category(scrapeme_rows), source_caps["scrapeme"]),
        *balanced_take(bucket_by_category(book_rows), source_caps["books-to-scrape"]),
    ]
    pools = {
        "open-prices": open_price_rows,
        "scrapeme": scrapeme_rows,
        "books-to-scrape": book_rows,
    }
    category_cap = max(80, round(target * 0.20))
    rows = fill_to_target(selected, pools, target=target, category_cap=category_cap)
    rows = deduplicate(rows)
    if len(rows) < target:
        source_sizes = {source: len(values) for source, values in pools.items()}
        raise CatalogBuildError(
            f"公开测试数据不足：目标 {target}，实际 {len(rows)}，补充池 {source_sizes}"
        )
    rows = rows[:target]
    rows.sort(key=lambda row: (row.category_key, str(row.attributes.get("catalog_source")), row.title, row.item_id))
    summary = write_snapshot(
        rows,
        output,
        target=target,
        retrieved_at=retrieved_at,
        source_caps=source_caps,
    )
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a balanced, multi-source public-demo catalog for ShopPilot."
    )
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    target = max(100, min(args.target, 1500))
    rows, summary = build(target=target, output=args.output)
    print(json.dumps({"count": len(rows), **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
