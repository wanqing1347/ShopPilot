from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from app.models import Candidate
from app.utils.currency import to_base
from app.utils.runtime import PROJECT_ROOT

BASE_URL = "https://www.web-scraping.dev"
ROBOTS_URL = f"{BASE_URL}/robots.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "public_demo" / "products.jsonl"
USER_AGENT = "ShopPilotResearchBot/1.0 (low-volume educational mock-catalog importer)"


class PublicDemoScrapeError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _robots_policy(text: str) -> tuple[list[str], float]:
    disallowed: list[str] = []
    crawl_delay = 2.0
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
            try:
                crawl_delay = max(0.0, float(value))
            except ValueError:
                pass
    return disallowed, crawl_delay


def _assert_allowed(url: str, disallowed: list[str]) -> None:
    path = urlparse(url).path or "/"
    for prefix in disallowed:
        if prefix != "/" and path.startswith(prefix):
            raise PublicDemoScrapeError(
                f"robots.txt 不允许访问该路径: {path} (Disallow: {prefix})"
            )
        if prefix == "/":
            raise PublicDemoScrapeError("robots.txt 禁止访问整个站点")


def _product_urls(html: str) -> list[str]:
    urls = re.findall(
        r'<h3[^>]*>\s*<a\s+href=["\']([^"\']+/product/\d+)["\']',
        html,
        flags=re.IGNORECASE,
    )
    result: list[str] = []
    for value in urls:
        url = urljoin(BASE_URL, value)
        if url not in result:
            result.append(url)
    return result


def _product_json_ld(html: str) -> dict[str, object]:
    blocks = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in blocks:
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        values = payload if isinstance(payload, list) else [payload]
        for value in values:
            if isinstance(value, dict) and value.get("@type") == "Product":
                return value
    raise PublicDemoScrapeError("商品页缺少可解析的 Product JSON-LD")


def _category(title: str) -> tuple[str, str]:
    normalized = title.lower()
    if re.search(r"\b(shoe|shoes|boot|boots|sandal|sandals|sneaker|sneakers)\b", normalized):
        return "demo_footwear", "Footwear"
    if re.search(r"\b(beanie|hat|shirt|dress|jacket|apparel)\b", normalized):
        return "demo_apparel", "Apparel"
    return "demo_consumables", "Consumables"


def _as_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _candidate(payload: dict[str, object], source_url: str, retrieved_at: str) -> Candidate:
    title = str(payload.get("name") or "").strip()
    offers = payload.get("offers") if isinstance(payload.get("offers"), dict) else {}
    price = _as_float(offers.get("lowPrice") or offers.get("price"))
    currency = str(offers.get("priceCurrency") or "USD").strip().upper()
    if not title or price is None:
        raise PublicDemoScrapeError(f"商品缺少标题或价格: {source_url}")

    rating_payload = (
        payload.get("aggregateRating")
        if isinstance(payload.get("aggregateRating"), dict)
        else {}
    )
    image = payload.get("image")
    if isinstance(image, list):
        image = image[0] if image else None
    brand_payload = payload.get("brand")
    if isinstance(brand_payload, dict):
        brand = str(brand_payload.get("name") or "").strip() or None
    else:
        brand = str(brand_payload or "").strip() or None
    product_id = urlparse(source_url).path.rstrip("/").split("/")[-1]
    category_key, category_name = _category(title)
    price_cny = round(to_base(price, currency), 2)
    availability = str(offers.get("availability") or "")

    return Candidate(
        item_id=f"public-demo:{product_id}",
        same_group_id=f"PUBLIC-DEMO:{product_id}",
        platform="public_demo",
        title=title,
        title_en=title,
        description="Public mock-commerce product used for catalog integration testing.",
        brand=brand,
        category_key=category_key,
        category_path=["Public mock-commerce catalog", category_name],
        price=price,
        currency=currency,
        price_cny=price_cny,
        shipping_cny=0.0,
        landed_price_cny=price_cny,
        rating=_as_float(rating_payload.get("ratingValue")),
        review_count=_as_int(rating_payload.get("reviewCount")),
        is_available="OutOfStock" not in availability,
        image_url=str(image).strip() if image else None,
        attributes={
            "category": category_name,
            "category_key": category_key,
            "category_path": ["Public mock-commerce catalog", category_name],
            "catalog_source": "web-scraping.dev",
            "mock_commerce_data": True,
            "source_description_length": len(str(payload.get("description") or "")),
        },
        ingested_at=retrieved_at,
        quality_grade="B",
        data_origin="public_demo_catalog",
        provider="web_scraping_dev_snapshot",
        source_url=source_url,
        retrieved_at=retrieved_at,
        verification_status="public_demo",
    )


def scrape(*, limit: int, pages: int, output: Path) -> list[Candidate]:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    with httpx.Client(headers=headers, timeout=20.0, follow_redirects=True) as client:
        robots_response = client.get(ROBOTS_URL)
        robots_response.raise_for_status()
        disallowed, crawl_delay = _robots_policy(robots_response.text)

        discovered: list[str] = []
        last_request_at = 0.0

        def get(url: str) -> httpx.Response:
            nonlocal last_request_at
            _assert_allowed(url, disallowed)
            wait = crawl_delay - (time.monotonic() - last_request_at)
            if wait > 0:
                time.sleep(wait)
            response = client.get(url)
            last_request_at = time.monotonic()
            response.raise_for_status()
            return response

        for page in range(1, max(1, pages) + 1):
            list_url = f"{BASE_URL}/products?page={page}"
            for url in _product_urls(get(list_url).text):
                if url not in discovered:
                    discovered.append(url)
                if len(discovered) >= limit:
                    break
            if len(discovered) >= limit:
                break

        rows: list[Candidate] = []
        retrieved_at = _utc_now()
        for url in discovered[:limit]:
            payload = _product_json_ld(get(url).text)
            rows.append(_candidate(payload, url, retrieved_at))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(row.model_dump_json() for row in rows) + "\n",
        encoding="utf-8",
    )
    summary = {
        "source": BASE_URL,
        "source_type": "public_mock_commerce_catalog",
        "retrieved_at": retrieved_at,
        "count": len(rows),
        "robots_respected": True,
        "request_delay_seconds": crawl_delay,
        "output": str(output),
        "notice": "Mock commerce data; not live marketplace offers.",
    }
    output.with_name("snapshot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import a small public mock-commerce product snapshot from web-scraping.dev."
    )
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--pages", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    limit = max(1, min(args.limit, 25))
    rows = scrape(limit=limit, pages=max(1, args.pages), output=args.output)
    print(
        json.dumps(
            {
                "count": len(rows),
                "output": str(args.output),
                "platform": "public_demo",
                "provider": "web_scraping_dev_snapshot",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
