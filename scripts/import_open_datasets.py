"""Import open-source e-commerce product datasets into Candidate schema v2 JSONL.

Source: https://github.com/luminati-io/eCommerce-dataset-samples
Real marketplace listings (1000 rows per platform, captured by a public scraper
project). Prices are kept in the listing's original currency — no FX conversion.

Usage:
    python scripts/import_open_datasets.py \
        --input data/open_datasets \
        --output data/open_datasets/products.jsonl
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.catalog.public_demo_taxonomy import (
    CATEGORY_ALIASES,
    CATEGORY_LABEL_ZH,
    infer_category_key,
)
from app.models import Candidate
from app.utils.runtime import PROJECT_ROOT

DEFAULT_INPUT = PROJECT_ROOT / "data" / "open_datasets"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "open_datasets" / "products.jsonl"

# per-source: (csv file, platform name, title field, price field, currency field,
#             rating field, reviews field, brand field, image field, url field,
#             category fields: csv column names carrying the real category path)
SOURCE_SPECS: list[dict[str, object]] = [
    {
        "file": "amazon-products.csv",
        "platform": "amazon",
        "title": "title",
        "price": "final_price",
        "price_alt": "initial_price",
        "currency": "currency",
        "rating": "rating",
        "reviews": "reviews_count",
        "brand": "brand",
        "image": "image_url",
        "url": "url",
        "categories": ["categories"],
    },
    {
        "file": "lazada-products.csv",
        "platform": "lazada",
        "title": "title",
        "price": "final_price",
        "price_alt": "initial_price",
        "currency": "currency",
        "rating": "rating",
        "reviews": "reviews",
        "brand": "seller_name",
        "image": "image",
        "url": "url",
        "categories": ["breadcrumb"],
    },
    {
        "file": "shein-products.csv",
        "platform": "shein",
        "title": "product_name",
        "price": "final_price",
        "price_alt": "initial_price",
        "currency": "currency",
        "rating": "rating",
        "reviews": "reviews_count",
        "brand": "brand",
        "image": "main_image",
        "url": "url",
        "categories": ["category_tree"],
    },
    {
        "file": "shopee-products.csv",
        "platform": "shopee",
        "title": "title",
        "price": "final_price",
        "price_alt": "initial_price",
        "currency": "currency",
        "rating": "rating",
        "reviews": "reviews",
        "brand": "seller_name",
        "image": "image",
        "url": "url",
        "categories": ["breadcrumb"],
    },
    {
        "file": "walmart-products.csv",
        "platform": "walmart",
        "title": "product_name",
        "price": "final_price",
        "price_alt": "initial_price",
        "currency": "currency",
        "rating": "rating",
        "reviews": "review_count",
        "brand": "brand",
        "image": "main_image",
        "url": "url",
        "categories": ["category_name", "category_path"],
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clean_text(value: Any, limit: int = 300) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def parse_price(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in {"null", "none", "-"}:
        return None
    # scientific notation (e.g. walmart 2.29e+01)
    try:
        return float(text)
    except ValueError:
        pass
    # strip currency symbols/commas: "$1,234.56" / "57.79"
    cleaned = re.sub(r"[^\d.,-]", "", text)
    cleaned = cleaned.replace(",", "")
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_rating(value: Any) -> float | None:
    text = clean_text(value, limit=40)
    if not text or text in {"0", "-", "null"}:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        rating = float(match.group(1))
        return rating if rating > 0 else None
    return None


def parse_reviews(value: Any) -> int | None:
    text = clean_text(value, limit=40)
    if not text or text in {"0", "-", "null", "N/A"}:
        return None
    match = re.search(r"([0-9][0-9.,]*\s*[KkMm]?|[0-9][0-9.,]*)", text)
    if not match:
        return None
    raw = match.group(1).replace(",", "").strip()
    try:
        if raw.endswith(("K", "k")):
            return int(float(raw[:-1]) * 1000)
        if raw.endswith(("M", "m")):
            return int(float(raw[:-1]) * 1_000_000)
        return int(float(raw))
    except ValueError:
        return None


def extract_url(value: Any) -> str | None:
    text = str(value or "").strip().strip('"').strip("'")
    if not text or text == "[]":
        return None
    # some image/url cells are JSON arrays like ["https://..."]
    match = re.search(r"https?://[^\"'\s]+", text)
    return match.group(0)[:500] if match else None


def extract_categories(row: dict[str, Any], spec: dict[str, object]) -> tuple[str, list[str]]:
    """Parse real category fields from the row. Returns (joined_text, raw_paths)."""
    field_names = spec.get("categories", [])
    assert isinstance(field_names, list)
    parts: list[str] = []
    raw_paths: list[str] = []
    for field in field_names:
        raw = row.get(field)  # type: ignore[arg-type]
        if not raw:
            continue
        text = str(raw).strip()
        if not text or text in {"[]", "null"}:
            continue
        raw_paths.append(text[:300])
        parsed: Any = None
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("name") or item.get("label") or ""))
        else:
            # plain string like "Eye Shadow Stick" or "/cp/eye-shadow-stick/7896251"
            parts.append(text)
    joined = " ".join(part for part in parts if part)
    return joined, raw_paths


# Minimal multilingual synonyms for marketplace category paths (es/id/ja), so
# real listings from Shopee MX / Lazada ID / Amazon JP get useful labels too.
_EXTRA_ALIASES: dict[str, tuple[str, ...]] = {
    "apparel": (
        "ropa",
        "moda",
        "camisa",
        "camisas",
        "blusa",
        "tops",
        "vestido",
        "pants",
        "jeans",
        "shirt",
        "t-shirt",
        "hoodie",
        "sweater",
        "dress",
        "fashion clothing",
        "ropa de mujer",
        "ropa de hombre",
        "niños",
        "kids",
        "swimsuit",
        "swimsuits",
        "bra",
        "bras",
        "underwear",
        "pajama",
        "pajamas",
        "sleepwear",
        "lingerie",
        "เสื้อผ้า",
        "เสื้อ",
        "ชุด",
    ),
    "footwear": ("zapatos", "zapatillas", "calzado", "sneaker", "sneakers", "botas", "sandal", "shoe", "shoes", "boots", "boot", "pumps", "heels", "heel"),
    "bags": ("bolso", "bolsos", "mochila", "handbag", "backpack", "bag", "bags", "wallet", "卡包", "バッグ"),
    "beauty": ("beauty", "cosmetic", "cosmetics", "makeup", "maquillaje", "lipstick", "eye shadow", "eyeshadow", "mascara", "护肤", "美容"),
    "skincare": ("skincare", "skin care", "serum", "moisturizer", "crema", "pelembap", "pembersih", "perawatan kulit", "护肤", "化粧水", "スキンケア"),
    "fragrances": ("perfume", "cologne", "fragancia", "香水", "フレグランス"),
    "electronics": (
        "electronics",
        "electronic",
        "televisi",
        "tv",
        "television",
        "televisión",
        "camera",
        "cámaras",
        "audio",
        "video",
        "computer",
        "computers",
        "laptop",
        "notebook",
        "tablet",
        "smartphone",
        "monitor",
        "smart tv",
        "led tv",
        "4k tv",
        "手机",
        "数码",
        "電気",
        "家電",
    ),
    "mobile_accessories": ("charger", "cable", "power bank", "case for", "phone case", "screen protector", "手机壳", "充电器"),
    "headphones": ("headphone", "headphones", "earbud", "earbuds", "auriculares", "audífonos", "イヤホン", "耳机", "ヘッドホン"),
    "keyboard": ("keyboard", "teclado", "キーボード", "键盘"),
    "laptops": ("laptop", "laptops", "notebook computer", "ノートパソコン", "笔记本电脑"),
    "tablets": ("tablet", "ipad", "tableta", "タブレット", "平板"),
    "smartphones": ("smartphone", "smartphones", "iphone", "celular", "teléfono", "telefono", "móvil", "movil", "手机", "スマホ", "智能手机"),
    "watches": ("watch", "watches", "reloj", "腕時計", "手表"),
    "sunglasses": ("sunglasses", "sun glasses", "gafas", "lentes de sol", "太阳镜", "サングラス"),
    "jewellery": ("jewelry", "jewellery", "joyería", "necklace", "ring", "earring", "anillo", "collar", "aretes", "珠宝", "首饰", "アクセサリー"),
    "groceries": ("grocery", "groceries", "food", "beverage", "snack", "alimentos", "bebida", "食品", "飲料", "食品飲料"),
    "furniture": ("furniture", "mueble", "muebles", "sofa", "cama", "mesa", "silla", "家具", "収納家具"),
    "home_decoration": ("home decor", "home decoration", "decoración", "decoration", "decor", "rug", "rugs", "curtain", "curtains", "sheet", "sheets", "bedding", "blanket", "blankets", "pillowcase", "pillowcases", "家居", "インテリア"),
    "kitchen_accessories": ("kitchen", "cookware", "cocina", "utensilios", "厨房", "キッチン"),
    "lighting": ("lamp", "lampar", "lámpara", "luz", "照明", "灯具", "flashlight", "led light"),
    "tools": ("tool", "tools", "hammer", "drill", "screwdriver", "wrench", "wrenches", "herramienta", "herramientas", "工具", "工具套装"),
    "sports_accessories": ("sports", "fitness", "yoga", "gym", "deporte", "ejercicio", "运动", "スポーツ"),
    "motorcycle": ("motorcycle", "motorbike", "helmet", "moto", "casco", "摩托车", "バイク"),
    "vehicles": ("vehicle", "vehicles", "car", "auto", "automotive", "automóvil", "汽车", "車"),
    "collectibles": ("toy", "toys", "collectible", "collectibles", "lego", "figurine", "juguete", "juguetes", "hobbies", "coleccionables", "modelo", "玩具", "ホビー", "おもちゃ"),
    "books": ("book", "books", "libro", "novel", "本", "書籍", "图书"),
    "coffee_cup": ("coffee", "mug", "taza", "马克杯", "咖啡", "マグカップ"),
    "thermos": ("thermos", "tumbler", "vacuum bottle", "botella", "保温杯", "水筒"),
    "travel_storage": ("travel", "luggage", "suitcase", "maleta", "旅行", "収納"),
}

# precompile lowercase aliases -> category key
_ALIAS_LOOKUP: list[tuple[str, str]] = sorted(
    (
        (alias.lower(), key)
        for key, aliases in _EXTRA_ALIASES.items()
        for alias in aliases
    ),
    key=lambda pair: len(pair[0]),
    reverse=True,
)
_ALIAS_LOOKUP += sorted(
    (
        (alias.lower(), key)
        for key, aliases in CATEGORY_ALIASES.items()
        for alias in aliases
    ),
    key=lambda pair: len(pair[0]),
    reverse=True,
)


def _match(alias: str, text: str) -> bool:
    """Word-boundary match; skips very short aliases to avoid false hits."""
    if len(alias) < 3:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text) is not None


def classify_by_text(category_text: str, title: str) -> str:
    """Best-effort category mapping from real category path text + title.

    Specificity order: product title (most specific) -> category leaf segments
    (later path entries are more specific) -> whole category text. Falls back
    to the project's title-rule taxonomy, then miscellaneous.
    """
    title_lower = title.lower()
    for alias, key in _ALIAS_LOOKUP:
        if _match(alias, title_lower):
            return key
    segments = [seg.strip() for seg in re.split(r"[>/\s]+", category_text.lower()) if seg.strip()]
    for segment in reversed(segments):
        for alias, key in _ALIAS_LOOKUP:
            if _match(alias, segment):
                return key
    haystack = f"{category_text} {title}".lower()
    for alias, key in _ALIAS_LOOKUP:
        if _match(alias, haystack):
            return key
    return infer_category_key(category_text, title)


# Context fixes for aliases that are ambiguous outside their own domain.
# e.g. "ring" in an impact-wrench title is a mechanical part, not jewellery;
# "bag" inside vacuum-cleaner parts is a dust bag, not a handbag.
_CONTEXT_FIXES: list[tuple[re.Pattern[str], set[str], str]] = [
    (
        re.compile(r"\b(wrench|wrenches|impact driver|socket set|drill|screwdriver)\b"),
        {"jewellery"},
        "tools",
    ),
    (
        re.compile(r"\b(vacuum|vacuum cleaner|hepa)\b"),
        {"bags"},
        "electronics",
    ),
    (
        re.compile(r"\b(air purifier|purifier|vacuum)\b"),
        {"home_decoration"},
        "electronics",
    ),
    # "collar" is Spanish for necklace, but English "collar" on clothing = neckline
    (
        re.compile(r"\b(tshirt|t-shirt|shirt|blouse|blusa|tops?|sweater|hoodie|dress|clothing)\b"),
        {"jewellery"},
        "apparel",
    ),
    # "cream" in food context is soup/cream-of, not skincare
    (
        re.compile(r"\b(soup|broth|condensed|food|beverage|drink|snack)\b"),
        {"skincare", "beauty"},
        "groceries",
    ),
    # "socks" are apparel, not footwear
    (
        re.compile(r"\b(socks?|calcetines|medias)\b"),
        {"footwear"},
        "apparel",
    ),
    # "coffee table" is a piece of furniture, not a coffee cup/mug
    (
        re.compile(r"\b(coffee|tea)\s+(table|tables|desk|desks)\b"),
        {"coffee_cup", "thermos"},
        "furniture",
    ),
    # whipped cream / topping / dessert is food, not skincare ("cream" alias)
    (
        re.compile(r"\b(whipped cream|cream topping|topping|dessert|yogurt|milkshake|whipping)\b"),
        {"skincare", "beauty", "fragrances"},
        "groceries",
    ),
]


def classify(category_text: str, title: str) -> str:
    """classify_by_text + contextual overrides for ambiguous aliases."""
    key = classify_by_text(category_text, title)
    text = f"{category_text} {title}".lower()
    for pattern, wrong_keys, replacement in _CONTEXT_FIXES:
        if key in wrong_keys and pattern.search(text):
            return replacement
    return key


def to_candidate(
    row: dict[str, Any],
    spec: dict[str, object],
    retrieved_at: str,
    seq: int = 0,
) -> Candidate | None:
    title = clean_text(row.get(spec["title"]))  # type: ignore[arg-type]
    if len(title) < 4:
        return None
    price = parse_price(row.get(spec["price"])) or parse_price(row.get(spec["price_alt"]))  # type: ignore[arg-type]
    if price is None or price <= 0:
        return None
    currency = clean_text(row.get(spec["currency"]), limit=8).upper() or "USD"  # type: ignore[arg-type]
    if currency not in {"USD", "IDR", "MXN", "JPY", "EUR", "GBP"}:
        currency = "USD"
    platform = str(spec["platform"])
    url = extract_url(row.get(spec["url"]))  # type: ignore[arg-type]
    image = extract_url(row.get(spec["image"]))  # type: ignore[arg-type]
    raw_id = re.sub(r"\W+", "-", title)[:60].lower()
    unique_id = f"{raw_id}-{seq:05d}" if seq else raw_id
    category_text, category_paths = extract_categories(row, spec)
    category_key = classify(category_text, title)
    category_zh = CATEGORY_LABEL_ZH.get(category_key, "其他商品")
    attributes: dict[str, object] = {
        "category": category_zh,
        "category_key": category_key,
        "category_path": ["Open dataset snapshot", category_zh],
        "dataset_source": "luminati-io/eCommerce-dataset-samples",
        "price_is_original_currency": True,
        "for_personal_research": True,
    }
    if category_text:
        attributes["source_category_text"] = category_text[:300]
    if category_paths:
        attributes["source_category_raw"] = category_paths
    return Candidate(
        item_id=f"{platform}:open:{unique_id}",
        same_group_id=f"OPEN:{platform.upper()}:{unique_id}",
        platform=platform,  # type: ignore[arg-type]
        title=title,
        title_en=title,
        description=clean_text(row.get("description") or row.get("product_description") or "", limit=500),  # type: ignore[arg-type]
        brand=clean_text(row.get(spec["brand"]), limit=100) or None,  # type: ignore[arg-type]
        category_key=category_key,
        category_path=["Open dataset snapshot", category_zh],
        price=round(price, 2),
        currency=currency,
        price_cny=None,
        shipping_cny=None,
        landed_price_cny=None,
        rating=parse_rating(row.get(spec["rating"])),  # type: ignore[arg-type]
        review_count=parse_reviews(row.get(spec["reviews"])),  # type: ignore[arg-type]
        is_available=True,
        image_url=image,
        attributes=attributes,
        ingested_at=retrieved_at,
        quality_grade="B",
        data_origin="open_dataset_snapshot",
        provider="luminati_ecommerce_samples",
        source_url=url,
        retrieved_at=retrieved_at,
        verification_status="cached",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    csv.field_size_limit(10_000_000)
    all_rows: list[Candidate] = []
    retrieved_at = utc_now()
    per_platform: dict[str, int] = {}
    for spec in SOURCE_SPECS:
        path = args.input / spec["file"]
        if not path.exists():
            print(f"SKIP (missing): {path.name}", flush=True)
            continue
        with path.open(encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            count = 0
            for row in reader:
                count += 1
                candidate = to_candidate(row, spec, retrieved_at, seq=count)
                if candidate is not None:
                    all_rows.append(candidate)
        per_platform[spec["platform"]] = count
        print(f"{spec['platform']}: {count} rows", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for candidate in all_rows:
            handle.write(candidate.model_dump_json() + "\n")

    summary = {
        "dataset_name": "shoppilot_open_datasets",
        "schema_version": 2,
        "compatible_project": "shoppilot-agent>=0.8.0",
        "generated_at": retrieved_at,
        "source": "https://github.com/luminati-io/eCommerce-dataset-samples",
        "purpose": "personal research snapshot of real marketplace listings",
        "total": len(all_rows),
        "per_platform": per_platform,
        "platform_counts": per_platform,
        "category_keys": len({r.category_key for r in all_rows}),
        "pricing": "original platform currency only (no FX conversion)",
        "output": str(args.output),
    }
    (args.output.parent / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
