"""Convert a bounded SIGIR e-commerce sample into ShopPilot behavior JSONL.

SIGIR contains anonymous sessions, hashed SKUs, browsing events and search
result/click lists. It does not expose natural-language queries or a product
identity shared with ShopPilot, so this importer creates an explicitly
simulated join to the current catalog's same_group_id values. It never claims
the generated events are Amazon, Walmart or eBay user behavior.

The raw SIGIR files stay outside the repository. The default limits are
intentionally bounded because the challenge files are multi-gigabyte.

Usage:
    python scripts/import_sigir_behavior.py
    python scripts/import_sigir_behavior.py --max-search-rows 50000 --max-browsing-rows 100000
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.recall.catalog import load_catalog
from app.utils.runtime import PROJECT_ROOT


DEFAULT_SIGIR_TRAIN = (
    PROJECT_ROOT.parent / "data-SIGIR" / "SIGIR-ecom-data-challenge" / "train"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "offline_catalog"
DEFAULT_MAX_SEARCH_ROWS = 20_000
DEFAULT_MAX_BROWSING_ROWS = 50_000


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    return [str(item) for item in parsed if item]


def split_for_session(session_id: str) -> str:
    bucket = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "dev"
    return "test"


def event_type_for_browsing(row: dict[str, str]) -> str:
    action = " ".join(
        value.strip().lower()
        for value in (row.get("event_type", ""), row.get("product_action", ""))
        if value
    )
    if "purchase" in action or "order" in action:
        return "purchase"
    if "cart" in action or "favorite" in action or "wishlist" in action:
        return "favorite"
    if "detail" in action or "click" in action or "product" in action:
        return "click"
    return "impression"


def iso_timestamp(epoch_ms: str | None) -> str | None:
    if not epoch_ms:
        return None
    try:
        timestamp = int(float(epoch_ms)) / 1000
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def read_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def stable_group_for_sku(sku_hash: str, group_ids: list[str]) -> str:
    if not group_ids:
        raise ValueError("当前商品目录没有可用的 same_group_id")
    digest = hashlib.sha256(sku_hash.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % len(group_ids)
    return group_ids[index]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def convert(
    sigir_train_dir: Path,
    output_dir: Path,
    *,
    max_search_rows: int = DEFAULT_MAX_SEARCH_ROWS,
    max_browsing_rows: int = DEFAULT_MAX_BROWSING_ROWS,
) -> dict[str, Any]:
    if max_search_rows <= 0 or max_browsing_rows <= 0:
        raise ValueError("max_search_rows 和 max_browsing_rows 必须大于 0")

    products = load_catalog()
    group_meta: dict[str, dict[str, str]] = {}
    for product in products:
        group_meta.setdefault(
            product.same_group_id,
            {
                "category_key": product.category_key,
                "category": str(product.attributes.get("category") or product.category_key),
            },
        )
    group_ids = sorted(group_meta)
    if not group_ids:
        raise ValueError("当前离线商品目录没有可用于行为映射的商品组")

    search_path = sigir_train_dir / "search_train.csv"
    browsing_path = sigir_train_dir / "browsing_train.csv"
    if not search_path.exists() or not browsing_path.exists():
        raise FileNotFoundError(f"SIGIR train 文件不完整: {sigir_train_dir}")

    queries: dict[str, dict[str, Any]] = {}
    session_query_ids: dict[str, str] = {}
    interactions: list[dict[str, Any]] = []
    sessions: set[str] = set()
    event_counts: Counter[str] = Counter()
    sku_cache: dict[str, str] = {}
    sequence = 0

    def group_for_sku(sku_hash: str) -> str:
        if sku_hash not in sku_cache:
            sku_cache[sku_hash] = stable_group_for_sku(sku_hash, group_ids)
        return sku_cache[sku_hash]

    def ensure_query(session_id: str, group_id: str, query_id: str | None = None) -> str:
        nonlocal sequence
        sessions.add(session_id)
        resolved_query_id = query_id or f"SIGIR-Q-{len(queries) + 1:07d}"
        if resolved_query_id not in queries:
            meta = group_meta[group_id]
            queries[resolved_query_id] = {
                "query_id": resolved_query_id,
                "query": f"{meta['category']} 商品搜索",
                "language": "zh-CN",
                "user_id": f"SIGIR-SESSION-{session_id[:16]}",
                "category_key": meta["category_key"],
                "constraints": {},
                "relevant_group_ids": [],
                "split": split_for_session(session_id),
                "label_origin": "sigir_simulated_click_behavior",
                "behavior_origin": "sigir_simulation",
            }
        session_query_ids.setdefault(session_id, resolved_query_id)
        return resolved_query_id

    def add_event(
        *,
        session_id: str,
        query_id: str,
        group_id: str,
        event_type: str,
        rank: int | None,
        timestamp: str | None,
    ) -> None:
        nonlocal sequence
        sequence += 1
        interactions.append(
            {
                "event_id": f"SIGIR-E-{sequence:09d}",
                "user_id": f"SIGIR-SESSION-{session_id[:16]}",
                "query_id": query_id,
                "item_id": f"SIGIR-SIM-{group_id}",
                "same_group_id": group_id,
                "platform": "public_demo",
                "event_type": event_type,
                "rank": rank,
                "timestamp": timestamp,
                "event_origin": "sigir_simulation",
            }
        )
        event_counts[event_type] += 1

    search_count = 0
    for search_count, row in enumerate(read_csv_rows(search_path), start=1):
        if search_count > max_search_rows:
            break
        session_id = str(row.get("session_id_hash") or "")
        if not session_id:
            continue
        product_skus = parse_list(row.get("product_skus_hash"))
        clicked_skus = parse_list(row.get("clicked_skus_hash"))
        all_skus = list(dict.fromkeys([*product_skus, *clicked_skus]))
        if not all_skus:
            continue
        first_group = group_for_sku(all_skus[0])
        query_id = ensure_query(session_id, first_group, f"SIGIR-Q-{search_count:07d}")
        clicked_groups = {group_for_sku(sku) for sku in clicked_skus}
        queries[query_id]["relevant_group_ids"] = sorted(clicked_groups)
        for rank, sku_hash in enumerate(product_skus, start=1):
            group_id = group_for_sku(sku_hash)
            add_event(
                session_id=session_id,
                query_id=query_id,
                group_id=group_id,
                event_type="impression",
                rank=rank,
                timestamp=iso_timestamp(row.get("server_timestamp_epoch_ms")),
            )
        for sku_hash in clicked_skus:
            group_id = group_for_sku(sku_hash)
            add_event(
                session_id=session_id,
                query_id=query_id,
                group_id=group_id,
                event_type="click",
                rank=(product_skus.index(sku_hash) + 1) if sku_hash in product_skus else None,
                timestamp=iso_timestamp(row.get("server_timestamp_epoch_ms")),
            )

    browsing_count = 0
    for browsing_count, row in enumerate(read_csv_rows(browsing_path), start=1):
        if browsing_count > max_browsing_rows:
            break
        session_id = str(row.get("session_id_hash") or "")
        sku_hash = str(row.get("product_sku_hash") or "")
        if not session_id or not sku_hash:
            continue
        group_id = group_for_sku(sku_hash)
        query_id = session_query_ids.get(session_id)
        query_id = ensure_query(session_id, group_id, query_id)
        event_type = event_type_for_browsing(row)
        if event_type in {"click", "favorite", "purchase"}:
            queries[query_id]["relevant_group_ids"] = sorted(
                set(queries[query_id]["relevant_group_ids"]) | {group_id}
            )
        add_event(
            session_id=session_id,
            query_id=query_id,
            group_id=group_id,
            event_type=event_type,
            rank=None,
            timestamp=iso_timestamp(row.get("server_timestamp_epoch_ms")),
        )

    usable_query_ids = {
        query_id for query_id, query in queries.items() if query["relevant_group_ids"]
    }
    query_rows = [queries[query_id] for query_id in queries if query_id in usable_query_ids]
    interaction_rows = [row for row in interactions if row["query_id"] in usable_query_ids]
    output_event_counts = Counter(str(row["event_type"]) for row in interaction_rows)
    users = [
        {
            "user_id": f"SIGIR-SESSION-{session_id[:16]}",
            "profile_origin": "sigir_simulation",
            "behavior_origin": "sigir_simulation",
        }
        for session_id in sorted(sessions)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    query_count = _write_jsonl(output_dir / "queries.jsonl", query_rows)
    interaction_count = _write_jsonl(output_dir / "interactions.jsonl", interaction_rows)
    user_count = _write_jsonl(output_dir / "users.jsonl", users)
    summary = {
        "dataset_name": "shoppilot_sigir_behavior_simulation",
        "source_dataset": "SIGIR-ecom-data-challenge",
        "source_policy": "non-commercial research and education only; do not redistribute raw data",
        "behavior_origin": "sigir_simulation",
        "mapping_strategy": "deterministic anonymous SIGIR SKU hash to current catalog same_group_id",
        "query_text_strategy": "category label template because SIGIR search queries are vectors, not natural-language text",
        "max_search_rows": max_search_rows,
        "max_browsing_rows": max_browsing_rows,
        "source_rows_read": {"search": min(search_count, max_search_rows), "browsing": min(browsing_count, max_browsing_rows)},
        "output_counts": {
            "queries": query_count,
            "interactions": interaction_count,
            "users": user_count,
            "events_by_type": dict(sorted(output_event_counts.items())),
        },
        "platform_note": "platform=public_demo because SIGIR is not an Amazon/Walmart/eBay source",
    }
    (output_dir / "sigir_behavior_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigir-train-dir", type=Path, default=DEFAULT_SIGIR_TRAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-search-rows", type=int, default=DEFAULT_MAX_SEARCH_ROWS)
    parser.add_argument("--max-browsing-rows", type=int, default=DEFAULT_MAX_BROWSING_ROWS)
    args = parser.parse_args()
    summary = convert(
        args.sigir_train_dir,
        args.output_dir,
        max_search_rows=args.max_search_rows,
        max_browsing_rows=args.max_browsing_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
