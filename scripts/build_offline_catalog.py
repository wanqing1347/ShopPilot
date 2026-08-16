"""Merge normalized marketplace snapshots into one offline catalog.

The source directories are intentionally left untouched. Live marketplace
snapshots are relabeled as cached offline observations so callers cannot
mistake them for current inventory.

Usage:
    python scripts/build_offline_catalog.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models import Candidate
from app.utils.runtime import PROJECT_ROOT


DEFAULT_OPEN = PROJECT_ROOT / "data" / "open_datasets" / "products.jsonl"
DEFAULT_LIVE = PROJECT_ROOT / "data" / "live_catalog" / "products.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "offline_catalog" / "products.jsonl"

# Keep offline fallback partitions aligned with the live provider routes.
OFFLINE_PLATFORM_MAP = {
    "amazon": "amazon",
    "amazon_jp": "amazon",
    "walmart": "walmart",
    "lazada": "walmart",
    "shein": "walmart",
    "rakuten": "ebay",
    "shopee": "ebay",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_candidates(path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                source_platform = str(payload.get("platform") or "")
                payload["platform"] = OFFLINE_PLATFORM_MAP.get(source_platform, source_platform)
                candidates.append(Candidate.model_validate(payload))
            except Exception as exc:
                raise ValueError(f"无法解析 {path}:{line_number}: {exc}") from exc
    return candidates


def offline_copy(candidate: Candidate, source_kind: str) -> Candidate:
    attributes = dict(candidate.attributes)
    source_platform = candidate.platform
    target_platform = OFFLINE_PLATFORM_MAP.get(source_platform, source_platform)
    attributes.update(
        {
            "offline_snapshot": True,
        }
    )
    return Candidate.model_validate(
        {
            **candidate.model_dump(mode="json"),
            "platform": target_platform,
            "attributes": attributes,
            "data_origin": "offline_snapshot",
            "provider": "offline_snapshot",
            "verification_status": "cached",
            "retrieved_at": candidate.retrieved_at or candidate.ingested_at,
        }
    )


def dedupe_key(candidate: Candidate) -> tuple[str, str]:
    # Search-result snapshots often reuse the search page URL for every row;
    # item_id is the stable product identity in the normalized schema.
    return candidate.platform, candidate.item_id


def build(open_path: Path, live_path: Path, output_path: Path) -> dict[str, Any]:
    sources: list[tuple[str, Path]] = []
    if open_path.exists():
        sources.append(("open_dataset", open_path))
    if live_path.exists():
        sources.append(("live_snapshot", live_path))
    if not sources:
        raise FileNotFoundError("没有找到可用的 products.jsonl 输入文件")

    merged: dict[tuple[str, str], Candidate] = {}
    source_counts: Counter[str] = Counter()
    for source_kind, path in sources:
        for candidate in read_candidates(path):
            normalized = offline_copy(candidate, source_kind)
            key = dedupe_key(normalized)
            # Keep the open dataset row when the same product ID appears in
            # both snapshots; it has the more stable public-dataset identity.
            if key not in merged or source_kind == "open_dataset":
                merged[key] = normalized
            source_counts[source_kind] += 1

    rows = sorted(
        merged.values(),
        key=lambda candidate: (candidate.platform, candidate.item_id),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in rows:
            handle.write(candidate.model_dump_json() + "\n")

    output_platform_counts = Counter(candidate.platform for candidate in rows)
    summary = {
        "dataset_name": "shoppilot_offline_catalog",
        "schema_version": 2,
        "generated_at": utc_now(),
        "purpose": "offline cached marketplace snapshots; not current inventory",
        "input_counts": dict(source_counts),
        "output_counts": {"offline_snapshot": len(rows)},
        "total": len(rows),
        "platform_counts": dict(sorted(output_platform_counts.items())),
        "verification_status": "cached",
        "output": str(output_path),
    }
    (output_path.parent / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", dest="open_path", type=Path, default=DEFAULT_OPEN)
    parser.add_argument("--live", dest="live_path", type=Path, default=DEFAULT_LIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build(args.open_path, args.live_path, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
