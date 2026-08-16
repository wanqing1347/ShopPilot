"""Refresh only LTR behavior priors from the normalized SIGIR events.

The packaged model is trained with the project's BGE-compatible embedding
configuration. When BGE is unavailable locally, this script preserves the
packaged feature weights and refreshes the group-level interaction priors from
the new behavior snapshot. It does not relabel a hashing model as BGE.

Usage:
    git show HEAD:app/recall/artifacts/ltr-v1.json |
      .venv312/Scripts/python.exe scripts/refresh_ltr_behavior_priors.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.recall.catalog import dataset_root
from app.utils.runtime import PROJECT_ROOT


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_priors(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group_id = str(row.get("same_group_id") or "")
        event_type = str(row.get("event_type") or "")
        if group_id and event_type:
            counts[group_id][event_type] += 1
    priors: dict[str, list[float]] = {}
    for group_id, events in counts.items():
        impressions = max(1, events["impression"])
        priors[group_id] = [
            (events["click"] + 1.0) / (impressions + 4.0),
            (events["favorite"] + 0.5) / (impressions + 6.0),
            (events["purchase"] + 0.25) / (impressions + 8.0),
            (events["dislike"] + 0.25) / (impressions + 8.0),
        ]
    return priors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "app" / "recall" / "artifacts" / "ltr-v1.json")
    args = parser.parse_args()

    base = json.load(sys.stdin)
    interactions_path = dataset_root() / "interactions.jsonl"
    queries_path = dataset_root() / "queries.jsonl"
    interactions = load_jsonl(interactions_path)
    queries = load_jsonl(queries_path)
    base["group_priors"] = build_priors(interactions)
    training = dict(base.get("training") or {})
    training.update(
        {
            "behavior_source": "SIGIR-ecom-data-challenge",
            "behavior_origin": "sigir_simulation",
            "behavior_query_count": len(queries),
            "behavior_event_count": len(interactions),
            "behavior_prior_group_count": len(base["group_priors"]),
            "behavior_refreshed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    base["training"] = training
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "embedding_provider": base.get("embedding_provider"),
        "embedding_model": base.get("embedding_model"),
        "behavior_query_count": len(queries),
        "behavior_event_count": len(interactions),
        "behavior_prior_group_count": len(base["group_priors"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
