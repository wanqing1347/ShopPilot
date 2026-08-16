from __future__ import annotations

import json

from app.recall.catalog import dataset_root
from app.recall.evaluation import load_evaluation_queries


def test_sigir_behavior_snapshot_is_wired_into_offline_catalog() -> None:
    root = dataset_root()
    summary = json.loads(
        (root / "sigir_behavior_summary.json").read_text(encoding="utf-8")
    )

    assert summary["dataset_name"] == "shoppilot_sigir_behavior_simulation"
    assert summary["behavior_origin"] == "sigir_simulation"
    assert summary["output_counts"]["queries"] > 0
    assert summary["output_counts"]["interactions"] > 0
    assert summary["output_counts"]["users"] > 0

    queries = load_evaluation_queries(root / "queries.jsonl")
    assert len(queries) == summary["output_counts"]["queries"]
    assert {query.split for query in queries} == {"train", "dev", "test"}
    assert all(query.relevant_group_ids for query in queries)


def test_sigir_interactions_are_explicitly_simulated() -> None:
    root = dataset_root()
    with (root / "interactions.jsonl").open(encoding="utf-8") as handle:
        first = json.loads(next(handle))

    assert first["event_origin"] == "sigir_simulation"
    assert first["platform"] == "public_demo"
    assert first["same_group_id"]
