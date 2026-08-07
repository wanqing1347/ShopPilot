from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from app.recall.catalog import dataset_root
from app.recall.evaluation import evaluate_retriever, load_evaluation_queries
from app.recall.hybrid import clear_retriever_cache, get_hybrid_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="仅使用 dev split 调整 BM25/Vector RRF 和规则重排权重。"
    )
    parser.add_argument("--bm25", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--vector", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--rerank", type=float, nargs="+", default=[0.0, 0.1, 0.15, 0.25])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/retrieval-tuning-dev.json"),
    )
    return parser.parse_args()


def objective(metrics: dict[str, Any]) -> float:
    return (
        metrics["ndcg"]["@10"] * 0.60
        + metrics["recall"]["@10"] * 0.30
        + metrics["hard_constraint_satisfaction"]["@10"] * 0.10
    )


def main() -> None:
    args = parse_args()
    clear_retriever_cache()
    retriever = get_hybrid_retriever()
    queries = load_evaluation_queries(dataset_root() / "queries.jsonl", split="dev")
    original = {
        name: os.environ.get(name)
        for name in (
            "SHOPPILOT_RETRIEVAL_BM25_WEIGHT",
            "SHOPPILOT_RETRIEVAL_VECTOR_WEIGHT",
            "SHOPPILOT_RETRIEVAL_RERANK_WEIGHT",
        )
    }
    trials: list[dict[str, Any]] = []
    try:
        for bm25_weight in args.bm25:
            for vector_weight in args.vector:
                for rerank_weight in args.rerank:
                    os.environ["SHOPPILOT_RETRIEVAL_BM25_WEIGHT"] = str(bm25_weight)
                    os.environ["SHOPPILOT_RETRIEVAL_VECTOR_WEIGHT"] = str(vector_weight)
                    os.environ["SHOPPILOT_RETRIEVAL_RERANK_WEIGHT"] = str(rerank_weight)
                    report = evaluate_retriever(
                        retriever,
                        queries,
                        k_values=(5, 10, 20),
                        variants={"hybrid_rules": ("hybrid", "rules", None)},
                    )
                    metrics = report["modes"]["hybrid_rules"]
                    trials.append(
                        {
                            "bm25_weight": bm25_weight,
                            "vector_weight": vector_weight,
                            "rerank_weight": rerank_weight,
                            "objective": round(objective(metrics), 8),
                            "metrics": metrics,
                        }
                    )
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    trials.sort(key=lambda row: row["objective"], reverse=True)
    output = {
        "selection_split": "dev",
        "test_split_used": False,
        "embedding_provider": retriever.provider.name,
        "embedding_model": getattr(retriever.provider, "model_name", None),
        "best": trials[0],
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["best"], ensure_ascii=False, indent=2))
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
