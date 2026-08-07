from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.recall.catalog import dataset_root
from app.recall.evaluation import evaluate_retriever, load_evaluation_queries
from app.recall.hybrid import get_hybrid_retriever
from app.recall.ltr import load_compatible_reranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ShopPilot BM25/vector/hybrid retrieval")
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="test")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "retrieval-evaluation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = None if args.split == "all" else args.split
    queries = load_evaluation_queries(
        dataset_root() / "queries.jsonl",
        split=split,
    )
    retriever = get_hybrid_retriever()
    learned_model, learned_error = load_compatible_reranker(
        provider_name=retriever.provider.name,
        model_name=getattr(retriever.provider, "model_name", None),
    )
    variants = {
        "lexical_rules": ("lexical", "rules", None),
        "vector_rules": ("vector", "rules", None),
        "hybrid_rules": ("hybrid", "rules", None),
    }
    if learned_model is not None:
        variants["hybrid_ltr"] = ("hybrid", "learned", learned_model)
    report = evaluate_retriever(
        retriever,
        queries,
        k_values=tuple(sorted(set(max(1, value) for value in args.k))),
        variants=variants,
    )
    report["learned_reranker_available"] = learned_model is not None
    if learned_error:
        report["learned_reranker_error"] = learned_error
    report["requested_split"] = args.split
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nreport={args.output.resolve()}")


if __name__ == "__main__":
    main()
