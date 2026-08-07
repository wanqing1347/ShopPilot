from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.knowledge.catalog import load_knowledge_documents
from app.knowledge.evaluation import (
    build_knowledge_evaluation_cases,
    evaluate_knowledge_retriever,
)
from app.knowledge.retriever import get_category_knowledge_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate category knowledge retrieval")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "knowledge-retrieval-evaluation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = list(load_knowledge_documents())
    cases = build_knowledge_evaluation_cases(documents)
    report = evaluate_knowledge_retriever(
        get_category_knowledge_retriever(),
        cases,
        k_values=tuple(sorted(set(max(1, value) for value in args.k))),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nreport={args.output.resolve()}")


if __name__ == "__main__":
    main()
