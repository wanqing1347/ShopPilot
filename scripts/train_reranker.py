from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.recall.hybrid import clear_retriever_cache, get_hybrid_retriever
from app.recall.ltr import clear_ltr_cache, reranker_model_path
from app.recall.ltr_training import train_and_select_reranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 train/dev Query 和行为记录训练 pairwise LTR reranker。"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=reranker_model_path(),
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[0.03, 0.1, 0.3, 1.0, 3.0],
    )
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--negatives-per-positive", type=int, default=16)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/reranker-training-report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clear_retriever_cache()
    clear_ltr_cache()
    retriever = get_hybrid_retriever()
    report = train_and_select_reranker(
        retriever,
        output_path=args.output.resolve(),
        c_values=tuple(args.c_values),
        candidate_pool=max(10, args.candidate_pool),
        negatives_per_positive=max(1, args.negatives_per_positive),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={args.report.resolve()}")


if __name__ == "__main__":
    main()
