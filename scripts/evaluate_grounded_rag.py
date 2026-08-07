from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.knowledge.grounded_evaluation import evaluate_grounded_rag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live grounded CategoryInsight RAG evaluation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "grounded-rag-evaluation.json",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    os.environ["SHOPPILOT_KNOWLEDGE_SYNTHESIS_ENABLED"] = "true"
    report = await evaluate_grounded_rag()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nreport={args.output.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
