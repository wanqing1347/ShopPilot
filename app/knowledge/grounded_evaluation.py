from __future__ import annotations

import os
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any

from app.tools.category_insight import category_insight


@dataclass(frozen=True)
class GroundedRagCase:
    case_id: str
    category: str
    category_key: str | None
    query: str
    expect_evidence: bool = True


DEFAULT_GROUNDED_RAG_CASES: tuple[GroundedRagCase, ...] = (
    GroundedRagCase(
        case_id="coffee-cup",
        category="咖啡杯",
        category_key="coffee_cup",
        query="预算300元，粗陶和玻璃咖啡杯怎么选，跨平台比价要注意什么？",
    ),
    GroundedRagCase(
        case_id="travel-storage",
        category="旅行收纳",
        category_key="travel_storage",
        query="旅行收纳有哪些常见材质，按到手价跨平台比较时应该注意什么？",
    ),
    GroundedRagCase(
        case_id="backpack",
        category="双肩包",
        category_key="backpack",
        query="双肩包选购时材质、预算和排序因素应该怎么权衡？",
    ),
    GroundedRagCase(
        case_id="keyboard",
        category="键盘",
        category_key="keyboard",
        query="键盘常见材质有哪些，当前合成商品的价格层级如何？",
    ),
    GroundedRagCase(
        case_id="headphones",
        category="耳机",
        category_key="headphones",
        query="耳机选购时应该先看哪些硬约束，材质和跨平台到手价怎么比较？",
    ),
    GroundedRagCase(
        case_id="thermos",
        category="保温杯",
        category_key="thermos",
        query="304、316不锈钢和其他材质如何选择，当前合成数据价格带如何？",
    ),
    GroundedRagCase(
        case_id="unknown-category",
        category="望远镜",
        category_key=None,
        query="望远镜应该怎么选？",
        expect_evidence=False,
    ),
)


async def evaluate_grounded_rag(
    cases: tuple[GroundedRagCase, ...] = DEFAULT_GROUNDED_RAG_CASES,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        output = await category_insight(
            case.category,
            query=case.query,
            category_key=case.category_key,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        citation_ids = {citation.doc_id for citation in output.citations}
        claim_ids = [citation_id for claim in output.grounded_claims for citation_id in claim.citation_ids]
        invalid_reference_count = sum(citation_id not in citation_ids for citation_id in claim_ids)
        claim_count = len(output.grounded_claims)
        claim_citation_coverage = (
            sum(bool(claim.citation_ids) for claim in output.grounded_claims) / claim_count
            if claim_count
            else 0.0
        )
        if case.expect_evidence:
            passed = (
                output.answer_mode == "llm_grounded"
                and claim_count > 0
                and invalid_reference_count == 0
                and output.citation_validation.get("valid") is True
            )
        else:
            passed = (
                output.answer_mode == "no_evidence"
                and not output.grounded_claims
                and not output.grounded_answer
                and not output.citations
            )
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "category_key": case.category_key,
                "expect_evidence": case.expect_evidence,
                "passed": passed,
                "answer_mode": output.answer_mode,
                "grounded_answer": output.grounded_answer,
                "claim_count": claim_count,
                "claim_citation_coverage": round(claim_citation_coverage, 4),
                "invalid_reference_count": invalid_reference_count,
                "invalid_generated_claim_count": int(
                    output.citation_validation.get("invalid_claim_count") or 0
                ),
                "numeric_grounding_rate": float(
                    output.citation_validation.get("numeric_grounding_rate") or 0.0
                ),
                "attempts": int(output.citation_validation.get("attempts") or 0),
                "duration_ms": duration_ms,
                "validation": output.citation_validation,
            }
        )

    known_rows = [row for row in rows if row["expect_evidence"]]
    unknown_rows = [row for row in rows if not row["expect_evidence"]]
    return {
        "model_name": os.getenv("LLM_MODEL_NAME") or os.getenv("LLM_MAIN") or "qwen-max",
        "case_count": len(rows),
        "known_case_count": len(known_rows),
        "unknown_case_count": len(unknown_rows),
        "grounded_success_rate": round(
            mean(row["passed"] for row in known_rows) if known_rows else 0.0,
            4,
        ),
        "no_evidence_refusal_rate": round(
            mean(row["passed"] for row in unknown_rows) if unknown_rows else 0.0,
            4,
        ),
        "claim_citation_coverage": round(
            mean(row["claim_citation_coverage"] for row in known_rows) if known_rows else 0.0,
            4,
        ),
        "invalid_reference_count": sum(row["invalid_reference_count"] for row in rows),
        "invalid_generated_claim_count": sum(
            row["invalid_generated_claim_count"] for row in known_rows
        ),
        "numeric_grounding_rate": round(
            mean(row["numeric_grounding_rate"] for row in known_rows) if known_rows else 0.0,
            4,
        ),
        "average_duration_ms": round(mean(row["duration_ms"] for row in rows), 2),
        "cases": rows,
    }
