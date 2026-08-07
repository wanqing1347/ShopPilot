from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from app.agent.llm import get_llm
from app.agent.settings import (
    knowledge_synthesis_max_attempts,
    knowledge_synthesis_max_claims,
    knowledge_synthesis_min_claims,
    knowledge_synthesis_min_token_overlap,
)
from app.models import CategoryInsightOutput, GroundedClaim
from app.recall.tokenizer import tokenize

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])")
_UNSUPPORTED_QUALIFIERS = (
    "大多数",
    "绝大多数",
    "多数",
    "全部",
    "所有",
    "唯一",
    "必然",
    "一定",
    "保证",
    "官方",
    "真实可追溯",
    "最常见",
    "占比最高",
)

_SYNTHESIS_PROMPT = """你是 ShopPilot 的品类知识 Grounded Synthesis 模块。

你只能使用输入 JSON 中的 evidence 和 product_statistics，不能使用外部知识，也不能自行补充品牌、性能、价格、销量、材质或结论。

输出要求：
1. 输出 3 到 6 条简洁中文 claims；每条 claim 必须包含 citation_ids。
2. citation_ids 只能从 allowed_citation_ids 中选择，不得创造新 ID。
3. 涉及商品统计、价格区间、销量、评分或分布时，必须引用 CATALOG_STATS:<category_key>。
4. 涉及选购原则、材质、价格、排序或跨平台比价方法时，引用对应 K 开头知识文档。
5. 不要把合成数据描述成真实平台市场结论；必要时明确“当前合成数据集”。
6. 输入证据不足时，不要猜测；少输出也比编造更好。
7. 除非证据原文明确支持，不得使用“多数、全部、唯一、必然、保证、官方、真实可追溯、最常见、占比最高”等绝对化或数量化措辞。
8. claim 文本中不要写方括号引用，也不要重复 K0021、CATALOG_STATS 等引用 ID；引用只放在 citation_ids 字段。
"""


class _GroundedClaimDraft(BaseModel):
    text: str = Field(min_length=1, max_length=260)
    citation_ids: list[str] = Field(min_length=1, max_length=6)


class _GroundedDraft(BaseModel):
    claims: list[_GroundedClaimDraft] = Field(min_length=1, max_length=8)


@dataclass(frozen=True)
class GroundedSynthesisResult:
    success: bool
    answer: str = ""
    claims: list[GroundedClaim] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    error: str | None = None


def _product_statistics_payload(insight: CategoryInsightOutput) -> dict[str, Any]:
    return {
        "category": insight.category,
        "category_key": insight.category_key,
        "components": insight.components,
        "bestsellers": [item.model_dump(mode="json") for item in insight.bestsellers],
        "attributes": [item.model_dump(mode="json") for item in insight.attributes],
        "price_tiers": [item.model_dump(mode="json") for item in insight.price_tiers],
        "product_count": insight.retrieval.get("product_count", 0),
        "data_boundary": "这些统计仅来自当前六品类合成商品数据集，不代表真实平台市场。",
    }


def _support_score(text: str, support: str) -> float:
    claim_tokens = {token for token in tokenize(text) if len(token) >= 2}
    if not claim_tokens:
        return 0.0
    support_tokens = set(tokenize(support))
    return len(claim_tokens & support_tokens) / len(claim_tokens)


def _numbers_supported(text: str, support: str, *, extra_context: str = "") -> bool:
    claim_numbers = _NUMBER_RE.findall(text)
    if not claim_numbers:
        return True
    support_numbers = [
        float(value.replace(",", ""))
        for value in _NUMBER_RE.findall(support + "\n" + extra_context)
    ]
    for raw in claim_numbers:
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            return False
        if any(math.isclose(value, candidate, rel_tol=1e-9, abs_tol=1e-6) for candidate in support_numbers):
            continue
        if "%" in text and any(
            math.isclose(value / 100.0, candidate, rel_tol=1e-9, abs_tol=1e-6)
            for candidate in support_numbers
        ):
            continue
        return False
    return True


def _validate_draft(
    draft: _GroundedDraft,
    *,
    support_by_id: dict[str, str],
    user_query: str = "",
) -> tuple[list[GroundedClaim], list[dict[str, Any]]]:
    allowed = set(support_by_id)
    valid: list[GroundedClaim] = []
    invalid: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    minimum_overlap = knowledge_synthesis_min_token_overlap()
    maximum_claims = knowledge_synthesis_max_claims()

    for raw in draft.claims[:maximum_claims]:
        text = " ".join(raw.text.split()).strip()
        citation_ids = list(dict.fromkeys(value.strip() for value in raw.citation_ids if value.strip()))
        reasons: list[str] = []
        unknown = [value for value in citation_ids if value not in allowed]
        if not text:
            reasons.append("empty_text")
        if text in seen_text:
            reasons.append("duplicate_text")
        if not citation_ids:
            reasons.append("missing_citation")
        if unknown:
            reasons.append("unknown_citation:" + ",".join(unknown))

        support = "\n".join(support_by_id[value] for value in citation_ids if value in allowed)
        overlap = _support_score(text, support) if support else 0.0
        if support and overlap < minimum_overlap:
            reasons.append(f"low_token_overlap:{overlap:.4f}")
        if support and not _numbers_supported(text, support, extra_context=user_query):
            reasons.append("unsupported_number")
        combined_support = support + "\n" + user_query
        unsupported_qualifiers = [
            qualifier
            for qualifier in _UNSUPPORTED_QUALIFIERS
            if qualifier in text and qualifier not in combined_support
        ]
        if unsupported_qualifiers:
            reasons.append("unsupported_qualifier:" + ",".join(unsupported_qualifiers))

        if reasons:
            invalid.append(
                {
                    "text": text,
                    "citation_ids": citation_ids,
                    "reasons": reasons,
                    "support_score": round(overlap, 4),
                }
            )
            continue
        seen_text.add(text)
        valid.append(
            GroundedClaim(
                text=text,
                citation_ids=citation_ids,
                support_score=round(overlap, 4),
            )
        )
    return valid, invalid


def _render_answer(claims: list[GroundedClaim]) -> str:
    return "\n".join(
        f"- {claim.text} " + "".join(f"[{citation_id}]" for citation_id in claim.citation_ids)
        for claim in claims
    )


async def synthesize_grounded_insight(
    insight: CategoryInsightOutput,
    *,
    query: str,
) -> GroundedSynthesisResult:
    started = time.perf_counter()
    statistics_id = f"CATALOG_STATS:{insight.category_key}"
    statistics = _product_statistics_payload(insight)
    support_by_id = {
        citation.doc_id: f"{citation.title}\n{citation.snippet}"
        for citation in insight.citations
    }
    support_by_id[statistics_id] = json.dumps(statistics, ensure_ascii=False, sort_keys=True)
    payload = {
        "user_query": query,
        "category": insight.category,
        "category_key": insight.category_key,
        "allowed_citation_ids": list(support_by_id),
        "evidence": [citation.model_dump(mode="json") for citation in insight.citations],
        "product_statistics": {"citation_id": statistics_id, **statistics},
    }
    model = get_llm().with_structured_output(_GroundedDraft, method="function_calling")
    invalid_history: list[dict[str, Any]] = []
    attempts = knowledge_synthesis_max_attempts()

    for attempt in range(1, attempts + 1):
        request = dict(payload)
        if invalid_history:
            request["repair_feedback"] = invalid_history[-1]
        try:
            raw = await model.ainvoke(
                [
                    ("system", _SYNTHESIS_PROMPT),
                    ("user", json.dumps(request, ensure_ascii=False)),
                ]
            )
            draft = raw if isinstance(raw, _GroundedDraft) else _GroundedDraft.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 - live provider errors must degrade safely
            return GroundedSynthesisResult(
                success=False,
                validation={
                    "valid": False,
                    "attempts": attempt,
                    "allowed_citation_ids": list(support_by_id),
                    "valid_claim_count": 0,
                    "invalid_claim_count": 0,
                },
                duration_ms=int((time.perf_counter() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

        valid, invalid = _validate_draft(
            draft,
            support_by_id=support_by_id,
            user_query=query,
        )
        validation = {
            "validator_version": 1,
            "structured_output_method": "function_calling",
            "valid": len(valid) >= knowledge_synthesis_min_claims(),
            "attempts": attempt,
            "allowed_citation_ids": list(support_by_id),
            "valid_claim_count": len(valid),
            "invalid_claim_count": len(invalid),
            "invalid_claims": invalid,
            "citation_coverage": round(
                sum(bool(claim.citation_ids) for claim in valid) / max(1, len(valid)),
                4,
            ),
            "numeric_grounding_rate": round(
                sum(
                    _numbers_supported(
                        claim.text,
                        "\n".join(support_by_id[value] for value in claim.citation_ids),
                        extra_context=query,
                    )
                    for claim in valid
                )
                / max(1, len(valid)),
                4,
            ),
        }
        if validation["valid"]:
            return GroundedSynthesisResult(
                success=True,
                answer=_render_answer(valid),
                claims=valid,
                validation=validation,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        invalid_history.append(validation)

    return GroundedSynthesisResult(
        success=False,
        validation=invalid_history[-1] if invalid_history else {"valid": False, "attempts": 0},
        duration_ms=int((time.perf_counter() - started) * 1000),
        error="grounded_claim_validation_failed",
    )
