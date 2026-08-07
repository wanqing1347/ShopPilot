from __future__ import annotations

from typing import Any

import pytest

from app.knowledge.synthesis import synthesize_grounded_insight
from app.models import (
    AttributeDist,
    Bestseller,
    CategoryInsightOutput,
    KnowledgeCitation,
    PriceTier,
)
from app.tools.category_insight import category_insight


class _FakeStructuredModel:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages: list[tuple[str, str]]) -> Any:
        del messages
        self.calls += 1
        response = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


class _FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.structured = _FakeStructuredModel(responses)

    def with_structured_output(self, schema: Any, method: str = "function_calling") -> Any:
        del schema, method
        return self.structured


def _insight() -> CategoryInsightOutput:
    return CategoryInsightOutput(
        category="咖啡杯",
        category_key="coffee_cup",
        components=["带盖", "耐高温"],
        bestsellers=[
            Bestseller(
                name="陶瓷随行咖啡杯",
                typical_price_cny=220.0,
                why_popular="当前合成目录销量较高",
            )
        ],
        attributes=[
            AttributeDist(name="材质", distribution={"陶瓷": 0.6, "玻璃": 0.4})
        ],
        price_tiers=[
            PriceTier(tier="budget", range_cny=(50.0, 150.0), notes="低位"),
            PriceTier(tier="mid", range_cny=(150.0, 300.0), notes="中位"),
            PriceTier(tier="premium", range_cny=(300.0, 500.0), notes="高位"),
        ],
        citations=[
            KnowledgeCitation(
                doc_id="K0027",
                category_key="coffee_cup",
                title="材质说明",
                snippet="咖啡杯常见材质包括陶瓷、粗陶、炻器、高硼硅玻璃、骨瓷。",
                source="synthetic_project_knowledge",
            )
        ],
        retrieval={"product_count": 200},
        confidence=0.9,
    )


@pytest.mark.asyncio
async def test_grounded_synthesis_accepts_only_supported_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLLM(
        [
            {
                "claims": [
                    {
                        "text": "咖啡杯常见材质包括陶瓷和高硼硅玻璃。",
                        "citation_ids": ["K0027"],
                    },
                    {
                        "text": "当前合成商品目录包含200条咖啡杯商品。",
                        "citation_ids": ["CATALOG_STATS:coffee_cup"],
                    },
                ]
            }
        ]
    )
    monkeypatch.setattr("app.knowledge.synthesis.get_llm", lambda: fake)
    monkeypatch.setenv("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MIN_CLAIMS", "2")
    monkeypatch.setenv("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MAX_ATTEMPTS", "1")

    result = await synthesize_grounded_insight(_insight(), query="咖啡杯材质和价格怎么选")

    assert result.success is True
    assert len(result.claims) == 2
    assert result.validation["citation_coverage"] == 1.0
    assert result.validation["numeric_grounding_rate"] == 1.0
    assert "[K0027]" in result.answer
    assert "[CATALOG_STATS:coffee_cup]" in result.answer


@pytest.mark.asyncio
async def test_grounded_synthesis_rejects_invented_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLLM(
        [
            {
                "claims": [
                    {
                        "text": "咖啡杯都适合放入洗碗机。",
                        "citation_ids": ["K9999"],
                    }
                ]
            }
        ]
    )
    monkeypatch.setattr("app.knowledge.synthesis.get_llm", lambda: fake)
    monkeypatch.setenv("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MIN_CLAIMS", "1")
    monkeypatch.setenv("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MAX_ATTEMPTS", "1")

    result = await synthesize_grounded_insight(_insight(), query="咖啡杯怎么清洗")

    assert result.success is False
    assert result.claims == []
    assert result.validation["invalid_claim_count"] == 1
    assert "unknown_citation:K9999" in result.validation["invalid_claims"][0]["reasons"]


@pytest.mark.asyncio
async def test_grounded_synthesis_rejects_unsupported_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLLM(
        [
            {
                "claims": [
                    {
                        "text": "当前合成商品目录包含9999条咖啡杯商品。",
                        "citation_ids": ["CATALOG_STATS:coffee_cup"],
                    }
                ]
            }
        ]
    )
    monkeypatch.setattr("app.knowledge.synthesis.get_llm", lambda: fake)
    monkeypatch.setenv("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MIN_CLAIMS", "1")
    monkeypatch.setenv("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MAX_ATTEMPTS", "1")

    result = await synthesize_grounded_insight(_insight(), query="有多少咖啡杯商品")

    assert result.success is False
    assert "unsupported_number" in result.validation["invalid_claims"][0]["reasons"]


@pytest.mark.asyncio
async def test_grounded_synthesis_rejects_unsupported_qualifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLLM(
        [
            {
                "claims": [
                    {
                        "text": "中位价格区间覆盖当前合成数据集中的多数商品。",
                        "citation_ids": ["CATALOG_STATS:coffee_cup"],
                    }
                ]
            }
        ]
    )
    monkeypatch.setattr("app.knowledge.synthesis.get_llm", lambda: fake)
    monkeypatch.setenv("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MIN_CLAIMS", "1")
    monkeypatch.setenv("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MAX_ATTEMPTS", "1")

    result = await synthesize_grounded_insight(_insight(), query="咖啡杯价格怎么选")

    assert result.success is False
    assert any(
        reason.startswith("unsupported_qualifier:多数")
        for reason in result.validation["invalid_claims"][0]["reasons"]
    )


@pytest.mark.asyncio
async def test_document_ids_in_claim_text_are_not_treated_as_fact_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLLM(
        [
            {
                "claims": [
                    {
                        "text": "咖啡杯材质可参考K0027中的陶瓷和高硼硅玻璃说明。",
                        "citation_ids": ["K0027"],
                    }
                ]
            }
        ]
    )
    monkeypatch.setattr("app.knowledge.synthesis.get_llm", lambda: fake)
    monkeypatch.setenv("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MIN_CLAIMS", "1")
    monkeypatch.setenv("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MAX_ATTEMPTS", "1")

    result = await synthesize_grounded_insight(_insight(), query="咖啡杯材质怎么选")

    assert result.success is True
    assert result.validation["invalid_claim_count"] == 0


@pytest.mark.asyncio
async def test_no_evidence_never_calls_grounded_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unexpected(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("no_evidence must not call the LLM synthesizer")

    monkeypatch.setenv("SHOPPILOT_KNOWLEDGE_SYNTHESIS_ENABLED", "true")
    monkeypatch.setattr("app.tools.category_insight.synthesize_grounded_insight", _unexpected)

    output = await category_insight("望远镜", query="望远镜怎么选")

    assert output.answer_mode == "no_evidence"
    assert output.grounded_answer == ""
    assert output.citation_validation == {}
