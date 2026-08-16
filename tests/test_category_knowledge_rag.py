from __future__ import annotations

import pytest

from app.knowledge.catalog import load_knowledge_documents
from app.knowledge.evaluation import (
    build_knowledge_evaluation_cases,
    evaluate_knowledge_retriever,
)
from app.knowledge.retriever import (
    clear_category_knowledge_retriever_cache,
    get_category_knowledge_retriever,
)
from app.tools.category_insight import category_insight


def test_knowledge_catalog_has_six_complete_categories() -> None:
    documents = load_knowledge_documents()
    grouped: dict[str, set[str]] = {}
    for document in documents:
        grouped.setdefault(document.category_key, set()).add(document.title)

    assert len(documents) == 30
    assert set(grouped) == {
        "travel_storage",
        "backpack",
        "keyboard",
        "headphones",
        "thermos",
        "coffee_cup",
    }
    assert all(
        titles == {"选购要点", "材质说明", "价格分层", "排序建议", "跨平台比价"}
        for titles in grouped.values()
    )


def test_hybrid_knowledge_retrieval_is_category_scoped_and_cited() -> None:
    clear_category_knowledge_retriever_cache()
    retriever = get_category_knowledge_retriever()
    result = retriever.search(
        query="咖啡杯有哪些常见材质，粗陶和玻璃怎么选",
        category="咖啡杯",
        top_k=3,
    )

    assert result.hits
    assert result.hits[0].document.doc_id == "K0027"
    assert all(hit.document.category_key == "coffee_cup" for hit in result.hits)
    assert result.diagnostics["partition_size"] == 5
    assert result.diagnostics["vector_engine"] == "faiss_hnsw"
    assert result.diagnostics["bm25_count"] == 5
    assert result.diagnostics["vector_count"] == 5


def test_unknown_category_returns_no_knowledge_evidence() -> None:
    retriever = get_category_knowledge_retriever()
    result = retriever.search(
        query="望远镜应该怎么选",
        category="望远镜",
        top_k=5,
    )

    assert result.hits == []
    assert result.total_candidates == 0
    assert result.diagnostics["reason"] == "unknown_category"


@pytest.mark.asyncio
async def test_category_insight_uses_dataset_evidence_and_product_statistics() -> None:
    output = await category_insight(
        "咖啡杯",
        query="预算300元，想买粗陶咖啡杯，关注材质和跨平台到手价",
        category_key="coffee_cup",
    )

    assert output.category == "咖啡杯"
    assert output.category_key == "coffee_cup"
    assert output.answer_mode == "deterministic_evidence"
    assert len(output.citations) == 6
    knowledge_citations = [item for item in output.citations if item.doc_id.startswith("K00")]
    assert len(knowledge_citations) == 5
    assert any(item.doc_id == "CATALOG_STATS:coffee_cup" for item in output.citations)
    assert all(citation.category_key == "coffee_cup" for citation in output.citations)
    assert "[K" in output.evidence_summary
    assert len(output.bestsellers) == 3
    attribute_names = {attribute.name for attribute in output.attributes}
    assert {"材质", "风格"}.issubset(attribute_names)
    assert attribute_names <= {"材质", "风格", "功能"}
    assert len(output.price_tiers) == 3
    assert output.retrieval["product_count"] > 0
    assert output.confidence > 0.8


@pytest.mark.asyncio
async def test_category_insight_does_not_fallback_for_unknown_category() -> None:
    output = await category_insight("望远镜", query="望远镜怎么选")

    assert output.answer_mode == "no_evidence"
    assert output.citations == []
    assert output.bestsellers == []
    assert output.confidence == 0.0


def test_knowledge_retrieval_evaluation_reports_all_channels() -> None:
    documents = list(load_knowledge_documents())
    cases = build_knowledge_evaluation_cases(documents)
    report = evaluate_knowledge_retriever(
        get_category_knowledge_retriever(),
        cases,
        k_values=(1, 3, 5),
    )

    assert report["case_count"] == 30
    assert set(report["modes"]) == {"lexical", "vector", "hybrid"}
    for mode in report["modes"].values():
        assert 0 <= mode["recall"]["@1"] <= 1
        assert mode["recall"]["@5"] == 1.0
        assert 0 <= mode["mrr"]["@5"] <= 1
