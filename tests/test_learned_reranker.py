from __future__ import annotations

import json

from app.agent.settings import retrieval_embedding_query_prompt
from app.models import Candidate
from app.recall.hybrid import SearchFilters, clear_retriever_cache, get_hybrid_retriever
from app.recall.ltr import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    LearnedReranker,
    RerankSignals,
    clear_ltr_cache,
    reranker_model_path,
)


def _candidate(item_id: str, material: str) -> Candidate:
    return Candidate(
        item_id=item_id,
        same_group_id=item_id,
        platform="amazon",
        title=f"{material} 咖啡杯",
        category_key="coffee_cup",
        category_path=["家居", "咖啡器具", "咖啡杯"],
        price=20,
        currency="USD",
        price_cny=143.6,
        landed_price_cny=180,
        rating=4.5,
        sales=500,
        delivery_days=8,
        attributes={
            "category": "咖啡杯",
            "category_key": "coffee_cup",
            "material": material,
            "style": "简约",
            "features": ["带盖"],
            "tags": ["咖啡杯", material, "带盖"],
        },
    )


def _material_model() -> LearnedReranker:
    weights = [0.0] * len(FEATURE_NAMES)
    weights[FEATURE_NAMES.index("material_match")] = 5.0
    return LearnedReranker(
        model_version=1,
        feature_version=FEATURE_VERSION,
        feature_names=FEATURE_NAMES,
        weights=tuple(weights),
        scales=tuple(1.0 for _ in FEATURE_NAMES),
        embedding_provider="hashing",
        embedding_model=None,
        group_priors={},
        training={"test_split_used": False},
    )


def test_learned_reranker_rewards_query_material_match() -> None:
    model = _material_model()
    signals = RerankSignals(
        normalized_rrf=0.5,
        bm25_score=1.0,
        vector_score=0.5,
        bm25_rank=2,
        vector_rank=2,
        rule_score=0.5,
    )
    ceramic = _candidate("ceramic", "陶瓷")
    plastic = _candidate("plastic", "塑料")

    ceramic_score = model.score(
        candidate=ceramic,
        query="想买陶瓷咖啡杯",
        user_preferences=["偏好陶瓷"],
        budget_cny=250,
        signals=signals,
        candidate_text="陶瓷 咖啡杯 带盖",
    )
    plastic_score = model.score(
        candidate=plastic,
        query="想买陶瓷咖啡杯",
        user_preferences=["偏好陶瓷"],
        budget_cny=250,
        signals=signals,
        candidate_text="塑料 咖啡杯 带盖",
    )

    assert ceramic_score > plastic_score


def test_packaged_ltr_artifact_is_bge_compatible_and_test_safe() -> None:
    raw = json.loads(reranker_model_path().read_text(encoding="utf-8"))
    model = LearnedReranker.from_document(raw)

    assert model.feature_version == FEATURE_VERSION
    assert model.feature_names == FEATURE_NAMES
    assert model.embedding_provider == "sentence_transformers"
    assert model.embedding_model == "BAAI/bge-small-zh-v1.5"
    assert model.training["test_split_used"] is False
    assert model.training["trained_splits"] == ["train", "dev"]
    assert model.compatible("sentence_transformers", "BAAI/bge-small-zh-v1.5")
    assert not model.compatible("hashing", None)


def test_auto_reranker_falls_back_when_embedding_is_incompatible(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SHOPPILOT_RETRIEVAL_EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("SHOPPILOT_RETRIEVAL_RERANKER", "auto")
    clear_retriever_cache()
    clear_ltr_cache()
    try:
        retriever = get_hybrid_retriever()
        result = retriever.search(
            query="陶瓷咖啡杯",
            filters=SearchFilters(category_key="coffee_cup"),
            top_k=5,
        )
    finally:
        clear_retriever_cache()
        clear_ltr_cache()

    assert result.diagnostics["reranker_requested"] == "auto"
    assert result.diagnostics["reranker_applied"] == "rules"
    assert "不兼容" in result.diagnostics["reranker_fallback"]
    assert all(hit.learned_score is None for hit in result.hits)


def test_injected_learned_model_is_used_for_top_n(monkeypatch) -> None:
    monkeypatch.setenv("SHOPPILOT_RETRIEVAL_EMBEDDING_PROVIDER", "hashing")
    clear_retriever_cache()
    try:
        retriever = get_hybrid_retriever()
        result = retriever.search(
            query="陶瓷咖啡杯",
            filters=SearchFilters(category_key="coffee_cup"),
            top_k=5,
            reranker="learned",
            learned_model=_material_model(),
        )
    finally:
        clear_retriever_cache()

    assert result.diagnostics["reranker_applied"] == "learned"
    assert all(hit.learned_score is not None for hit in result.hits)


def test_bge_query_prompt_has_retrieval_instruction(monkeypatch) -> None:
    monkeypatch.setenv(
        "SHOPPILOT_RETRIEVAL_EMBEDDING_MODEL",
        "BAAI/bge-small-zh-v1.5",
    )
    monkeypatch.delenv("SHOPPILOT_RETRIEVAL_EMBEDDING_QUERY_PROMPT", raising=False)

    assert retrieval_embedding_query_prompt().startswith("为这个句子生成表示")
