from __future__ import annotations

import concurrent.futures
import time
from functools import lru_cache

import pytest

from app.recall.bm25 import BM25Index
from app.recall.catalog import dataset_root
from app.recall.evaluation import evaluate_retriever, load_evaluation_queries
from app.recall.hybrid import (
    SearchFilters,
    _embedding_fingerprint,
    candidate_text,
    clear_retriever_cache,
    get_hybrid_retriever,
)
from app.recall.embeddings import HashingEmbeddingProvider
from app.recall.tokenizer import tokenize


def test_embedding_fingerprint_depends_on_vector_text_not_file_mtime() -> None:
    from app.recall.catalog import load_catalog

    candidates = list(load_catalog()[:3])
    texts = [candidate_text(candidate) for candidate in candidates]
    provider = HashingEmbeddingProvider(dimension=128)

    first, first_digest = _embedding_fingerprint(candidates, texts, provider)
    second, second_digest = _embedding_fingerprint(candidates, list(texts), provider)
    changed, changed_digest = _embedding_fingerprint(
        candidates,
        [texts[0] + " changed", *texts[1:]],
        provider,
    )

    assert first == second
    assert first_digest == second_digest
    assert changed != first
    assert changed_digest != first_digest


def test_get_hybrid_retriever_single_flights_concurrent_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.recall.hybrid as hybrid

    clear_retriever_cache()
    build_calls = 0

    @lru_cache(maxsize=4)
    def slow_cached_retriever(*args):
        nonlocal build_calls
        build_calls += 1
        time.sleep(0.05)
        return object()

    monkeypatch.setattr(hybrid, "_cached_retriever", slow_cached_retriever)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: get_hybrid_retriever(), range(4)))

    assert build_calls == 1
    assert len({id(result) for result in results}) == 1


def test_bm25_tokenizer_ranks_matching_chinese_product() -> None:
    documents = [
        tokenize("小众手作粗陶咖啡杯 带壶嘴 耐高温"),
        tokenize("主动降噪头戴耳机 通透模式 低延迟"),
        tokenize("旅行收纳袋 防泼水 可压缩"),
    ]
    index = BM25Index(documents)

    ranked = index.rank(tokenize("粗陶咖啡杯带壶嘴"), limit=3)

    assert ranked[0][0] == 0
    assert ranked[0][1] > 0


def test_hybrid_retriever_uses_faiss_partition_and_hard_filters() -> None:
    clear_retriever_cache()
    retriever = get_hybrid_retriever()
    result = retriever.search(
        query="预算300元，小众手作粗陶咖啡杯，最好带壶嘴",
        filters=SearchFilters(
            platform="amazon",
            category="咖啡杯",
            max_landed_price_cny=300,
            excluded_terms=("塑料",),
        ),
        top_k=10,
        user_preferences=["偏好小众手作", "偏好粗陶"],
    )

    assert result.hits
    assert result.diagnostics["mode"] == "hybrid"
    assert result.diagnostics["vector_engine"] == "faiss_hnsw"
    assert result.diagnostics["resolved_category_key"] == "coffee_cup"
    assert result.diagnostics["partition_size"] == 50
    assert result.diagnostics["bm25_count"] > 0
    assert result.diagnostics["vector_count"] > 0
    assert all(hit.candidate.platform == "amazon" for hit in result.hits)
    assert all(hit.candidate.category_key == "coffee_cup" for hit in result.hits)
    assert all((hit.candidate.landed_price_cny or 0) <= 300 for hit in result.hits)
    assert all("塑料" not in str(hit.candidate.attributes) for hit in result.hits)
    assert all(hit.bm25_rank is not None or hit.vector_rank is not None for hit in result.hits)


def test_unknown_category_does_not_fall_back_to_unrelated_products() -> None:
    retriever = get_hybrid_retriever()

    result = retriever.search(
        query="想买望远镜",
        filters=SearchFilters(platform="amazon", category="望远镜"),
        top_k=10,
    )

    assert result.hits == []
    assert result.total_candidates == 0
    assert result.diagnostics["resolved_category_key"] is None


def test_retrieval_evaluator_reports_three_channels() -> None:
    retriever = get_hybrid_retriever()
    queries = load_evaluation_queries(
        dataset_root() / "queries.jsonl",
        split="test",
    )[:6]

    report = evaluate_retriever(
        retriever,
        queries,
        k_values=(5, 10),
    )

    assert report["query_count"] == 6
    assert report["embedding_provider"] == "hashing"
    assert set(report["modes"]) == {"lexical", "vector", "hybrid"}
    for mode in report["modes"].values():
        assert 0 <= mode["recall"]["@5"] <= 1
        assert 0 <= mode["mrr"]["@10"] <= 1
        assert 0 <= mode["ndcg"]["@10"] <= 1
        assert 0 <= mode["hard_constraint_satisfaction"]["@10"] <= 1
