from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from app.agent.settings import (
    retrieval_backend,
    retrieval_bm25_weight,
    retrieval_candidate_pool,
    retrieval_embedding_dimension,
    retrieval_embedding_model,
    retrieval_embedding_provider,
    retrieval_embedding_query_prompt,
    retrieval_faiss_threads,
    retrieval_hnsw_ef_construction,
    retrieval_hnsw_ef_search,
    retrieval_hnsw_m,
    retrieval_index_dir,
    retrieval_rerank_top_n,
    retrieval_rerank_weight,
    retrieval_reranker,
    retrieval_rrf_k,
    retrieval_vector_weight,
)
from app.models import Candidate, Platform
from app.recall.bm25 import BM25Index
from app.recall.catalog import load_catalog, products_file
from app.recall.embeddings import EmbeddingProvider, create_embedding_provider
from app.recall.ltr import (
    LearnedReranker,
    RerankSignals,
    RerankerMode,
    load_compatible_reranker,
)
from app.recall.tokenizer import normalize_text, tokenize, unique_tokens
from app.recall.vector_index import VectorIndex
from app.utils.runtime import PROJECT_ROOT

RetrievalMode = Literal["lexical", "vector", "hybrid"]


@dataclass(frozen=True)
class SearchFilters:
    platform: Platform | None = None
    category: str | None = None
    category_key: str | None = None
    max_landed_price_cny: float | None = None
    excluded_terms: tuple[str, ...] = ()
    include_unavailable: bool = False


@dataclass(frozen=True)
class RetrievalHit:
    candidate: Candidate
    final_score: float
    rrf_score: float
    rerank_score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    bm25_rank: int | None = None
    vector_rank: int | None = None
    learned_score: float | None = None


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[RetrievalHit]
    total_candidates: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Partition:
    candidates: list[Candidate]
    texts: list[str]
    bm25: BM25Index
    vector: VectorIndex


def candidate_text(candidate: Candidate) -> str:
    attributes = candidate.attributes
    return " ".join(
        part
        for part in [
            candidate.title,
            candidate.title_en or "",
            candidate.description,
            candidate.brand or "",
            candidate.category_key,
            " ".join(candidate.category_path),
            str(attributes.get("category", "")),
            str(attributes.get("category_key", "")),
            str(attributes.get("material", "")),
            str(attributes.get("style", "")),
            str(attributes.get("color", "")),
            " ".join(str(value) for value in attributes.get("features", [])),
            " ".join(str(value) for value in attributes.get("tags", [])),
        ]
        if part
    )


def _cache_root() -> Path:
    configured = Path(retrieval_index_dir()).expanduser()
    path = configured if configured.is_absolute() else PROJECT_ROOT / configured
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _embedding_fingerprint(
    candidates: list[Candidate],
    provider: EmbeddingProvider,
) -> str:
    stat = products_file().stat()
    material = {
        "dataset_path": str(products_file()),
        "dataset_mtime_ns": stat.st_mtime_ns,
        "dataset_size": stat.st_size,
        "count": len(candidates),
        "first": candidates[0].item_id if candidates else None,
        "last": candidates[-1].item_id if candidates else None,
        "provider": provider.name,
        "model": retrieval_embedding_model()
        if provider.name == "sentence_transformers"
        else None,
        "dimension": provider.dimension,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:24]


def _load_or_create_vectors(
    candidates: list[Candidate],
    texts: list[str],
    provider: EmbeddingProvider,
) -> tuple[list[list[float]], bool]:
    try:
        import numpy as np
    except ImportError:
        return provider.embed_documents(texts), False

    fingerprint = _embedding_fingerprint(candidates, provider)
    vector_path = _cache_root() / f"embeddings-{fingerprint}.npy"
    metadata_path = _cache_root() / f"embeddings-{fingerprint}.json"
    if vector_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            matrix = np.load(vector_path, allow_pickle=False)
            if (
                matrix.shape == (len(candidates), provider.dimension)
                and metadata.get("item_ids") == [item.item_id for item in candidates]
            ):
                return matrix.astype("float32").tolist(), True
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    vectors = provider.embed_documents(texts)
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.shape != (len(candidates), provider.dimension):
        raise ValueError(
            "Embedding 输出维度错误: "
            f"expected={(len(candidates), provider.dimension)}, actual={matrix.shape}"
        )
    temporary = vector_path.with_suffix(".tmp.npy")
    np.save(temporary, matrix, allow_pickle=False)
    temporary.replace(vector_path)
    metadata_path.write_text(
        json.dumps(
            {
                "provider": provider.name,
                "dimension": provider.dimension,
                "item_ids": [item.item_id for item in candidates],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return matrix.tolist(), False


def _partition_keys(candidate: Candidate) -> tuple[tuple[str | None, str | None], ...]:
    return (
        (None, None),
        (None, candidate.category_key),
        (candidate.platform, None),
        (candidate.platform, candidate.category_key),
    )


def _negative_terms(constraints: list[str]) -> tuple[str, ...]:
    terms: list[str] = []
    patterns = (
        re.compile(r"(?:不要|不含|排除|避免|拒绝)(.+)"),
        re.compile(r"(.+?)(?:禁用|不可接受)"),
    )
    for constraint in constraints:
        normalized = normalize_text(constraint)
        if "预算" in normalized and "不超过" in normalized:
            continue
        for pattern in patterns:
            match = pattern.search(normalized)
            if not match:
                continue
            value = match.group(1).strip(" ，,。.;；:：")
            if value:
                terms.append(value)
            break
    return tuple(dict.fromkeys(terms))


def exclusions_from_constraints(constraints: list[str] | None) -> tuple[str, ...]:
    return _negative_terms(constraints or [])


class HybridRetriever:
    def __init__(self, candidates: list[Candidate]) -> None:
        if not candidates:
            raise ValueError("无法为零商品构建检索索引")
        self.candidates = candidates
        self.provider = create_embedding_provider()
        self.texts = [candidate_text(candidate) for candidate in candidates]
        self.vectors, self.embedding_cache_hit = _load_or_create_vectors(
            candidates,
            self.texts,
            self.provider,
        )
        grouped: dict[tuple[str | None, str | None], list[int]] = {}
        self.category_aliases: dict[str, str] = {}
        for index, candidate in enumerate(candidates):
            for key in _partition_keys(candidate):
                grouped.setdefault(key, []).append(index)
            aliases = [
                candidate.category_key,
                *candidate.category_path,
                str(candidate.attributes.get("category", "")),
                str(candidate.attributes.get("category_key", "")),
            ]
            for alias in aliases:
                normalized = normalize_text(alias)
                if normalized:
                    self.category_aliases[normalized] = candidate.category_key

        self.partitions: dict[tuple[str | None, str | None], _Partition] = {}
        for key, global_indices in grouped.items():
            partition_candidates = [candidates[index] for index in global_indices]
            partition_texts = [self.texts[index] for index in global_indices]
            partition_vectors = [self.vectors[index] for index in global_indices]
            self.partitions[key] = _Partition(
                candidates=partition_candidates,
                texts=partition_texts,
                bm25=BM25Index([tokenize(text) for text in partition_texts]),
                vector=VectorIndex(partition_vectors),
            )

    def resolve_category_key(
        self,
        category: str | None,
        category_key: str | None,
    ) -> str | None:
        if category_key:
            normalized = normalize_text(category_key)
            return self.category_aliases.get(normalized)
        if not category:
            return None
        return self.category_aliases.get(normalize_text(category))

    def _partition(self, filters: SearchFilters) -> tuple[_Partition, str | None]:
        category_key = self.resolve_category_key(filters.category, filters.category_key)
        key = (filters.platform, category_key)
        partition = self.partitions.get(key)
        if partition is not None:
            return partition, category_key
        fallback = self.partitions.get((filters.platform, None))
        if fallback is None:
            fallback = self.partitions[(None, None)]
        return fallback, category_key

    @staticmethod
    def _eligible(candidate: Candidate, filters: SearchFilters) -> bool:
        if not filters.include_unavailable and not candidate.is_available:
            return False
        if filters.platform and candidate.platform != filters.platform:
            return False
        if filters.max_landed_price_cny is not None:
            price = candidate.landed_price_cny or candidate.price_cny
            if price is not None and price > filters.max_landed_price_cny:
                return False
        if filters.excluded_terms:
            haystack = normalize_text(candidate_text(candidate))
            if any(normalize_text(term) in haystack for term in filters.excluded_terms):
                return False
        return True

    @staticmethod
    def _business_score(
        candidate: Candidate,
        preference_tokens: list[str],
        max_budget: float | None,
        text: str,
    ) -> float:
        normalized_text = normalize_text(text)
        preference_score = 0.0
        if preference_tokens:
            hits = sum(token in normalized_text for token in preference_tokens)
            preference_score = hits / len(preference_tokens)
        rating_score = max(0.0, min(float(candidate.rating or 0.0) / 5.0, 1.0))
        sales_score = min(math.log1p(float(candidate.sales or 0.0)) / math.log1p(5000), 1.0)
        delivery_score = max(0.0, 1.0 - max(0, (candidate.delivery_days or 30) - 5) / 30)
        quality_score = 1.0 if candidate.quality_grade == "A" else 0.4
        budget_score = 0.5
        price = candidate.landed_price_cny or candidate.price_cny
        if max_budget and price is not None:
            budget_score = max(0.0, min(1.0, 1.0 - price / max_budget * 0.45))
        return (
            preference_score * 0.40
            + rating_score * 0.20
            + sales_score * 0.12
            + budget_score * 0.15
            + delivery_score * 0.08
            + quality_score * 0.05
        )

    def search(
        self,
        *,
        query: str,
        filters: SearchFilters,
        top_k: int,
        user_preferences: list[str] | None = None,
        mode: RetrievalMode | None = None,
        reranker: RerankerMode | None = None,
        learned_model: LearnedReranker | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        selected_mode = mode or retrieval_backend()
        partition, resolved_category = self._partition(filters)
        query_text = " ".join(
            part
            for part in [query, *(user_preferences or [])]
            if part
        )
        query_tokens = tokenize(query_text)
        pool = min(
            len(partition.candidates),
            max(top_k, retrieval_candidate_pool()),
        )
        category_requested = bool(filters.category or filters.category_key)
        eligible_indices = (
            set()
            if category_requested and resolved_category is None
            else {
                index
                for index, candidate in enumerate(partition.candidates)
                if self._eligible(candidate, filters)
            }
        )

        lexical_ranked: list[tuple[int, float]] = []
        if selected_mode in {"lexical", "hybrid"}:
            lexical_ranked = [
                pair
                for pair in partition.bm25.rank(query_tokens, len(partition.candidates))
                if pair[0] in eligible_indices
            ][:pool]

        vector_ranked: list[tuple[int, float]] = []
        if selected_mode in {"vector", "hybrid"}:
            query_vector = self.provider.embed_query(query_text)
            vector_ranked = [
                pair
                for pair in partition.vector.search(query_vector, len(partition.candidates))
                if pair[0] in eligible_indices
            ][:pool]

        fused: dict[int, float] = {}
        rrf_k = retrieval_rrf_k()
        for rank, (index, _) in enumerate(lexical_ranked, start=1):
            fused[index] = fused.get(index, 0.0) + retrieval_bm25_weight() / (
                rrf_k + rank
            )
        for rank, (index, _) in enumerate(vector_ranked, start=1):
            fused[index] = fused.get(index, 0.0) + retrieval_vector_weight() / (
                rrf_k + rank
            )
        if not fused:
            fused = {index: 0.0 for index in eligible_indices}

        bm25_scores = {index: score for index, score in lexical_ranked}
        vector_scores = {index: score for index, score in vector_ranked}
        bm25_ranks = {index: rank for rank, (index, _) in enumerate(lexical_ranked, 1)}
        vector_ranks = {index: rank for rank, (index, _) in enumerate(vector_ranked, 1)}
        maximum_rrf = max(fused.values(), default=1.0)
        minimum_rrf = min(fused.values(), default=0.0)
        spread = max(maximum_rrf - minimum_rrf, 1e-12)
        preference_tokens = unique_tokens(user_preferences or [])
        rerank_weight = retrieval_rerank_weight()
        requested_reranker = reranker or retrieval_reranker()
        applied_reranker = requested_reranker
        reranker_fallback: str | None = None
        resolved_learned_model = learned_model
        if requested_reranker in {"auto", "learned"}:
            if resolved_learned_model is None:
                resolved_learned_model, reranker_fallback = load_compatible_reranker(
                    provider_name=self.provider.name,
                    model_name=getattr(self.provider, "model_name", None),
                )
            applied_reranker = (
                "learned" if resolved_learned_model is not None else "rules"
            )

        pending: list[dict[str, Any]] = []
        for index, rrf_score in fused.items():
            candidate = partition.candidates[index]
            normalized_rrf = (
                1.0
                if maximum_rrf == minimum_rrf and maximum_rrf > 0
                else (rrf_score - minimum_rrf) / spread
            )
            business_score = self._business_score(
                candidate,
                preference_tokens,
                filters.max_landed_price_cny,
                partition.texts[index],
            )
            rule_final = (
                (1.0 - rerank_weight) * normalized_rrf
                + rerank_weight * business_score
            )
            pending.append(
                {
                    "index": index,
                    "candidate": candidate,
                    "rrf_score": rrf_score,
                    "normalized_rrf": normalized_rrf,
                    "business_score": business_score,
                    "rule_final": rule_final,
                }
            )
        pending.sort(
            key=lambda row: (
                -row["rule_final"],
                -(row["candidate"].rating or 0.0),
                -(row["candidate"].sales or 0),
                row["candidate"].item_id,
            )
        )
        if applied_reranker == "learned":
            pending = pending[: max(top_k, retrieval_rerank_top_n())]

        hits: list[RetrievalHit] = []
        for row in pending:
            index = int(row["index"])
            candidate = row["candidate"]
            learned_score: float | None = None
            if applied_reranker == "learned" and resolved_learned_model is not None:
                learned_score = resolved_learned_model.score(
                    candidate=candidate,
                    query=query,
                    user_preferences=user_preferences or [],
                    budget_cny=filters.max_landed_price_cny,
                    signals=RerankSignals(
                        normalized_rrf=float(row["normalized_rrf"]),
                        bm25_score=bm25_scores.get(index, 0.0),
                        vector_score=vector_scores.get(index, 0.0),
                        bm25_rank=bm25_ranks.get(index),
                        vector_rank=vector_ranks.get(index),
                        rule_score=float(row["business_score"]),
                    ),
                    candidate_text=partition.texts[index],
                )
                final_score = learned_score
            elif applied_reranker == "none":
                final_score = float(row["normalized_rrf"])
            else:
                final_score = float(row["rule_final"])
            hits.append(
                RetrievalHit(
                    candidate=candidate,
                    final_score=round(final_score, 8),
                    rrf_score=round(float(row["rrf_score"]), 8),
                    rerank_score=round(float(row["business_score"]), 8),
                    bm25_score=round(bm25_scores.get(index, 0.0), 8),
                    vector_score=round(vector_scores.get(index, 0.0), 8),
                    bm25_rank=bm25_ranks.get(index),
                    vector_rank=vector_ranks.get(index),
                    learned_score=(
                        round(learned_score, 8)
                        if learned_score is not None
                        else None
                    ),
                )
            )
        hits.sort(
            key=lambda hit: (
                -hit.final_score,
                -(hit.candidate.rating or 0.0),
                -(hit.candidate.sales or 0),
                hit.candidate.item_id,
            )
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        diagnostics: dict[str, Any] = {
            "mode": selected_mode,
            "embedding_provider": self.provider.name,
            "embedding_model": getattr(self.provider, "model_name", None),
            "embedding_dimension": self.provider.dimension,
            "embedding_cache_hit": self.embedding_cache_hit,
            "vector_engine": partition.vector.engine,
            "resolved_category_key": resolved_category,
            "partition_size": len(partition.candidates),
            "eligible_count": len(eligible_indices),
            "bm25_count": len(lexical_ranked),
            "vector_count": len(vector_ranked),
            "fused_count": len(fused),
            "reranker_requested": requested_reranker,
            "reranker_applied": applied_reranker,
            "rerank_top_n": max(top_k, retrieval_rerank_top_n()),
            "duration_ms": duration_ms,
        }
        if reranker_fallback:
            diagnostics["reranker_fallback"] = reranker_fallback
        if resolved_learned_model is not None and applied_reranker == "learned":
            diagnostics["reranker_model_version"] = resolved_learned_model.model_version
            diagnostics["reranker_training"] = resolved_learned_model.training
        return RetrievalResult(
            hits=hits[:top_k],
            total_candidates=len(eligible_indices),
            diagnostics=diagnostics,
        )


@lru_cache(maxsize=4)
def _cached_retriever(
    dataset_path: str,
    dataset_mtime_ns: int,
    provider_name: str,
    model_name: str,
    query_prompt: str,
    embedding_dimension: int,
    index_dir: str,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
    faiss_threads: int,
) -> HybridRetriever:
    del (
        dataset_path,
        dataset_mtime_ns,
        provider_name,
        model_name,
        query_prompt,
        embedding_dimension,
        index_dir,
        hnsw_m,
        hnsw_ef_construction,
        hnsw_ef_search,
        faiss_threads,
    )
    return HybridRetriever(list(load_catalog()))


def get_hybrid_retriever() -> HybridRetriever:
    path = products_file()
    return _cached_retriever(
        str(path),
        path.stat().st_mtime_ns,
        retrieval_embedding_provider(),
        retrieval_embedding_model(),
        retrieval_embedding_query_prompt(),
        retrieval_embedding_dimension(),
        retrieval_index_dir(),
        retrieval_hnsw_m(),
        retrieval_hnsw_ef_construction(),
        retrieval_hnsw_ef_search(),
        retrieval_faiss_threads(),
    )


def clear_retriever_cache() -> None:
    _cached_retriever.cache_clear()
