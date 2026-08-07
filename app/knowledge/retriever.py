from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from app.agent.settings import (
    knowledge_bm25_weight,
    knowledge_candidate_pool,
    knowledge_rrf_k,
    knowledge_top_k,
    knowledge_vector_weight,
    retrieval_embedding_dimension,
    retrieval_embedding_model,
    retrieval_embedding_provider,
    retrieval_embedding_query_prompt,
    retrieval_faiss_threads,
    retrieval_hnsw_ef_construction,
    retrieval_hnsw_ef_search,
    retrieval_hnsw_m,
    retrieval_index_dir,
)
from app.knowledge.catalog import knowledge_file, load_knowledge_documents
from app.knowledge.models import KnowledgeDocument, KnowledgeHit, KnowledgeSearchResult
from app.recall.bm25 import BM25Index
from app.recall.embeddings import EmbeddingProvider, create_embedding_provider
from app.recall.tokenizer import normalize_text, tokenize
from app.recall.vector_index import VectorIndex
from app.utils.runtime import PROJECT_ROOT

KnowledgeMode = Literal["lexical", "vector", "hybrid"]


@dataclass
class _KnowledgePartition:
    documents: list[KnowledgeDocument]
    bm25: BM25Index
    vector: VectorIndex


def _cache_root() -> Path:
    configured = Path(retrieval_index_dir()).expanduser()
    path = configured if configured.is_absolute() else PROJECT_ROOT / configured
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _fingerprint(
    documents: list[KnowledgeDocument],
    provider: EmbeddingProvider,
) -> str:
    stat = knowledge_file().stat()
    material = {
        "knowledge_path": str(knowledge_file()),
        "knowledge_mtime_ns": stat.st_mtime_ns,
        "knowledge_size": stat.st_size,
        "doc_ids": [document.doc_id for document in documents],
        "provider": provider.name,
        "model": retrieval_embedding_model()
        if provider.name == "sentence_transformers"
        else None,
        "query_prompt": retrieval_embedding_query_prompt(),
        "dimension": provider.dimension,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:24]


def _load_or_create_vectors(
    documents: list[KnowledgeDocument],
    provider: EmbeddingProvider,
) -> tuple[list[list[float]], bool]:
    texts = [document.text for document in documents]
    try:
        import numpy as np
    except ImportError:
        return provider.embed_documents(texts), False

    fingerprint = _fingerprint(documents, provider)
    vector_path = _cache_root() / f"knowledge-embeddings-{fingerprint}.npy"
    metadata_path = _cache_root() / f"knowledge-embeddings-{fingerprint}.json"
    if vector_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            matrix = np.load(vector_path, allow_pickle=False)
            if (
                matrix.shape == (len(documents), provider.dimension)
                and metadata.get("doc_ids") == [document.doc_id for document in documents]
            ):
                return matrix.astype("float32").tolist(), True
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    matrix = np.asarray(provider.embed_documents(texts), dtype="float32")
    expected_shape = (len(documents), provider.dimension)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"知识 Embedding 维度错误: expected={expected_shape}, actual={matrix.shape}"
        )
    temporary = vector_path.with_suffix(".tmp.npy")
    np.save(temporary, matrix, allow_pickle=False)
    temporary.replace(vector_path)
    metadata_path.write_text(
        json.dumps(
            {
                "provider": provider.name,
                "model": retrieval_embedding_model(),
                "dimension": provider.dimension,
                "doc_ids": [document.doc_id for document in documents],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return matrix.tolist(), False


class CategoryKnowledgeRetriever:
    def __init__(self, documents: list[KnowledgeDocument]) -> None:
        if not documents:
            raise ValueError("无法为零篇知识文档构建索引")
        self.documents = documents
        self.provider = create_embedding_provider()
        self.vectors, self.embedding_cache_hit = _load_or_create_vectors(
            documents,
            self.provider,
        )
        self.category_aliases: dict[str, str] = {}
        grouped: dict[str | None, list[int]] = {None: list(range(len(documents)))}
        for index, document in enumerate(documents):
            grouped.setdefault(document.category_key, []).append(index)
            for alias in [document.category_key, *document.category_path, document.category]:
                normalized = normalize_text(alias)
                if normalized:
                    self.category_aliases[normalized] = document.category_key

        self.partitions: dict[str | None, _KnowledgePartition] = {}
        for category_key, indices in grouped.items():
            partition_documents = [documents[index] for index in indices]
            partition_vectors = [self.vectors[index] for index in indices]
            self.partitions[category_key] = _KnowledgePartition(
                documents=partition_documents,
                bm25=BM25Index([tokenize(document.text) for document in partition_documents]),
                vector=VectorIndex(partition_vectors),
            )

    def resolve_category_key(
        self,
        category: str | None,
        category_key: str | None,
    ) -> str | None:
        requested = category_key or category
        if not requested:
            return None
        return self.category_aliases.get(normalize_text(requested))

    def search(
        self,
        *,
        query: str,
        category: str | None = None,
        category_key: str | None = None,
        top_k: int | None = None,
        mode: KnowledgeMode = "hybrid",
    ) -> KnowledgeSearchResult:
        started = time.perf_counter()
        requested_category = bool(category or category_key)
        resolved_category = self.resolve_category_key(category, category_key)
        if requested_category and resolved_category is None:
            return KnowledgeSearchResult(
                hits=[],
                total_candidates=0,
                diagnostics={
                    "mode": mode,
                    "resolved_category_key": None,
                    "embedding_provider": self.provider.name,
                    "embedding_model": retrieval_embedding_model(),
                    "embedding_dimension": self.provider.dimension,
                    "vector_engine": "none",
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "reason": "unknown_category",
                },
            )

        partition = self.partitions[resolved_category if requested_category else None]
        limit = max(1, top_k or knowledge_top_k())
        pool = min(
            len(partition.documents),
            max(limit, knowledge_candidate_pool()),
        )
        query_text = " ".join(
            part for part in [category or "", category_key or "", query] if part
        )

        lexical_ranked: list[tuple[int, float]] = []
        if mode in {"lexical", "hybrid"}:
            lexical_ranked = partition.bm25.rank(tokenize(query_text), pool)

        vector_ranked: list[tuple[int, float]] = []
        if mode in {"vector", "hybrid"}:
            vector_ranked = partition.vector.search(
                self.provider.embed_query(query_text),
                pool,
            )

        fused: dict[int, float] = {}
        rrf_k = knowledge_rrf_k()
        for rank, (index, _) in enumerate(lexical_ranked, start=1):
            fused[index] = fused.get(index, 0.0) + knowledge_bm25_weight() / (
                rrf_k + rank
            )
        for rank, (index, _) in enumerate(vector_ranked, start=1):
            fused[index] = fused.get(index, 0.0) + knowledge_vector_weight() / (
                rrf_k + rank
            )

        bm25_scores = {index: score for index, score in lexical_ranked}
        vector_scores = {index: score for index, score in vector_ranked}
        bm25_ranks = {index: rank for rank, (index, _) in enumerate(lexical_ranked, 1)}
        vector_ranks = {index: rank for rank, (index, _) in enumerate(vector_ranked, 1)}
        hits = [
            KnowledgeHit(
                document=partition.documents[index],
                score=round(score, 8),
                bm25_score=round(bm25_scores.get(index, 0.0), 8),
                vector_score=round(vector_scores.get(index, 0.0), 8),
                bm25_rank=bm25_ranks.get(index),
                vector_rank=vector_ranks.get(index),
            )
            for index, score in fused.items()
        ]
        hits.sort(
            key=lambda hit: (
                -hit.score,
                hit.document.doc_id,
            )
        )
        return KnowledgeSearchResult(
            hits=hits[:limit],
            total_candidates=len(partition.documents),
            diagnostics={
                "mode": mode,
                "resolved_category_key": resolved_category,
                "partition_size": len(partition.documents),
                "embedding_provider": self.provider.name,
                "embedding_model": retrieval_embedding_model(),
                "embedding_dimension": self.provider.dimension,
                "embedding_cache_hit": self.embedding_cache_hit,
                "vector_engine": partition.vector.engine,
                "bm25_count": len(lexical_ranked),
                "vector_count": len(vector_ranked),
                "fused_count": len(fused),
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )


@lru_cache(maxsize=4)
def _cached_retriever(
    path_text: str,
    modified_ns: int,
    provider_name: str,
    model_name: str,
    dimension: int,
    query_prompt: str,
    index_dir: str,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
    faiss_threads: int,
) -> CategoryKnowledgeRetriever:
    del (
        path_text,
        modified_ns,
        provider_name,
        model_name,
        dimension,
        query_prompt,
        index_dir,
        hnsw_m,
        hnsw_ef_construction,
        hnsw_ef_search,
        faiss_threads,
    )
    return CategoryKnowledgeRetriever(list(load_knowledge_documents()))


def get_category_knowledge_retriever() -> CategoryKnowledgeRetriever:
    path = knowledge_file()
    return _cached_retriever(
        str(path),
        path.stat().st_mtime_ns,
        retrieval_embedding_provider(),
        retrieval_embedding_model(),
        retrieval_embedding_dimension(),
        retrieval_embedding_query_prompt(),
        retrieval_index_dir(),
        retrieval_hnsw_m(),
        retrieval_hnsw_ef_construction(),
        retrieval_hnsw_ef_search(),
        retrieval_faiss_threads(),
    )


def clear_category_knowledge_retriever_cache() -> None:
    _cached_retriever.cache_clear()
