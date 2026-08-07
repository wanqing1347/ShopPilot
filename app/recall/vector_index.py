from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.agent.settings import (
    retrieval_hnsw_ef_construction,
    retrieval_faiss_threads,
    retrieval_hnsw_ef_search,
    retrieval_hnsw_m,
)


class VectorIndex:
    """Normalized cosine search using Faiss HNSW with an exact Python fallback."""

    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self.vectors = [list(vector) for vector in vectors]
        self.dimension = len(self.vectors[0]) if self.vectors else 0
        self.engine = "python_exact"
        self._faiss: Any | None = None
        self._index: Any | None = None
        self._matrix: Any | None = None
        if not self.vectors:
            return
        if any(len(vector) != self.dimension for vector in self.vectors):
            raise ValueError("向量维度不一致")
        try:
            import faiss
            import numpy as np
        except ImportError:
            return

        faiss.omp_set_num_threads(retrieval_faiss_threads())
        matrix = np.asarray(self.vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        try:
            index = faiss.IndexHNSWFlat(
                self.dimension,
                retrieval_hnsw_m(),
                faiss.METRIC_INNER_PRODUCT,
            )
        except TypeError:
            index = faiss.index_factory(
                self.dimension,
                f"HNSW{retrieval_hnsw_m()},Flat",
                faiss.METRIC_INNER_PRODUCT,
            )
        index.hnsw.efConstruction = retrieval_hnsw_ef_construction()
        index.hnsw.efSearch = retrieval_hnsw_ef_search()
        index.add(matrix)
        self._faiss = faiss
        self._index = index
        self._matrix = matrix
        self.engine = "faiss_hnsw"

    def search(self, query_vector: Sequence[float], limit: int) -> list[tuple[int, float]]:
        if not self.vectors or limit <= 0:
            return []
        limit = min(limit, len(self.vectors))
        if self._index is not None:
            import numpy as np

            query = np.asarray([query_vector], dtype="float32")
            self._faiss.normalize_L2(query)
            scores, indices = self._index.search(query, limit)
            return [
                (int(index), float(score))
                for index, score in zip(indices[0], scores[0], strict=True)
                if index >= 0
            ]

        ranked = [
            (
                index,
                sum(left * right for left, right in zip(query_vector, vector, strict=True)),
            )
            for index, vector in enumerate(self.vectors)
        ]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked[:limit]
