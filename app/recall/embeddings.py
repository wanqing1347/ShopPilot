from __future__ import annotations

import hashlib
import math
from collections import Counter
from functools import lru_cache
from collections.abc import Sequence
from typing import Protocol

from app.agent.settings import (
    retrieval_embedding_dimension,
    retrieval_embedding_model,
    retrieval_embedding_provider,
    retrieval_embedding_query_prompt,
)
from app.recall.tokenizer import tokenize


class EmbeddingProvider(Protocol):
    name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]


class HashingEmbeddingProvider:
    """Dependency-free normalized feature hashing baseline.

    It provides deterministic dense vectors for local/offline development. The
    provider is intentionally replaceable by SentenceTransformers without changing
    the Faiss or retrieval layers.
    """

    name = "hashing"

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension or retrieval_embedding_dimension()
        self._query_cache: dict[str, list[float]] = {}

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        counts = Counter(tokenize(text))
        for token, frequency in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            weight = 1.0 + math.log(max(1, frequency))
            vector[index] += sign * weight
        return _normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached
        vector = self._embed(text)
        if len(self._query_cache) >= 4096:
            self._query_cache.clear()
        self._query_cache[text] = vector
        return vector


class SentenceTransformerEmbeddingProvider:
    name = "sentence_transformers"

    def __init__(
        self,
        model_name: str | None = None,
        query_prompt: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence_transformers embedding 需要安装 `.[embedding]`"
            ) from exc
        self.model_name = model_name or retrieval_embedding_model()
        self.query_prompt = (
            retrieval_embedding_query_prompt() if query_prompt is None else query_prompt
        )
        self._query_cache: dict[str, list[float]] = {}
        self._model = SentenceTransformer(self.model_name)
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = self._model.get_sentence_embedding_dimension
        dimension = get_dimension()
        if dimension is None:
            raise RuntimeError(f"无法确定 embedding 维度: {self.model_name}")
        self.dimension = int(dimension)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        encode_document = getattr(self._model, "encode_document", None)
        encoder = encode_document or self._model.encode
        vectors = encoder(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached
        encode_query = getattr(self._model, "encode_query", None)
        encoder = encode_query or self._model.encode
        query_text = f"{self.query_prompt}{text}" if self.query_prompt else text
        vector = encoder(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        result = vector.astype("float32").tolist()
        if len(self._query_cache) >= 4096:
            self._query_cache.clear()
        self._query_cache[text] = result
        return result


@lru_cache(maxsize=4)
def _cached_embedding_provider(
    provider: str,
    model_name: str,
    dimension: int,
    query_prompt: str,
) -> EmbeddingProvider:
    if provider == "sentence_transformers":
        return SentenceTransformerEmbeddingProvider(
            model_name=model_name,
            query_prompt=query_prompt,
        )
    return HashingEmbeddingProvider(dimension=dimension)


def create_embedding_provider() -> EmbeddingProvider:
    return _cached_embedding_provider(
        retrieval_embedding_provider(),
        retrieval_embedding_model(),
        retrieval_embedding_dimension(),
        retrieval_embedding_query_prompt(),
    )


def clear_embedding_provider_cache() -> None:
    _cached_embedding_provider.cache_clear()
