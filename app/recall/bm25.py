from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


class BM25Index:
    """Small deterministic BM25Okapi implementation for the offline catalog."""

    def __init__(
        self,
        documents: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(document) for document in documents]
        self.lengths = [sum(counter.values()) for counter in self.term_frequencies]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for counter in self.term_frequencies:
            document_frequency.update(counter.keys())
        count = len(self.term_frequencies)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        if not self.term_frequencies:
            return []
        query_counts = Counter(query_tokens)
        scores = [0.0] * len(self.term_frequencies)
        average_length = max(self.average_length, 1.0)
        for index, frequencies in enumerate(self.term_frequencies):
            document_length = self.lengths[index]
            norm = self.k1 * (
                1.0 - self.b + self.b * document_length / average_length
            )
            score = 0.0
            for term, query_frequency in query_counts.items():
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                denominator = term_frequency + norm
                score += (
                    self.idf.get(term, 0.0)
                    * term_frequency
                    * (self.k1 + 1.0)
                    / denominator
                    * (1.0 + math.log1p(query_frequency - 1))
                )
            scores[index] = score
        return scores

    def rank(self, query_tokens: Sequence[str], limit: int) -> list[tuple[int, float]]:
        ranked = [
            (index, score)
            for index, score in enumerate(self.scores(query_tokens))
            if score > 0.0
        ]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked[: max(0, limit)]
