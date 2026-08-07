from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from app.recall.hybrid import HybridRetriever, RetrievalHit, RetrievalMode, SearchFilters
from app.recall.ltr import LearnedReranker, RerankerMode


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    query: str
    category_key: str
    relevant_group_ids: set[str]
    split: str
    budget_cny_max: float | None = None
    preferred_material: str | None = None
    preferred_style: str | None = None
    required_feature: str | None = None


@dataclass(frozen=True)
class QueryEvaluation:
    query_id: str
    recall_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    mrr_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    hard_constraint_at_k: dict[int, float]


def load_evaluation_queries(
    path: Path,
    *,
    split: str | None = None,
) -> list[EvaluationQuery]:
    queries: list[EvaluationQuery] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Query JSON 解析失败: {path}:{line_no}: {exc}") from exc
            if split and raw.get("split") != split:
                continue
            constraints = raw.get("constraints") or {}
            queries.append(
                EvaluationQuery(
                    query_id=str(raw["query_id"]),
                    query=str(raw["query"]),
                    category_key=str(raw["category_key"]),
                    relevant_group_ids={str(value) for value in raw["relevant_group_ids"]},
                    split=str(raw.get("split") or "unknown"),
                    budget_cny_max=(
                        float(constraints["budget_cny_max"])
                        if constraints.get("budget_cny_max") is not None
                        else None
                    ),
                    preferred_material=constraints.get("preferred_material"),
                    preferred_style=constraints.get("preferred_style"),
                    required_feature=constraints.get("required_feature"),
                )
            )
    return queries


def _group_level_hits(hits: Iterable[RetrievalHit]) -> list[RetrievalHit]:
    seen: set[str] = set()
    result: list[RetrievalHit] = []
    for hit in hits:
        group_id = hit.candidate.same_group_id
        if group_id in seen:
            continue
        seen.add(group_id)
        result.append(hit)
    return result


def _dcg(relevance: list[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevance))


def _hard_constraint_satisfied(hit: RetrievalHit, query: EvaluationQuery) -> bool:
    candidate = hit.candidate
    if not candidate.is_available:
        return False
    if query.budget_cny_max is not None:
        price = candidate.landed_price_cny or candidate.price_cny
        if price is not None and price > query.budget_cny_max:
            return False
    if query.required_feature:
        features = {
            str(value)
            for value in candidate.attributes.get("features", [])
        }
        if query.required_feature not in features:
            return False
    return True


def evaluate_query(
    retriever: HybridRetriever,
    query: EvaluationQuery,
    *,
    mode: RetrievalMode,
    k_values: tuple[int, ...],
    reranker: RerankerMode = "rules",
    learned_model: LearnedReranker | None = None,
) -> QueryEvaluation:
    maximum_k = max(k_values)
    preferences = [
        value
        for value in [
            query.preferred_material,
            query.preferred_style,
            query.required_feature,
        ]
        if value
    ]
    result = retriever.search(
        query=query.query,
        filters=SearchFilters(
            category_key=query.category_key,
            max_landed_price_cny=query.budget_cny_max,
        ),
        top_k=max(maximum_k * 4, maximum_k),
        user_preferences=preferences,
        mode=mode,
        reranker=reranker,
        learned_model=learned_model,
    )
    group_hits = _group_level_hits(result.hits)
    ranked_groups = [hit.candidate.same_group_id for hit in group_hits]
    evaluations: dict[str, dict[int, float]] = {
        "recall": {},
        "hit_rate": {},
        "mrr": {},
        "ndcg": {},
        "hard": {},
    }
    for k in k_values:
        top_hits = group_hits[:k]
        top_groups = ranked_groups[:k]
        relevant_flags = [
            1 if group_id in query.relevant_group_ids else 0
            for group_id in top_groups
        ]
        relevant_count = sum(relevant_flags)
        evaluations["recall"][k] = relevant_count / max(
            1, len(query.relevant_group_ids)
        )
        evaluations["hit_rate"][k] = 1.0 if relevant_count else 0.0
        first_rank = next(
            (index for index, flag in enumerate(relevant_flags, start=1) if flag),
            None,
        )
        evaluations["mrr"][k] = 1.0 / first_rank if first_rank else 0.0
        ideal = [1] * min(k, len(query.relevant_group_ids))
        ideal_dcg = _dcg(ideal)
        evaluations["ndcg"][k] = _dcg(relevant_flags) / ideal_dcg if ideal_dcg else 0.0
        evaluations["hard"][k] = (
            mean(_hard_constraint_satisfied(hit, query) for hit in top_hits)
            if top_hits
            else 0.0
        )
    return QueryEvaluation(
        query_id=query.query_id,
        recall_at_k=evaluations["recall"],
        hit_rate_at_k=evaluations["hit_rate"],
        mrr_at_k=evaluations["mrr"],
        ndcg_at_k=evaluations["ndcg"],
        hard_constraint_at_k=evaluations["hard"],
    )


def evaluate_retriever(
    retriever: HybridRetriever,
    queries: list[EvaluationQuery],
    *,
    modes: tuple[RetrievalMode, ...] = ("lexical", "vector", "hybrid"),
    k_values: tuple[int, ...] = (5, 10, 20),
    variants: dict[
        str,
        tuple[RetrievalMode, RerankerMode, LearnedReranker | None],
    ] | None = None,
) -> dict[str, Any]:
    if not queries:
        raise ValueError("评测 Query 为空")
    report: dict[str, Any] = {
        "query_count": len(queries),
        "splits": sorted({query.split for query in queries}),
        "k_values": list(k_values),
        "embedding_provider": retriever.provider.name,
        "embedding_model": getattr(retriever.provider, "model_name", None),
        "embedding_dimension": retriever.provider.dimension,
        "modes": {},
    }
    selected_variants = variants or {
        mode: (mode, "rules", None)
        for mode in modes
    }
    for variant_name, (mode, reranker, learned_model) in selected_variants.items():
        rows = [
            evaluate_query(
                retriever,
                query,
                mode=mode,
                k_values=k_values,
                reranker=reranker,
                learned_model=learned_model,
            )
            for query in queries
        ]
        mode_metrics: dict[str, dict[str, float]] = {}
        for metric_name, attribute in (
            ("recall", "recall_at_k"),
            ("hit_rate", "hit_rate_at_k"),
            ("mrr", "mrr_at_k"),
            ("ndcg", "ndcg_at_k"),
            ("hard_constraint_satisfaction", "hard_constraint_at_k"),
        ):
            mode_metrics[metric_name] = {
                f"@{k}": round(
                    mean(getattr(row, attribute)[k] for row in rows),
                    6,
                )
                for k in k_values
            }
        mode_metrics["configuration"] = {
            "retrieval_mode": mode,
            "reranker": reranker,
        }
        report["modes"][variant_name] = mode_metrics
    return report
