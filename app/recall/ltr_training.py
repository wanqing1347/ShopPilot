from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.recall.catalog import dataset_root
from app.recall.evaluation import EvaluationQuery, evaluate_retriever, load_evaluation_queries
from app.recall.hybrid import HybridRetriever, SearchFilters, candidate_text
from app.recall.ltr import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    LearnedReranker,
    RerankSignals,
    reranker_model_path,
)


@dataclass(frozen=True)
class TrainingExamples:
    pair_differences: list[list[float]]
    labels: list[int]
    feature_rows: list[list[float]]
    query_count: int
    positive_count: int
    negative_count: int


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败: {path}:{line_no}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"JSONL 行必须是对象: {path}:{line_no}")
            rows.append(raw)
    return rows


def interaction_priors(
    interactions_path: Path,
    *,
    allowed_query_ids: set[str],
) -> dict[str, tuple[float, float, float, float]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in _load_jsonl(interactions_path):
        if str(row.get("query_id")) not in allowed_query_ids:
            continue
        group_id = str(row.get("same_group_id") or "")
        event_type = str(row.get("event_type") or "")
        if group_id and event_type:
            counts[group_id][event_type] += 1

    priors: dict[str, tuple[float, float, float, float]] = {}
    for group_id, events in counts.items():
        impressions = max(1, events["impression"])
        priors[group_id] = (
            (events["click"] + 1.0) / (impressions + 4.0),
            (events["favorite"] + 0.5) / (impressions + 6.0),
            (events["purchase"] + 0.25) / (impressions + 8.0),
            (events["dislike"] + 0.25) / (impressions + 8.0),
        )
    return priors


def _feature_extractor(
    retriever: HybridRetriever,
    priors: dict[str, tuple[float, float, float, float]],
) -> LearnedReranker:
    return LearnedReranker(
        model_version=1,
        feature_version=FEATURE_VERSION,
        feature_names=FEATURE_NAMES,
        weights=tuple(0.0 for _ in FEATURE_NAMES),
        scales=tuple(1.0 for _ in FEATURE_NAMES),
        embedding_provider=retriever.provider.name,
        embedding_model=getattr(retriever.provider, "model_name", None),
        group_priors=priors,
        training={},
    )


def _query_preferences(query: EvaluationQuery) -> list[str]:
    return [
        value
        for value in (
            query.preferred_material,
            query.preferred_style,
            query.required_feature,
        )
        if value
    ]


def build_training_examples(
    retriever: HybridRetriever,
    queries: Iterable[EvaluationQuery],
    *,
    priors: dict[str, tuple[float, float, float, float]],
    candidate_pool: int = 80,
    negatives_per_positive: int = 16,
) -> TrainingExamples:
    extractor = _feature_extractor(retriever, priors)
    pair_differences: list[list[float]] = []
    labels: list[int] = []
    feature_rows: list[list[float]] = []
    positive_count = 0
    negative_count = 0
    processed_queries = 0

    for query in queries:
        preferences = _query_preferences(query)
        result = retriever.search(
            query=query.query,
            filters=SearchFilters(
                category_key=query.category_key,
                max_landed_price_cny=query.budget_cny_max,
            ),
            top_k=candidate_pool,
            user_preferences=preferences,
            mode="hybrid",
            reranker="none",
        )
        # Labels are group-level. Keep the strongest platform offer for each group.
        group_hits = {}
        for hit in result.hits:
            group_hits.setdefault(hit.candidate.same_group_id, hit)
        hits = list(group_hits.values())
        positives = [
            hit
            for hit in hits
            if hit.candidate.same_group_id in query.relevant_group_ids
        ]
        negatives = [
            hit
            for hit in hits
            if hit.candidate.same_group_id not in query.relevant_group_ids
        ]
        if not positives or not negatives:
            continue

        feature_by_item: dict[str, list[float]] = {}
        for hit in hits:
            features = extractor.feature_vector(
                candidate=hit.candidate,
                query=query.query,
                user_preferences=preferences,
                budget_cny=query.budget_cny_max,
                signals=RerankSignals(
                    normalized_rrf=hit.final_score,
                    bm25_score=hit.bm25_score,
                    vector_score=hit.vector_score,
                    bm25_rank=hit.bm25_rank,
                    vector_rank=hit.vector_rank,
                    rule_score=hit.rerank_score,
                ),
                candidate_text=candidate_text(hit.candidate),
            )
            feature_by_item[hit.candidate.item_id] = features
            feature_rows.append(features)

        selected_negatives = negatives[: max(1, negatives_per_positive)]
        for positive in positives:
            positive_features = feature_by_item[positive.candidate.item_id]
            for negative in selected_negatives:
                negative_features = feature_by_item[negative.candidate.item_id]
                difference = [
                    pos - neg
                    for pos, neg in zip(
                        positive_features,
                        negative_features,
                        strict=True,
                    )
                ]
                pair_differences.append(difference)
                labels.append(1)
                pair_differences.append([-value for value in difference])
                labels.append(0)
                positive_count += 1
                negative_count += 1
        processed_queries += 1

    if not pair_differences:
        raise ValueError("没有生成可训练的 pairwise LTR 样本")
    return TrainingExamples(
        pair_differences=pair_differences,
        labels=labels,
        feature_rows=feature_rows,
        query_count=processed_queries,
        positive_count=positive_count,
        negative_count=negative_count,
    )


def _feature_scales(rows: list[list[float]]) -> tuple[float, ...]:
    import numpy as np

    matrix = np.asarray(rows, dtype="float64")
    scales = matrix.std(axis=0)
    scales = np.where(scales < 0.02, 1.0, scales)
    return tuple(float(value) for value in scales)


def fit_pairwise_model(
    retriever: HybridRetriever,
    examples: TrainingExamples,
    *,
    priors: dict[str, tuple[float, float, float, float]],
    regularization_c: float,
    training: dict[str, Any],
) -> LearnedReranker:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    scales = _feature_scales(examples.feature_rows)
    matrix = np.asarray(examples.pair_differences, dtype="float64")
    matrix = matrix / np.asarray(scales, dtype="float64")
    labels = np.asarray(examples.labels, dtype="int64")
    estimator = LogisticRegression(
        C=regularization_c,
        fit_intercept=False,
        solver="liblinear",
        max_iter=2000,
        random_state=20260805,
    )
    estimator.fit(matrix, labels)
    weights = tuple(float(value) for value in estimator.coef_[0])
    return LearnedReranker(
        model_version=1,
        feature_version=FEATURE_VERSION,
        feature_names=FEATURE_NAMES,
        weights=weights,
        scales=scales,
        embedding_provider=retriever.provider.name,
        embedding_model=getattr(retriever.provider, "model_name", None),
        group_priors=priors,
        training=training,
    )


def _selection_score(metrics: dict[str, Any]) -> float:
    return (
        metrics["ndcg"]["@10"] * 0.60
        + metrics["recall"]["@10"] * 0.30
        + metrics["hard_constraint_satisfaction"]["@10"] * 0.10
    )


def train_and_select_reranker(
    retriever: HybridRetriever,
    *,
    output_path: Path | None = None,
    c_values: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0),
    candidate_pool: int = 80,
    negatives_per_positive: int = 16,
) -> dict[str, Any]:
    root = dataset_root()
    queries_path = root / "queries.jsonl"
    interactions_path = root / "interactions.jsonl"
    train_queries = load_evaluation_queries(queries_path, split="train")
    dev_queries = load_evaluation_queries(queries_path, split="dev")
    train_ids = {query.query_id for query in train_queries}
    train_priors = interaction_priors(
        interactions_path,
        allowed_query_ids=train_ids,
    )
    train_examples = build_training_examples(
        retriever,
        train_queries,
        priors=train_priors,
        candidate_pool=candidate_pool,
        negatives_per_positive=negatives_per_positive,
    )

    trials: list[dict[str, Any]] = []
    best_model: LearnedReranker | None = None
    best_score = float("-inf")
    best_c = c_values[0]
    for regularization_c in c_values:
        model = fit_pairwise_model(
            retriever,
            train_examples,
            priors=train_priors,
            regularization_c=regularization_c,
            training={"stage": "selection", "regularization_c": regularization_c},
        )
        report = evaluate_retriever(
            retriever,
            dev_queries,
            k_values=(5, 10, 20),
            variants={
                "hybrid_ltr": ("hybrid", "learned", model),
            },
        )
        metrics = report["modes"]["hybrid_ltr"]
        score = _selection_score(metrics)
        trial = {
            "regularization_c": regularization_c,
            "selection_score": round(score, 8),
            "metrics": metrics,
        }
        trials.append(trial)
        if score > best_score:
            best_score = score
            best_model = model
            best_c = regularization_c

    if best_model is None:
        raise RuntimeError("LTR dev 选参失败")

    combined_queries = [*train_queries, *dev_queries]
    combined_ids = {query.query_id for query in combined_queries}
    combined_priors = interaction_priors(
        interactions_path,
        allowed_query_ids=combined_ids,
    )
    combined_examples = build_training_examples(
        retriever,
        combined_queries,
        priors=combined_priors,
        candidate_pool=candidate_pool,
        negatives_per_positive=negatives_per_positive,
    )
    trained_at = datetime.now(timezone.utc).isoformat()
    final_model = fit_pairwise_model(
        retriever,
        combined_examples,
        priors=combined_priors,
        regularization_c=best_c,
        training={
            "training_id": f"ltr-v1-{trained_at}",
            "trained_at": trained_at,
            "trained_splits": ["train", "dev"],
            "selection_split": "dev",
            "test_split_used": False,
            "selected_regularization_c": best_c,
            "candidate_pool": candidate_pool,
            "negatives_per_positive": negatives_per_positive,
            "train_query_count": len(train_queries),
            "dev_query_count": len(dev_queries),
            "final_query_count": combined_examples.query_count,
            "final_pair_count": len(combined_examples.labels),
            "interaction_prior_splits": ["train", "dev"],
            "selection_score": round(best_score, 8),
        },
    )
    destination = output_path or reranker_model_path()
    final_model.save(destination)
    return {
        "model_path": str(destination),
        "embedding_provider": retriever.provider.name,
        "embedding_model": getattr(retriever.provider, "model_name", None),
        "feature_names": list(FEATURE_NAMES),
        "selected_regularization_c": best_c,
        "selection_score": round(best_score, 8),
        "selection_trials": trials,
        "train_examples": {
            "query_count": train_examples.query_count,
            "pair_count": len(train_examples.labels),
        },
        "final_examples": {
            "query_count": combined_examples.query_count,
            "pair_count": len(combined_examples.labels),
        },
        "test_split_used": False,
    }
