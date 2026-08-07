from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any

from app.knowledge.models import KnowledgeDocument
from app.knowledge.retriever import CategoryKnowledgeRetriever, KnowledgeMode


@dataclass(frozen=True)
class KnowledgeEvaluationCase:
    case_id: str
    query: str
    category_key: str
    relevant_doc_ids: frozenset[str]


_THEME_QUERIES = {
    "选购要点": "购买这个品类时应该优先关注哪些硬约束和软偏好",
    "材质说明": "常见材质有哪些，各自需要关注什么",
    "价格分层": "价格区间和预算档位如何划分",
    "排序建议": "搜索结果应该根据哪些因素排序推荐",
    "跨平台比价": "跨平台同款商品怎样比较到手价",
}


def build_knowledge_evaluation_cases(
    documents: list[KnowledgeDocument],
) -> list[KnowledgeEvaluationCase]:
    cases: list[KnowledgeEvaluationCase] = []
    for document in documents:
        intent = _THEME_QUERIES.get(document.title, document.title)
        cases.append(
            KnowledgeEvaluationCase(
                case_id=f"knowledge-{document.doc_id}",
                query=f"{document.category} {intent}",
                category_key=document.category_key,
                relevant_doc_ids=frozenset({document.doc_id}),
            )
        )
    return cases


def _dcg(relevance: list[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevance))


def evaluate_knowledge_retriever(
    retriever: CategoryKnowledgeRetriever,
    cases: list[KnowledgeEvaluationCase],
    *,
    modes: tuple[KnowledgeMode, ...] = ("lexical", "vector", "hybrid"),
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    if not cases:
        raise ValueError("知识检索评测用例为空")
    report: dict[str, Any] = {
        "case_count": len(cases),
        "k_values": list(k_values),
        "embedding_provider": retriever.provider.name,
        "embedding_dimension": retriever.provider.dimension,
        "modes": {},
    }
    maximum_k = max(k_values)
    for mode in modes:
        rows: list[dict[str, dict[int, float]]] = []
        for case in cases:
            result = retriever.search(
                query=case.query,
                category_key=case.category_key,
                top_k=maximum_k,
                mode=mode,
            )
            ranked_ids = [hit.document.doc_id for hit in result.hits]
            metrics: dict[str, dict[int, float]] = {
                "recall": {},
                "hit_rate": {},
                "mrr": {},
                "ndcg": {},
            }
            for k in k_values:
                top_ids = ranked_ids[:k]
                flags = [1 if doc_id in case.relevant_doc_ids else 0 for doc_id in top_ids]
                relevant_count = sum(flags)
                metrics["recall"][k] = relevant_count / len(case.relevant_doc_ids)
                metrics["hit_rate"][k] = 1.0 if relevant_count else 0.0
                first_rank = next(
                    (index for index, flag in enumerate(flags, start=1) if flag),
                    None,
                )
                metrics["mrr"][k] = 1.0 / first_rank if first_rank else 0.0
                ideal = [1] * min(k, len(case.relevant_doc_ids))
                ideal_dcg = _dcg(ideal)
                metrics["ndcg"][k] = _dcg(flags) / ideal_dcg if ideal_dcg else 0.0
            rows.append(metrics)
        report["modes"][mode] = {
            metric: {
                f"@{k}": round(mean(row[metric][k] for row in rows), 6)
                for k in k_values
            }
            for metric in ("recall", "hit_rate", "mrr", "ndcg")
        }
    return report
