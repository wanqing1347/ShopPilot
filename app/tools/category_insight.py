from __future__ import annotations

import asyncio
import math
import time
from collections import Counter, defaultdict
from statistics import median

from app.agent.settings import (
    knowledge_min_evidence,
    knowledge_synthesis_enabled,
    knowledge_top_k,
)
from app.api.monitor import monitor
from app.knowledge.retriever import get_category_knowledge_retriever
from app.knowledge.synthesis import synthesize_grounded_insight
from app.models import (
    AttributeDist,
    Bestseller,
    CategoryInsightOutput,
    KnowledgeCitation,
    PriceTier,
)
from app.recall.catalog import load_catalog


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = max(0.0, min(1.0, fraction)) * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _distribution(values: list[str], *, limit: int = 6) -> dict[str, float]:
    normalized = [value.strip() for value in values if value and value.strip()]
    if not normalized:
        return {}
    counts = Counter(normalized)
    total = sum(counts.values())
    return {
        name: round(count / total, 4)
        for name, count in counts.most_common(limit)
    }


def _catalog_statistics_citation(
    *,
    category_key: str,
    product_count: int,
    attributes: list[AttributeDist],
    price_tiers: list[PriceTier],
) -> KnowledgeCitation:
    attribute_text = "；".join(
        f"{attribute.name}高频项：" + "、".join(attribute.distribution.keys())
        for attribute in attributes
        if attribute.distribution
    )
    price_text = "；".join(
        f"{tier.tier} {tier.range_cny[0]:.2f}-{tier.range_cny[1]:.2f}元"
        for tier in price_tiers
    )
    return KnowledgeCitation(
        doc_id=f"CATALOG_STATS:{category_key}",
        category_key=category_key,
        title="当前合成商品目录统计",
        snippet=(
            f"当前可售合成商品 {product_count} 条。{attribute_text}。"
            f"到手价分层：{price_text}。这些统计不代表真实平台市场。"
        ),
        source="synthetic_catalog_statistics",
        score=1.0,
    )


def _product_statistics(category_key: str) -> tuple[
    list[str],
    list[Bestseller],
    list[AttributeDist],
    list[PriceTier],
    int,
]:
    products = [
        product
        for product in load_catalog()
        if product.category_key == category_key and product.is_available
    ]
    if not products:
        return [], [], [], [], 0

    features = [
        str(feature)
        for product in products
        for feature in product.attributes.get("features", [])
    ]
    components = [name for name, _ in Counter(features).most_common(5)]

    groups: dict[str, list] = defaultdict(list)
    for product in products:
        groups[product.same_group_id].append(product)
    ranked_groups = sorted(
        groups.values(),
        key=lambda rows: (
            sum(float(row.sales or 0) for row in rows),
            sum(float(row.rating or 0) for row in rows) / max(1, len(rows)),
        ),
        reverse=True,
    )
    bestsellers: list[Bestseller] = []
    for rows in ranked_groups[:3]:
        representative = max(
            rows,
            key=lambda row: (row.sales or 0, row.rating or 0, row.item_id),
        )
        prices = sorted(
            float(price)
            for row in rows
            if (price := row.landed_price_cny or row.price_cny) is not None
        )
        row_features = [str(value) for value in representative.attributes.get("features", [])]
        reason_parts = [
            f"跨平台累计销量 {sum(int(row.sales or 0) for row in rows)}",
            f"平均评分 {sum(float(row.rating or 0) for row in rows) / len(rows):.2f}",
        ]
        if row_features:
            reason_parts.append("功能包含" + "、".join(row_features[:2]))
        bestsellers.append(
            Bestseller(
                name=representative.title,
                typical_price_cny=round(median(prices), 2) if prices else 0.0,
                why_popular="；".join(reason_parts),
            )
        )

    materials = [str(product.attributes.get("material", "")) for product in products]
    styles = [str(product.attributes.get("style", "")) for product in products]
    feature_distribution = _distribution(features)
    attributes = [
        AttributeDist(name="材质", distribution=_distribution(materials)),
        AttributeDist(name="风格", distribution=_distribution(styles)),
    ]
    if feature_distribution:
        attributes.append(AttributeDist(name="功能", distribution=feature_distribution))

    prices = sorted(
        float(price)
        for product in products
        if (price := product.landed_price_cny or product.price_cny) is not None
    )
    low = _quantile(prices, 0.0)
    q33 = _quantile(prices, 0.33)
    q67 = _quantile(prices, 0.67)
    high = _quantile(prices, 1.0)
    price_tiers = [
        PriceTier(
            tier="budget",
            range_cny=(round(low, 2), round(q33, 2)),
            notes="当前可售合成商品到手价的低位区间",
        ),
        PriceTier(
            tier="mid",
            range_cny=(round(q33, 2), round(q67, 2)),
            notes="当前可售合成商品到手价的中位区间",
        ),
        PriceTier(
            tier="premium",
            range_cny=(round(q67, 2), round(high, 2)),
            notes="当前可售合成商品到手价的高位区间",
        ),
    ]
    return components, bestsellers, attributes, price_tiers, len(products)


async def category_insight(
    category: str,
    depth: str = "deep",
    *,
    query: str = "",
    category_key: str | None = None,
) -> CategoryInsightOutput:
    await monitor.report_tool_start(
        "category_insight",
        {
            "category": category,
            "category_key": category_key,
            "depth": depth,
            "query": query,
        },
    )
    started = time.perf_counter()
    retriever = await asyncio.to_thread(get_category_knowledge_retriever)
    evidence_query = query or f"{category} 选购要点 材质 价格 排序 跨平台比价"
    result = await asyncio.to_thread(
        retriever.search,
        query=evidence_query,
        category=category,
        category_key=category_key,
        top_k=3 if depth == "quick" else knowledge_top_k(),
        mode="hybrid",
    )

    resolved_category_key = result.diagnostics.get("resolved_category_key")
    if not isinstance(resolved_category_key, str) or not result.hits:
        duration_ms = int((time.perf_counter() - started) * 1000)
        output = CategoryInsightOutput(
            category=category,
            category_key=None,
            citations=[],
            evidence_summary="当前知识库没有该品类的可引用证据。",
            answer_mode="no_evidence",
            retrieval={**result.diagnostics, "duration_ms": duration_ms},
            confidence=0.0,
        )
        await monitor.report_knowledge_retrieval(
            category_key="",
            returned_count=0,
            evidence_sufficient=False,
            embedding_provider=str(result.diagnostics.get("embedding_provider") or "unknown"),
            vector_engine=str(result.diagnostics.get("vector_engine") or "none"),
            duration_ms=duration_ms,
        )
        await monitor.report_tool_end("category_insight", duration_ms)
        return output

    components, bestsellers, attributes, price_tiers, product_count = _product_statistics(
        resolved_category_key
    )
    knowledge_citations = [
        KnowledgeCitation(
            doc_id=hit.document.doc_id,
            category_key=hit.document.category_key,
            title=hit.document.title,
            snippet=(
                hit.document.content
                if len(hit.document.content) <= 220
                else hit.document.content[:217] + "..."
            ),
            source=hit.document.source,
            updated_at=hit.document.updated_at,
            score=hit.score,
            bm25_rank=hit.bm25_rank,
            vector_rank=hit.vector_rank,
        )
        for hit in result.hits
    ]
    evidence_sufficient = len(knowledge_citations) >= knowledge_min_evidence()
    visible_attributes = [] if depth == "quick" else attributes
    citations = [
        *knowledge_citations,
        _catalog_statistics_citation(
            category_key=resolved_category_key,
            product_count=product_count,
            attributes=visible_attributes,
            price_tiers=price_tiers,
        ),
    ]
    confidence = min(
        0.98,
        0.45
        + min(len(knowledge_citations), 5) * 0.08
        + min(product_count / 200.0, 1.0) * 0.12,
    )
    if not evidence_sufficient:
        confidence = min(confidence, 0.55)

    retrieval_duration_ms = int((time.perf_counter() - started) * 1000)
    output = CategoryInsightOutput(
        category=result.hits[0].document.category,
        category_key=resolved_category_key,
        components=components,
        bestsellers=bestsellers,
        attributes=visible_attributes,
        price_tiers=price_tiers,
        citations=citations,
        evidence_summary="\n".join(
            f"[{citation.doc_id}] {citation.title}：{citation.snippet}"
            for citation in citations
        ),
        answer_mode="deterministic_evidence" if evidence_sufficient else "no_evidence",
        retrieval={
            **result.diagnostics,
            "product_count": product_count,
            "knowledge_citation_count": len(knowledge_citations),
            "evidence_sufficient": evidence_sufficient,
            "retrieval_duration_ms": retrieval_duration_ms,
        },
        confidence=round(confidence, 4),
    )
    await monitor.report_knowledge_retrieval(
        category_key=resolved_category_key,
        returned_count=len(knowledge_citations),
        evidence_sufficient=evidence_sufficient,
        embedding_provider=str(result.diagnostics.get("embedding_provider") or "unknown"),
        vector_engine=str(result.diagnostics.get("vector_engine") or "unknown"),
        duration_ms=retrieval_duration_ms,
    )

    if evidence_sufficient and knowledge_synthesis_enabled():
        synthesis = await synthesize_grounded_insight(output, query=query or evidence_query)
        output.citation_validation = synthesis.validation
        output.retrieval["synthesis_duration_ms"] = synthesis.duration_ms
        output.retrieval["synthesis_success"] = synthesis.success
        if synthesis.success:
            output.grounded_answer = synthesis.answer
            output.grounded_claims = synthesis.claims
            output.answer_mode = "llm_grounded"
        else:
            output.retrieval["synthesis_error"] = synthesis.error or "validation_failed"
        await monitor.report_knowledge_synthesis(
            category_key=resolved_category_key,
            success=synthesis.success,
            claim_count=len(synthesis.claims),
            invalid_claim_count=int(synthesis.validation.get("invalid_claim_count") or 0),
            attempts=int(synthesis.validation.get("attempts") or 0),
            duration_ms=synthesis.duration_ms,
            fallback_reason=synthesis.error,
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    output.retrieval["duration_ms"] = duration_ms
    await monitor.report_tool_end("category_insight", duration_ms)
    return output
