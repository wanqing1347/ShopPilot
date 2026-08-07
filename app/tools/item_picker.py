from __future__ import annotations

import time

from app.api.monitor import monitor
from app.models import (
    CategoryInsightOutput,
    ItemPickerOutput,
    LandedCost,
    PickedItem,
)


def _contains_plastic(item: LandedCost) -> bool:
    material = str(item.attributes.get("material", ""))
    tags = [str(x) for x in item.attributes.get("tags", [])]
    material_hit = "塑料" in material and "非塑料" not in material
    tag_hit = any(tag in {"塑料", "PVC", "PE塑料"} for tag in tags)
    return material_hit or tag_hit or item.item_id.endswith("PLASTIC")


def _score(
    item: LandedCost,
    insight: CategoryInsightOutput,
    soft_preferences: list[str],
    cheapest_price: float,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    price_ratio = cheapest_price / max(item.landed_cny, 1.0)
    score += min(price_ratio, 1.0) * 0.35
    reasons.append(f"到手价约 ¥{item.landed_cny:.2f}")

    if item.rating is not None:
        score += max(0.0, min(item.rating / 5.0, 1.0)) * 0.25
        reasons.append(f"评分 {item.rating:.1f}")

    mid_tier = next((tier for tier in insight.price_tiers if tier.tier == "mid"), None)
    if mid_tier and mid_tier.range_cny[0] <= item.landed_cny <= mid_tier.range_cny[1]:
        score += 0.12
        reasons.append("位于品类主流价格带")

    material = str(item.attributes.get("material", ""))
    style = str(item.attributes.get("style", ""))
    tags = " ".join(str(x) for x in item.attributes.get("tags", []))
    searchable = f"{material} {style} {tags}"

    if "偏好小众" in soft_preferences and any(k in searchable for k in ("小众", "手作", "复古", "中古")):
        score += 0.15
        reasons.append("款式更小众")
    if "偏好耐用" in soft_preferences and any(k in searchable for k in ("帆布", "牛津布", "耐用", "尼龙")):
        score += 0.12
        reasons.append(f"{material}更偏耐用")
    if "偏好快速到货" in soft_preferences and item.eta_days <= 12:
        score += 0.1
        reasons.append(f"预计 {item.eta_days} 天到货")
    if "偏好性价比" in soft_preferences and item.landed_cny <= cheapest_price * 1.2:
        score += 0.1
        reasons.append("接近最低到手价")
    if item.duty_tier == "免征":
        score += 0.05

    return round(score, 3), reasons[:3]


async def item_picker(
    landed: list[LandedCost],
    insight: CategoryInsightOutput,
    hard_constraints: list[str] | None = None,
    soft_preferences: list[str] | None = None,
    budget_cny: float | None = None,
    top_n: int = 3,
) -> ItemPickerOutput:
    hard_constraints = hard_constraints or []
    soft_preferences = soft_preferences or []
    await monitor.report_tool_start(
        "item_picker",
        {
            "landed_count": len(landed),
            "hard_constraints": hard_constraints,
            "soft_preferences": soft_preferences,
            "budget_cny": budget_cny,
        },
    )
    started = time.perf_counter()

    eligible: list[LandedCost] = []
    rejected: list[str] = []
    for item in landed:
        if "不要塑料" in hard_constraints and _contains_plastic(item):
            rejected.append(f"{item.item_id}：含塑料，命中硬约束")
            continue
        if budget_cny is not None and item.landed_cny > budget_cny:
            rejected.append(f"{item.item_id}：到手价 ¥{item.landed_cny:.2f} 超预算")
            continue
        eligible.append(item)

    cheapest = min((item.landed_cny for item in eligible), default=1.0)
    scored: list[PickedItem] = []
    for item in eligible:
        score, reasons = _score(item, insight, soft_preferences, cheapest)
        scored.append(
            PickedItem(
                item_id=item.item_id,
                same_group_id=item.same_group_id,
                platform=item.platform,
                title=item.title,
                landed_cny=item.landed_cny,
                score=score,
                reasons=reasons,
                data_origin=item.data_origin,
                provider=item.provider,
                source_url=item.source_url,
                retrieved_at=item.retrieved_at,
                verification_status=item.verification_status,
            )
        )
    scored.sort(key=lambda item: (-item.score, item.landed_cny))

    await monitor.report_tool_end(
        "item_picker", int((time.perf_counter() - started) * 1000)
    )
    return ItemPickerOutput(picks=scored[: max(1, min(top_n, 5))], rejected_brief=rejected[:8])
