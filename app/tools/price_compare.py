from __future__ import annotations

import time

from app.api.monitor import monitor
from app.models import Candidate, PriceCompareOutput, PricePoint
from app.utils.currency import to_base


def _pack_note(candidate: Candidate) -> str | None:
    pack_size = candidate.attributes.get("pack_size")
    if isinstance(pack_size, int) and pack_size > 1:
        unit = round(candidate.price / pack_size, 2)
        return f"一套 {pack_size} 件，标价折合单件 {unit} {candidate.currency}"
    return None


async def price_compare(
    candidates: list[Candidate],
    base_currency: str = "CNY",
    top_n: int = 12,
) -> PriceCompareOutput:
    candidates = candidates[:100]
    top_n = max(1, min(top_n, 30))
    await monitor.report_tool_start(
        "price_compare",
        {"candidates_count": len(candidates), "base_currency": base_currency},
    )
    started = time.perf_counter()

    points: list[PricePoint] = []
    for candidate in candidates:
        try:
            price_base = to_base(candidate.price, candidate.currency, base_currency)
        except ValueError:
            continue
        points.append(
            PricePoint(
                item_id=candidate.item_id,
                same_group_id=candidate.same_group_id,
                platform=candidate.platform,
                title=candidate.title,
                price_local=candidate.price,
                currency_local=candidate.currency,
                price_cny=round(price_base, 2),
                shipping_cny=candidate.shipping_cny,
                delivery_days=candidate.delivery_days,
                rating=candidate.rating,
                sales=candidate.sales,
                attributes=candidate.attributes,
                note=_pack_note(candidate),
                data_origin=candidate.data_origin,
                provider=candidate.provider,
                source_url=candidate.source_url,
                retrieved_at=candidate.retrieved_at,
                verification_status=candidate.verification_status,
            )
        )

    points.sort(key=lambda point: point.price_cny)
    cheapest: dict[str, str] = {}
    for point in points:
        cheapest.setdefault(point.platform, point.item_id)

    await monitor.report_tool_end(
        "price_compare", int((time.perf_counter() - started) * 1000)
    )
    return PriceCompareOutput(
        base_currency=base_currency,
        ranked=points[:top_n],
        cheapest_per_platform=cheapest,
    )
