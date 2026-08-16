from __future__ import annotations

import time
from typing import Literal

from app.api.monitor import monitor
from app.models import LandedCost, PricePoint, ShippingCalcOutput

DUTY_TABLE: dict[str, tuple[float, Literal["免征", "标准", "高税"]]] = {
    "amazon": (0.13, "标准"),
    "amazon_jp": (0.13, "标准"),
    "shopee": (0.06, "免征"),
    "aliexpress": (0.13, "标准"),
    "ebay": (0.20, "高税"),
    # Cached open-dataset platforms use the same demo-only landed-cost model.
    # These are estimates, not destination-specific customs rulings.
    "lazada": (0.06, "免征"),
    "rakuten": (0.13, "标准"),
    "shein": (0.13, "标准"),
    "walmart": (0.13, "标准"),
    "public_demo": (0.0, "免征"),
}

SHIPPING_TABLE: dict[str, list[tuple[float, float, int]]] = {
    "amazon": [(0, 85, 12), (0.5, 130, 10), (2.0, 240, 8)],
    "amazon_jp": [(0, 95, 14), (0.5, 150, 12), (2.0, 280, 10)],
    "shopee": [(0, 35, 9), (0.5, 60, 9), (2.0, 120, 7)],
    "aliexpress": [(0, 20, 25), (0.5, 40, 22), (2.0, 90, 18)],
    "ebay": [(0, 90, 14), (0.5, 150, 12), (2.0, 300, 10)],
    "lazada": [(0, 35, 12), (0.5, 60, 10), (2.0, 120, 8)],
    "rakuten": [(0, 95, 14), (0.5, 150, 12), (2.0, 280, 10)],
    "shein": [(0, 30, 10), (0.5, 55, 9), (2.0, 110, 7)],
    "walmart": [(0, 85, 14), (0.5, 130, 12), (2.0, 240, 10)],
    "public_demo": [(0, 0, 0)],
}


def estimate_duty(price_cny: float, platform: str) -> tuple[float, str]:
    rate, tier = DUTY_TABLE.get(platform, (0.13, "标准"))
    return round(price_cny * rate, 2), tier


def estimate_shipping(weight_kg: float, platform: str) -> tuple[float, int]:
    table = SHIPPING_TABLE.get(platform, SHIPPING_TABLE["amazon"])
    fee, eta = table[0][1], table[0][2]
    for min_weight, candidate_fee, candidate_eta in table:
        if weight_kg >= min_weight:
            fee, eta = candidate_fee, candidate_eta
    return fee, eta


async def shipping_calc(
    points: list[PricePoint],
    destination: str = "CN",
) -> ShippingCalcOutput:
    await monitor.report_tool_start(
        "shipping_calc", {"items_count": len(points), "destination": destination}
    )
    started = time.perf_counter()
    landed: list[LandedCost] = []

    for point in points:
        raw_weight = point.attributes.get("weight_kg", 0.5)
        weight = float(raw_weight) if isinstance(raw_weight, (int, float)) else 0.5
        estimated_shipping, estimated_eta = estimate_shipping(weight, point.platform)
        shipping_cny = (
            point.shipping_cny
            if point.shipping_cny is not None
            else estimated_shipping
        )
        eta = (
            point.delivery_days
            if point.delivery_days is not None
            else estimated_eta
        )
        duty_cny, duty_tier = estimate_duty(point.price_cny, point.platform)
        landed.append(
            LandedCost(
                item_id=point.item_id,
                same_group_id=point.same_group_id,
                platform=point.platform,
                title=point.title,
                price_cny=point.price_cny,
                shipping_cny=shipping_cny,
                duty_cny=duty_cny,
                landed_cny=round(point.price_cny + shipping_cny + duty_cny, 2),
                eta_days=eta,
                duty_tier=duty_tier,
                rating=point.rating,
                sales=point.sales,
                attributes=point.attributes,
                data_origin=point.data_origin,
                provider=point.provider,
                source_url=point.source_url,
                retrieved_at=point.retrieved_at,
                verification_status=point.verification_status,
            )
        )

    landed.sort(key=lambda item: item.landed_cny)
    await monitor.report_tool_end(
        "shipping_calc", int((time.perf_counter() - started) * 1000)
    )
    return ShippingCalcOutput(destination=destination, items=landed)
