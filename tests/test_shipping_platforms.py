from __future__ import annotations

import pytest

from app.models import Candidate
from app.tools.shopping_summary import _platform_label
from app.tools.shipping_calc import estimate_duty, estimate_shipping


@pytest.mark.parametrize(
    ("platform", "expected_rate"),
    [("lazada", 0.06), ("shein", 0.13), ("walmart", 0.13)],
)
def test_open_dataset_platform_duty_rules(platform: str, expected_rate: float) -> None:
    duty, tier = estimate_duty(100.0, platform)

    assert duty == round(100.0 * expected_rate, 2)
    assert tier in {"免征", "标准", "高税"}


@pytest.mark.parametrize("platform", ["lazada", "shein", "walmart"])
def test_open_dataset_platform_shipping_rules(platform: str) -> None:
    fee, eta = estimate_shipping(0.5, platform)

    assert fee > 0
    assert eta > 0


@pytest.mark.parametrize(
    ("platform", "label"),
    [("lazada", "Lazada"), ("shein", "SHEIN"), ("walmart", "Walmart")],
)
def test_open_dataset_platform_display_labels(platform: str, label: str) -> None:
    item = Candidate(
        item_id=f"{platform}:test",
        same_group_id=f"TEST:{platform}",
        platform=platform,
        title="测试商品",
        category_key="footwear",
        price=10.0,
        currency="USD",
        verification_status="cached",
    )

    assert _platform_label(item) == label
