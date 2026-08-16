from __future__ import annotations

from typing import Final

# Demonstration rates retained from the original project. Production deployments
# should replace this table with a timestamped FX provider.
FX_RATES: Final[dict[str, float]] = {
    "CNY": 1.0,
    "USD": 7.18,
    "SGD": 5.32,
    "GBP": 9.05,
    "EUR": 7.78,
    "JPY": 0.046,
    # Static demo-only rates used by the public test catalog builders.
    "CAD": 5.15,
    "INR": 0.086,
    # Static demo-only rates for the cached open marketplace dataset.
    "IDR": 0.00045,
    "MXN": 0.42,
}


def to_base(amount: float, currency: str, base: str = "CNY") -> float:
    source = currency.upper()
    target = base.upper()
    if source not in FX_RATES or target not in FX_RATES:
        raise ValueError(f"未知币种: {currency} 或 {base}")
    return amount * FX_RATES[source] / FX_RATES[target]
