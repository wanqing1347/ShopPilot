from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Platform = Literal[
    "amazon",
    "amazon_jp",
    "shopee",
    "aliexpress",
    "ebay",
    "rakuten",
    "walmart",
    "lazada",
    "shein",
    "public_demo",
]


class QueryPlan(BaseModel):
    original_query: str
    category: str
    category_key: str | None = None
    budget_cny: float | None = None
    platforms: list[Platform]
    hard_constraints: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    """Canonical product contract shared by the dataset and Agent runtime."""

    schema_version: int = 2
    item_id: str
    same_group_id: str
    platform: Platform
    title: str
    title_en: str | None = None
    description: str = ""
    brand: str | None = None
    category_key: str
    category_path: list[str] = Field(default_factory=list)

    price: float
    currency: str
    price_cny: float | None = None
    original_price_cny: float | None = None
    shipping_cny: float | None = None
    estimated_tax_cny: float | None = None
    landed_price_cny: float | None = None

    rating: float | None = None
    review_count: int | None = None
    sales: int | None = None
    is_available: bool = True
    stock: int | None = None
    delivery_days: int | None = None
    image_url: str | None = None
    attributes: dict[str, object] = Field(default_factory=dict)

    source_updated_at: str | None = None
    ingested_at: str | None = None
    quality_grade: str | None = None
    data_origin: str = "synthetic"
    provider: str | None = None
    source_url: str | None = None
    retrieved_at: str | None = None
    expires_at: str | None = None
    verification_status: Literal[
        "live",
        "cached",
        "merchant",
        "user_supplied",
        "public_demo",
        "synthetic",
    ] = "synthetic"


class ItemSearchOutput(BaseModel):
    platform: Platform
    candidates: list[Candidate]
    total_recall: int
    truncated: bool
    retrieval: dict[str, object] = Field(default_factory=dict)


class PricePoint(BaseModel):
    item_id: str
    same_group_id: str | None = None
    platform: Platform
    title: str
    price_local: float
    currency_local: str
    price_cny: float
    shipping_cny: float | None = None
    delivery_days: int | None = None
    rating: float | None = None
    sales: int | None = None
    attributes: dict[str, object] = Field(default_factory=dict)
    note: str | None = None
    data_origin: str = "synthetic"
    provider: str | None = None
    source_url: str | None = None
    retrieved_at: str | None = None
    verification_status: str = "synthetic"


class PriceCompareOutput(BaseModel):
    base_currency: str = "CNY"
    ranked: list[PricePoint]
    cheapest_per_platform: dict[str, str]


class LandedCost(BaseModel):
    item_id: str
    same_group_id: str | None = None
    platform: Platform
    title: str
    price_cny: float
    shipping_cny: float
    duty_cny: float
    landed_cny: float
    eta_days: int
    duty_tier: Literal["免征", "标准", "高税"]
    rating: float | None = None
    sales: int | None = None
    attributes: dict[str, object] = Field(default_factory=dict)
    data_origin: str = "synthetic"
    provider: str | None = None
    source_url: str | None = None
    retrieved_at: str | None = None
    verification_status: str = "synthetic"


class ShippingCalcOutput(BaseModel):
    destination: str
    items: list[LandedCost]


class Bestseller(BaseModel):
    name: str
    typical_price_cny: float
    why_popular: str


class AttributeDist(BaseModel):
    name: str
    distribution: dict[str, float]


class PriceTier(BaseModel):
    tier: Literal["budget", "mid", "premium"]
    range_cny: tuple[float, float]
    notes: str


class KnowledgeCitation(BaseModel):
    doc_id: str
    category_key: str
    title: str
    snippet: str
    source: str
    updated_at: str | None = None
    score: float = 0.0
    bm25_rank: int | None = None
    vector_rank: int | None = None


class GroundedClaim(BaseModel):
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    support_score: float = 0.0


class CategoryInsightOutput(BaseModel):
    category: str
    category_key: str | None = None
    components: list[str] = Field(default_factory=list)
    bestsellers: list[Bestseller] = Field(default_factory=list)
    attributes: list[AttributeDist] = Field(default_factory=list)
    price_tiers: list[PriceTier] = Field(default_factory=list)
    citations: list[KnowledgeCitation] = Field(default_factory=list)
    evidence_summary: str = ""
    grounded_answer: str = ""
    grounded_claims: list[GroundedClaim] = Field(default_factory=list)
    citation_validation: dict[str, object] = Field(default_factory=dict)
    answer_mode: Literal["deterministic_evidence", "llm_grounded", "no_evidence"] = (
        "deterministic_evidence"
    )
    retrieval: dict[str, object] = Field(default_factory=dict)
    confidence: float = 0.0


class PickedItem(BaseModel):
    item_id: str
    same_group_id: str | None = None
    platform: Platform
    title: str
    landed_cny: float
    score: float
    reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    data_origin: str = "synthetic"
    provider: str | None = None
    source_url: str | None = None
    retrieved_at: str | None = None
    verification_status: str = "synthetic"


class ItemPickerOutput(BaseModel):
    picks: list[PickedItem]
    rejected_brief: list[str] = Field(default_factory=list)


class ShoppingSummaryOutput(BaseModel):
    final_text: str
    picks: list[PickedItem]
    learned_preferences: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    status: Literal["ok", "timeout", "error"]
    thread_id: str
    plan: QueryPlan | None = None
    final: str | None = None
    output_files: list[str] = Field(default_factory=list)
    error: str | None = None
