from __future__ import annotations

from app.agent.llm import get_llm
from app.models import QueryPlan

_PLANNER_PROMPT = """你是 ShopPilot 的购物需求 Planner。

把用户原始请求转换成 QueryPlan，严格遵守：
- original_query 必须保留用户原文。
- category 使用最准确、简洁的中文商品品类。
- category_key 优先使用稳定键：travel_storage、backpack、keyboard、headphones、thermos、coffee_cup、books、beauty、fragrances、furniture、groceries、home_decoration、kitchen_accessories、laptops、apparel、footwear、watches、mobile_accessories、motorcycle、skincare、smartphones、sports_accessories、sunglasses、tablets、vehicles、bags、jewellery、electronics、lighting、tools、collectibles、miscellaneous；无法判断时为 null。
- 支持平台 amazon、walmart、ebay、public_demo；amazon、walmart、ebay 均使用本地离线缓存，结果不代表实时官方库存或结算报价。
- “演示商城”“公开演示商城”“模拟电商目录”统一映射为 public_demo。
- 用户未指定平台时，platforms 必须包含 amazon、walmart、ebay；只有用户明确提到公开演示商城或模拟电商目录时才额外加入 public_demo。
- 用户指定平台时只保留指定平台。
- 金额统一理解为人民币预算并写入 budget_cny；未提供预算时为 null。
- 否定、排除、预算上限等写入 hard_constraints。
- 风格、耐用、时效、性价比等倾向写入 soft_preferences。
- 不要补充用户没有表达的约束或偏好。
"""


async def plan_query(query: str) -> QueryPlan:
    """Use the configured LLM to produce the structured shopping plan."""

    planner = get_llm().with_structured_output(
        QueryPlan,
        method="function_calling",
    )
    raw = await planner.ainvoke(
        [
            ("system", _PLANNER_PROMPT),
            ("user", query),
        ]
    )
    plan = raw if isinstance(raw, QueryPlan) else QueryPlan.model_validate(raw)
    if plan.original_query != query:
        plan = plan.model_copy(update={"original_query": query})
    return plan
