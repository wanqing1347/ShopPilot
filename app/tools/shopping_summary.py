from __future__ import annotations

import json
import re
import time

from pydantic import BaseModel, Field

from app.agent.llm import get_llm
from app.api.monitor import monitor
from app.models import CategoryInsightOutput, ItemPickerOutput, QueryPlan, ShoppingSummaryOutput


class _PreferenceDraft(BaseModel):
    learned_preferences: list[str] = Field(default_factory=list)


_PREFERENCE_PROMPT = """你只负责从购物请求中提取值得跨会话保留的稳定偏好。

规则：
- 只能从 user_query 和 soft_preferences 中提取或轻微规范化，禁止新增偏好。
- 不保存预算、临时平台、具体商品、价格、一次性数量或本次配送要求。
- 不生成购物清单，不评价商品，不补充任何事实。
- 没有稳定偏好时返回空列表。
"""

_TEMPORARY_PREFERENCE_RE = re.compile(
    r"\d|预算|元|amazon|amazon_jp|shopee|rakuten|ebay|lazada|shein|walmart|public_demo|演示商城|模拟电商目录|本次|这次|今天|明天|具体商品",
    re.IGNORECASE,
)


def _escape_markdown(value: str) -> str:
    return " ".join(value.replace("|", "\\|").split())


_PLATFORM_NAMES = {
    "amazon": "Amazon",
    "amazon_jp": "Amazon JP",
    "shopee": "Shopee",
    "aliexpress": "AliExpress",
    "ebay": "eBay",
    "lazada": "Lazada",
    "rakuten": "Rakuten",
    "shein": "SHEIN",
    "walmart": "Walmart",
}


def _platform_label(item: object) -> str:
    platform = str(getattr(item, "platform", ""))
    return _PLATFORM_NAMES.get(platform, "公开演示商城" if platform == "public_demo" else platform)


def _source_label(item) -> str:
    if item.verification_status == "live":
        return f"{item.provider or item.data_origin}（实时）"
    if item.verification_status == "cached":
        return f"{item.provider or item.data_origin}（缓存）"
    if item.verification_status == "merchant":
        return "商家授权数据"
    if item.verification_status == "user_supplied":
        return "用户提供"
    if item.verification_status == "public_demo":
        return "模拟电商商品目录（非真实交易平台）"
    return "离线缓存商品快照"


def _stable_preferences(plan: QueryPlan, model_values: list[str]) -> list[str]:
    source_text = " ".join([plan.original_query, *plan.soft_preferences])
    candidates = [*plan.soft_preferences, *model_values]
    result: list[str] = []
    for value in candidates:
        normalized = " ".join(value.split()).strip()
        if not normalized or _TEMPORARY_PREFERENCE_RE.search(normalized):
            continue
        if normalized not in plan.soft_preferences and normalized not in source_text:
            continue
        if normalized not in result:
            result.append(normalized)
    return result[:20]


def _render_summary(picker: ItemPickerOutput, plan: QueryPlan) -> str:
    lines = ["# 推荐购物清单", "", f"**需求品类**：{_escape_markdown(plan.category)}"]
    if plan.budget_cny is not None:
        lines.append(f"**预算上限**：¥{plan.budget_cny:.2f}")
    if plan.hard_constraints:
        lines.append("**硬约束**：" + "；".join(_escape_markdown(value) for value in plan.hard_constraints))
    if plan.soft_preferences:
        lines.append("**软偏好**：" + "；".join(_escape_markdown(value) for value in plan.soft_preferences))
    lines.append("")

    if picker.picks:
        lines.extend(
            [
                "| 排名 | 平台 | 商品 | 到手价（¥） | 数据来源 | 工具推荐理由 |",
                "| ---: | --- | --- | ---: | --- | --- |",
            ]
        )
        for rank, item in enumerate(picker.picks, start=1):
            reasons = "；".join(item.reasons) or "通过当前精排规则"
            if item.flags:
                reasons += "；注意：" + "、".join(item.flags)
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(rank),
                        _escape_markdown(_platform_label(item)),
                        _escape_markdown(item.title),
                        f"{item.landed_cny:.2f}",
                        _escape_markdown(_source_label(item)),
                        _escape_markdown(reasons),
                    ]
                )
                + " |"
            )
        if plan.budget_cny is not None:
            within_budget = sum(item.landed_cny <= plan.budget_cny for item in picker.picks)
            lines.extend(
                [
                    "",
                    f"预算检查：{within_budget}/{len(picker.picks)} 个入选商品不超过 ¥{plan.budget_cny:.2f}。",
                ]
            )
    else:
        lines.extend(["当前没有通过全部硬约束的候选商品。"])

    if picker.rejected_brief:
        lines.extend(
            [
                "",
                "## 淘汰与限制",
                *[f"- {_escape_markdown(value)}" for value in picker.rejected_brief],
            ]
        )

    lines.extend(
        [
            "",
            "> 当前商品可能来自 Amazon/Walmart/eBay 的在线查询或同名离线缓存回退；"
            "离线结果并非实时库存或结算报价，"
            "运费、汇率与税费为演示估算。",
        ]
    )
    return "\n".join(lines)


def _append_grounded_evidence(
    final_text: str,
    insight: CategoryInsightOutput | None,
) -> str:
    if insight is None or insight.answer_mode != "llm_grounded" or not insight.grounded_answer:
        return final_text
    return (
        final_text.rstrip()
        + "\n\n### 品类依据（项目知识库）\n"
        + insight.grounded_answer.strip()
        + "\n\n> 上述品类依据来自项目知识文档与离线快照统计，不代表实时平台市场。"
    )


async def shopping_summary(
    picker: ItemPickerOutput,
    plan: QueryPlan,
    insight: CategoryInsightOutput | None = None,
) -> ShoppingSummaryOutput:
    """Render factual shopping results deterministically; use the LLM only for memory extraction."""

    await monitor.report_tool_start(
        "shopping_summary", {"picks_count": len(picker.picks)}
    )
    started = time.perf_counter()
    model_values: list[str] = []
    payload = {
        "user_query": plan.original_query,
        "soft_preferences": plan.soft_preferences,
    }
    try:
        extractor = get_llm().with_structured_output(
            _PreferenceDraft,
            method="function_calling",
        )
        raw = await extractor.ainvoke(
            [
                ("system", _PREFERENCE_PROMPT),
                ("user", json.dumps(payload, ensure_ascii=False)),
            ]
        )
        draft = raw if isinstance(raw, _PreferenceDraft) else _PreferenceDraft.model_validate(raw)
        model_values = draft.learned_preferences
    except Exception:  # noqa: BLE001 - final rendering must survive provider failures
        model_values = []

    final_text = _append_grounded_evidence(_render_summary(picker, plan), insight)
    result = ShoppingSummaryOutput(
        final_text=final_text,
        picks=picker.picks,
        learned_preferences=_stable_preferences(plan, model_values),
    )
    await monitor.report_tool_end(
        "shopping_summary", int((time.perf_counter() - started) * 1000)
    )
    return result
