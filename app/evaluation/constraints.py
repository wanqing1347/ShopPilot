from __future__ import annotations

import re
from typing import Any

from app.evaluation.models import RuleCheck


def _plan(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("plan")
    return value if isinstance(value, dict) else {}


def _picks(result: dict[str, Any]) -> list[dict[str, Any]]:
    picker = result.get("picker")
    if not isinstance(picker, dict):
        return []
    raw = picker.get("picks") or []
    return [item for item in raw if isinstance(item, dict)]


def _candidate_lookup(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for output in result.get("search_outputs") or []:
        if not isinstance(output, dict):
            continue
        for candidate in output.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            item_id = str(candidate.get("item_id") or "")
            if item_id:
                lookup[item_id] = candidate
    return lookup


def _negative_terms(constraints: list[str]) -> list[str]:
    terms: list[str] = []
    patterns = (
        re.compile(r"(?:不要|不含|排除|避免|拒绝)(.+)"),
        re.compile(r"(.+?)(?:禁用|不可接受)"),
    )
    for raw in constraints:
        normalized = str(raw).strip()
        if not normalized or "预算" in normalized:
            continue
        for pattern in patterns:
            match = pattern.search(normalized)
            if not match:
                continue
            term = match.group(1).strip(" ，,。.;；:：")
            if term and len(term) <= 40:
                terms.append(term)
            break
    return list(dict.fromkeys(terms))


def _candidate_haystack(candidate: dict[str, Any]) -> str:
    attributes = candidate.get("attributes")
    attrs = attributes if isinstance(attributes, dict) else {}
    values: list[Any] = [
        candidate.get("title"),
        candidate.get("description"),
        attrs.get("material"),
        attrs.get("style"),
        attrs.get("color"),
        *(attrs.get("features") or []),
        *(attrs.get("tags") or []),
    ]
    return " ".join(str(value) for value in values if value).lower()


def evaluate_hard_constraints(result: dict[str, Any]) -> list[RuleCheck]:
    plan = _plan(result)
    picks = _picks(result)
    terminal_tool = str(result.get("terminal_tool") or "")
    is_shopping_result = terminal_tool == "shopping_summary" or bool(plan)
    checks: list[RuleCheck] = []

    if not is_shopping_result:
        checks.append(
            RuleCheck(
                rule="C004_PICKS_PRESENT",
                title="购物结果存在",
                section="hard_constraints",
                severity="error",
                status="skipped",
                message="当前任务不是购物结果，跳过商品约束检查。",
                max_points=5,
            )
        )
    else:
        checks.append(
            RuleCheck(
                rule="C004_PICKS_PRESENT",
                title="购物结果存在",
                section="hard_constraints",
                severity="error",
                status="pass" if picks else "fail",
                message=(
                    f"最终返回 {len(picks)} 个推荐商品。"
                    if picks
                    else "shopping_summary 已结束，但 picker.picks 为空。"
                ),
                max_points=5,
                details={"pick_count": len(picks)},
            )
        )

    budget = plan.get("budget_cny")
    if budget is None or not picks:
        checks.append(
            RuleCheck(
                rule="C001_BUDGET",
                title="预算硬约束",
                section="hard_constraints",
                severity="error",
                status="skipped",
                message="没有预算上限或没有最终商品，跳过预算检查。",
                max_points=8,
            )
        )
    else:
        try:
            budget_value = float(budget)
        except (TypeError, ValueError):
            budget_value = -1.0
        violations: list[dict[str, Any]] = []
        for pick in picks:
            landed = pick.get("landed_cny")
            if landed is None:
                violations.append(
                    {"item_id": pick.get("item_id"), "reason": "缺少 landed_cny"}
                )
                continue
            try:
                landed_value = float(landed)
            except (TypeError, ValueError):
                violations.append(
                    {"item_id": pick.get("item_id"), "reason": "landed_cny 非数值"}
                )
                continue
            if budget_value < 0 or landed_value > budget_value + 1e-6:
                violations.append(
                    {
                        "item_id": pick.get("item_id"),
                        "landed_cny": landed_value,
                        "budget_cny": budget_value,
                    }
                )
        checks.append(
            RuleCheck(
                rule="C001_BUDGET",
                title="预算硬约束",
                section="hard_constraints",
                severity="error",
                status="fail" if violations else "pass",
                message=(
                    f"发现 {len(violations)} 个商品违反预算或缺少到手价。"
                    if violations
                    else f"{len(picks)}/{len(picks)} 个商品均不超过 ¥{budget_value:.2f}。"
                ),
                max_points=8,
                details={"violations": violations, "budget_cny": budget_value},
            )
        )

    platforms = [str(value) for value in plan.get("platforms") or [] if value]
    if not platforms or not picks:
        checks.append(
            RuleCheck(
                rule="C002_PLATFORM",
                title="平台范围约束",
                section="hard_constraints",
                severity="error",
                status="skipped",
                message="没有平台范围或没有最终商品，跳过平台检查。",
                max_points=5,
            )
        )
    else:
        allowed = set(platforms)
        violations = [
            {
                "item_id": pick.get("item_id"),
                "platform": pick.get("platform"),
            }
            for pick in picks
            if str(pick.get("platform") or "") not in allowed
        ]
        checks.append(
            RuleCheck(
                rule="C002_PLATFORM",
                title="平台范围约束",
                section="hard_constraints",
                severity="error",
                status="fail" if violations else "pass",
                message=(
                    f"发现 {len(violations)} 个商品来自计划范围外的平台。"
                    if violations
                    else "最终商品平台均在 planner 给出的允许范围内。"
                ),
                max_points=5,
                details={"allowed_platforms": platforms, "violations": violations},
            )
        )

    hard_constraints = [str(value) for value in plan.get("hard_constraints") or []]
    exclusions = _negative_terms(hard_constraints)
    if not exclusions or not picks:
        checks.append(
            RuleCheck(
                rule="C003_EXCLUSIONS",
                title="排除类硬约束",
                section="hard_constraints",
                severity="error",
                status="skipped",
                message="没有可程序化识别的排除项或没有最终商品。",
                max_points=7,
                details={"hard_constraints": hard_constraints},
            )
        )
    else:
        candidates = _candidate_lookup(result)
        missing: list[str] = []
        violations: list[dict[str, Any]] = []
        for pick in picks:
            item_id = str(pick.get("item_id") or "")
            candidate = candidates.get(item_id)
            if candidate is None:
                missing.append(item_id)
                continue
            haystack = _candidate_haystack(candidate)
            matched = [term for term in exclusions if term.lower() in haystack]
            if matched:
                violations.append({"item_id": item_id, "matched_terms": matched})

        status = "pass"
        message = "最终商品未命中用户的排除类硬约束。"
        if violations:
            status = "fail"
            message = f"发现 {len(violations)} 个商品命中排除项。"
        elif missing:
            status = "fail"
            message = f"有 {len(missing)} 个最终商品无法回查候选属性，不能验证排除约束。"
        checks.append(
            RuleCheck(
                rule="C003_EXCLUSIONS",
                title="排除类硬约束",
                section="hard_constraints",
                severity="error",
                status=status,
                message=message,
                max_points=7,
                details={
                    "excluded_terms": exclusions,
                    "violations": violations,
                    "missing_candidate_ids": missing,
                },
            )
        )

    return checks
