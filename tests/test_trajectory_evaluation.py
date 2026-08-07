from __future__ import annotations

from typing import Any

from app.evaluation import evaluate_trajectory


def _event(event: str, *, actor: str = "root", **data: Any) -> dict[str, Any]:
    return {
        "event": event,
        "message": event,
        "data": {"actor_thread_id": actor, **data},
        "timestamp": "2026-08-07T00:00:00+00:00",
    }


def _successful_tool(tool: str, step: int) -> list[dict[str, Any]]:
    return [
        _event("assistant_call", step=step, tool_calls=[tool]),
        _event(
            "tool_attempt",
            tool_name=tool,
            attempt=1,
            max_attempts=3,
            timeout_sec=30,
            idempotency_key=f"{tool}:{step}",
        ),
        _event("tool_success", tool_name=tool, attempts=1, duration_ms=10, source="executed"),
    ]


def _good_trace() -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for step, tool in enumerate(
        [
            "planner",
            "item_search",
            "price_compare",
            "shipping_calc",
            "item_picker",
            "shopping_summary",
        ],
        start=1,
    ):
        trace.extend(_successful_tool(tool, step))
    trace.append(_event("task_result"))
    return trace


def _good_result() -> dict[str, Any]:
    candidate = {
        "item_id": "amazon:1",
        "platform": "amazon",
        "title": "尼龙旅行收纳袋",
        "description": "耐用旅行收纳",
        "attributes": {
            "material": "尼龙",
            "style": "小众",
            "features": ["可机洗"],
            "tags": ["旅行收纳"],
        },
    }
    return {
        "thread_id": "root",
        "terminated": True,
        "terminal_tool": "shopping_summary",
        "plan": {
            "budget_cny": 300.0,
            "platforms": ["amazon"],
            "hard_constraints": ["不要塑料"],
        },
        "search_outputs": [{"platform": "amazon", "candidates": [candidate]}],
        "picker": {
            "picks": [
                {
                    "item_id": "amazon:1",
                    "platform": "amazon",
                    "landed_cny": 199.0,
                }
            ]
        },
    }


def test_rule_based_trajectory_evaluator_accepts_valid_path() -> None:
    evaluation = evaluate_trajectory(_good_trace(), _good_result())

    assert evaluation.passed is True
    assert evaluation.score == 100.0
    assert evaluation.summary["errors"] == 0
    assert evaluation.metrics["tool_calls"] == 6
    assert evaluation.llm_judge is None


def test_rule_based_trajectory_evaluator_rejects_budget_violation() -> None:
    result = _good_result()
    result["picker"]["picks"][0]["landed_cny"] = 399.0

    evaluation = evaluate_trajectory(_good_trace(), result)
    budget = next(check for check in evaluation.checks if check.rule == "C001_BUDGET")

    assert evaluation.passed is False
    assert budget.status == "fail"
    assert evaluation.score < 100


def test_rule_based_trajectory_evaluator_detects_broken_retry_sequence() -> None:
    trace = _good_trace()
    insert_at = 5
    trace[insert_at:insert_at] = [
        _event(
            "tool_retry",
            tool_name="item_search",
            attempt=1,
            next_attempt=3,
            category="timeout",
            delay_sec=1.0,
        ),
        _event(
            "tool_attempt",
            tool_name="item_search",
            attempt=2,
            max_attempts=3,
            timeout_sec=30,
            idempotency_key="item_search:2",
        ),
    ]

    evaluation = evaluate_trajectory(trace, _good_result())
    retry = next(check for check in evaluation.checks if check.rule == "L003_RETRY_SEQUENCE")

    assert retry.status == "fail"
    assert evaluation.passed is False
