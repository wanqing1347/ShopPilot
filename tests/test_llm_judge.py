from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.evaluation.judge import evaluate_with_llm_judge
from app.evaluation.judge_context import build_judge_context
from app.evaluation.judge_models import JudgeAssessment, JudgeDimension


def _event(event: str, **data: Any) -> dict[str, Any]:
    return {
        "event": event,
        "message": event,
        "data": {"actor_thread_id": "root", **data},
        "timestamp": "2026-08-07T00:00:00+00:00",
    }


def _trace() -> list[dict[str, Any]]:
    return [
        _event("assistant_call", step=1, tool_calls=["planner"]),
        _event("tool_success", tool_name="planner", attempts=1, duration_ms=20, source="executed"),
        _event("assistant_token", delta="noise"),
        _event("assistant_call", step=2, tool_calls=["item_search"]),
        _event(
            "tool_start",
            tool_name="item_search",
            args={
                "query": "小众旅行收纳",
                "platform": "amazon",
                "category": "travel_storage",
                "budget_cny": 300,
                "top_k": 20,
            },
        ),
        _event("tool_retry", tool_name="item_search", attempt=1, next_attempt=2),
        _event("tool_success", tool_name="item_search", attempts=2, duration_ms=150, source="executed"),
        _event("assistant_call", step=3, tool_calls=["shopping_summary"]),
        _event("tool_success", tool_name="shopping_summary", attempts=1, duration_ms=30, source="executed"),
        _event("task_result"),
    ]


def _result() -> dict[str, Any]:
    return {
        "thread_id": "root",
        "user_id": "demo-user",
        "query": "预算300元，找小众旅行收纳，不要塑料",
        "plan": {
            "category": "旅行收纳",
            "category_key": "travel_storage",
            "budget_cny": 300,
            "platforms": ["amazon"],
            "hard_constraints": ["不要塑料"],
            "soft_preferences": ["小众"],
        },
        "picker": {
            "picks": [
                {
                    "item_id": "amazon:1",
                    "platform": "amazon",
                    "title": "尼龙旅行收纳袋",
                    "landed_cny": 199,
                    "score": 0.9,
                    "reasons": ["预算内", "小众"],
                    "flags": [],
                    "data_origin": "synthetic",
                    "verification_status": "synthetic",
                }
            ]
        },
        "agent_final_message": "推荐尼龙旅行收纳袋，到手价199元，并明确说明为合成演示数据。",
    }


def _rule_eval() -> dict[str, Any]:
    return {
        "score": 98,
        "passed": True,
        "summary": {"errors": 0, "warnings": 1},
        "metrics": {"tool_calls": 3, "tool_retries": 1},
        "checks": [
            {
                "rule": "E002_RETRY_RATE",
                "severity": "warning",
                "status": "fail",
                "message": "发生一次可恢复重试",
            }
        ],
    }


def test_judge_context_is_compact_and_keeps_semantic_evidence() -> None:
    context = build_judge_context(_trace(), _result(), _rule_eval())
    payload = json.loads(context)

    assert payload["user_request"].startswith("预算300元")
    assert payload["plan"]["budget_cny"] == 300
    assert len(payload["main_agent_trajectory"]) == 3
    assert payload["rule_evaluation"]["failed_checks"][0]["rule"] == "E002_RETRY_RATE"
    assert payload["main_agent_trajectory"][1]["calls"][0]["args"]["platform"] == "amazon"
    assert "assistant_token" not in context
    assert payload["final_picks"][0]["landed_cny"] == 199


@pytest.mark.asyncio
async def test_llm_judge_uses_structured_output_and_backend_weighting(monkeypatch: pytest.MonkeyPatch) -> None:
    assessment = JudgeAssessment(
        planning_quality=JudgeDimension(score=5, reason="需求识别完整"),
        tool_selection=JudgeDimension(score=4, reason="工具基本合理"),
        trajectory_efficiency=JudgeDimension(score=3, reason="存在一次低收益重试"),
        final_answer_quality=JudgeDimension(score=5, reason="答案完整且披露数据来源"),
        strengths=["计划完整"],
        issues=["有轻微冗余"],
        suggestions=["减少重复搜索"],
        verdict="good",
    )

    class FakeStructuredModel:
        async def ainvoke(self, messages: list[tuple[str, str]]) -> JudgeAssessment:
            assert messages[0][0] == "system"
            assert "user_request" in messages[1][1]
            return assessment

    class FakeJudgeModel:
        def with_structured_output(self, schema: type[JudgeAssessment], method: str):
            assert schema is JudgeAssessment
            assert method == "function_calling"
            return FakeStructuredModel()

    monkeypatch.setattr("app.evaluation.judge.get_judge_llm", lambda: FakeJudgeModel())
    monkeypatch.setattr("app.evaluation.judge.judge_model_name", lambda: "judge-test-model")

    result = await evaluate_with_llm_judge(
        trace=_trace(),
        result=_result(),
        rule_evaluation=_rule_eval(),
    )

    assert result["score"] == 86.0
    assert result["model"] == "judge-test-model"
    assert result["dimensions"]["tool_selection"]["score"] == 4
    assert result["verdict"] == "good"


@pytest.mark.asyncio
async def test_history_judge_is_persisted_and_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.api import history

    root = tmp_path / "output"
    thread_id = "a" * 32
    session = root / thread_id
    session.mkdir(parents=True)
    payload = _result() | {"thread_id": thread_id}
    (session / "result.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (session / "trace.json").write_text(json.dumps(_trace(), ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(history, "OUTPUT_ROOT", root)
    calls = 0

    async def fake_judge(**_: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "status": "completed",
            "judge_version": "trajectory_judge_v2",
            "context_version": "judge_context_v2",
            "model": "fake-judge",
            "evaluated_at": "2026-08-07T00:00:00+00:00",
            "score": 90.0,
            "weights": {},
            "dimensions": {},
            "strengths": [],
            "issues": [],
            "suggestions": [],
            "verdict": "good",
        }

    monkeypatch.setattr(history, "evaluate_with_llm_judge", fake_judge)

    first = await history.run_task_judge(thread_id, user_id="demo-user")
    second = await history.run_task_judge(thread_id, user_id="demo-user")

    assert first is not None and first["cached"] is False
    assert second is not None and second["cached"] is True
    assert calls == 1
    saved = json.loads((session / "evaluation.json").read_text(encoding="utf-8"))
    assert saved["llm_judge"]["model"] == "fake-judge"
