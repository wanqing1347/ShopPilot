from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from app.evaluation.judge_context import JUDGE_CONTEXT_VERSION, build_judge_context
from app.evaluation.judge_models import JudgeAssessment
from app.evaluation.judge_prompt import JUDGE_SYSTEM_PROMPT, JUDGE_VERSION

JUDGE_WEIGHTS = {
    "planning_quality": 25.0,
    "tool_selection": 30.0,
    "trajectory_efficiency": 20.0,
    "final_answer_quality": 25.0,
}


class JudgeConfigurationError(RuntimeError):
    """Raised when LLM-as-a-Judge is disabled or cannot be configured."""


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def judge_enabled() -> bool:
    return _bool_env("JUDGE_ENABLED", True)


def judge_model_name() -> str:
    return (
        os.getenv("JUDGE_MODEL_NAME")
        or os.getenv("LLM_MODEL_NAME")
        or os.getenv("LLM_MAIN")
        or "qwen-max"
    ).strip()


@lru_cache(maxsize=1)
def get_judge_llm():
    if not judge_enabled():
        raise JudgeConfigurationError("LLM-as-a-Judge 已通过 JUDGE_ENABLED=false 禁用。")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise JudgeConfigurationError("缺少 langchain-openai，无法运行 LLM Judge。") from exc

    base_url = (
        os.getenv("JUDGE_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or ""
    ).strip()
    api_key = (
        os.getenv("JUDGE_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ("EMPTY" if base_url.startswith(("http://127.0.0.1", "http://localhost")) else "")
    ).strip()
    if not api_key:
        raise JudgeConfigurationError(
            "缺少 JUDGE_API_KEY；也没有可回退的 LLM_API_KEY / OPENAI_API_KEY。"
        )

    try:
        temperature = float(os.getenv("JUDGE_TEMPERATURE", "0"))
    except ValueError:
        temperature = 0.0
    try:
        timeout = float(os.getenv("JUDGE_TIMEOUT_SEC", "75"))
    except ValueError:
        timeout = 75.0

    kwargs: dict[str, object] = {
        "model": judge_model_name(),
        "api_key": api_key,
        "temperature": temperature,
        "timeout": max(5.0, timeout),
        "max_retries": 1,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _score_assessment(assessment: JudgeAssessment) -> float:
    total = 0.0
    for key, weight in JUDGE_WEIGHTS.items():
        dimension = getattr(assessment, key)
        total += (dimension.score / 5.0) * weight
    return round(total, 1)


async def evaluate_with_llm_judge(
    *,
    trace: list[dict[str, Any]],
    result: dict[str, Any],
    rule_evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Run one semantic trajectory review and return a persistable judge payload."""

    context = build_judge_context(trace, result, rule_evaluation)
    model = get_judge_llm().with_structured_output(
        JudgeAssessment,
        method="function_calling",
    )
    raw = await model.ainvoke(
        [
            ("system", JUDGE_SYSTEM_PROMPT),
            ("user", context),
        ]
    )
    assessment = raw if isinstance(raw, JudgeAssessment) else JudgeAssessment.model_validate(raw)
    score = _score_assessment(assessment)
    return {
        "status": "completed",
        "judge_version": JUDGE_VERSION,
        "context_version": JUDGE_CONTEXT_VERSION,
        "model": judge_model_name(),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "rule_evaluator": rule_evaluation.get("evaluator"),
        "rule_score_at_judge": rule_evaluation.get("score"),
        "score": score,
        "weights": dict(JUDGE_WEIGHTS),
        "dimensions": {
            "planning_quality": assessment.planning_quality.model_dump(),
            "tool_selection": assessment.tool_selection.model_dump(),
            "trajectory_efficiency": assessment.trajectory_efficiency.model_dump(),
            "final_answer_quality": assessment.final_answer_quality.model_dump(),
        },
        "strengths": assessment.strengths,
        "issues": assessment.issues,
        "suggestions": assessment.suggestions,
        "verdict": assessment.verdict,
    }


def clear_judge_llm_cache() -> None:
    get_judge_llm.cache_clear()
