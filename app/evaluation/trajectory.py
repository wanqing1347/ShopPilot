from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.evaluation.constraints import evaluate_hard_constraints
from app.evaluation.models import RuleCheck, SectionScore, TrajectoryEvaluation
from app.evaluation.rules import collect_metrics, evaluate_trace_rules

SECTION_CONFIG: dict[str, tuple[str, float]] = {
    "lifecycle": ("生命周期", 25.0),
    "tool_correctness": ("工具调用正确性", 25.0),
    "hard_constraints": ("业务硬约束", 25.0),
    "termination": ("正常终止", 15.0),
    "efficiency": ("执行效率", 10.0),
}


def _section_scores(checks: list[RuleCheck]) -> dict[str, SectionScore]:
    grouped: dict[str, list[RuleCheck]] = defaultdict(list)
    for check in checks:
        grouped[check.section].append(check)

    sections: dict[str, SectionScore] = {}
    for key, (label, section_max) in SECTION_CONFIG.items():
        section_checks = grouped.get(key, [])
        applicable = [check for check in section_checks if check.status != "skipped"]
        possible = sum(check.max_points for check in applicable)
        earned = sum(check.max_points for check in applicable if check.status == "pass")
        score = section_max if possible <= 0 else section_max * earned / possible
        sections[key] = SectionScore(
            key=key,
            label=label,
            score=round(score, 1),
            max_score=section_max,
            passed_checks=sum(check.status == "pass" for check in section_checks),
            failed_checks=sum(check.status == "fail" for check in section_checks),
            skipped_checks=sum(check.status == "skipped" for check in section_checks),
        )
    return sections


def evaluate_trajectory(
    trace: list[dict[str, Any]],
    result: dict[str, Any],
) -> TrajectoryEvaluation:
    """Evaluate one completed Agent trajectory using deterministic invariants.

    The evaluator intentionally does not require one golden tool path. It checks
    lifecycle invariants, stage dependencies, business hard constraints, normal
    termination, and execution efficiency. Subjective planning/tool-choice quality
    is reserved for the future ``llm_judge`` layer in the same output schema.
    """

    safe_trace = [event for event in trace if isinstance(event, dict)]
    safe_result = result if isinstance(result, dict) else {}
    checks = [
        *evaluate_trace_rules(safe_trace, safe_result),
        *evaluate_hard_constraints(safe_result),
    ]
    sections = _section_scores(checks)
    score = round(sum(section.score for section in sections.values()), 1)

    failed_errors = [
        check for check in checks if check.status == "fail" and check.severity == "error"
    ]
    failed_warnings = [
        check for check in checks if check.status == "fail" and check.severity == "warning"
    ]
    skipped = [check for check in checks if check.status == "skipped"]
    metrics = collect_metrics(safe_trace, safe_result)
    metrics.update(
        {
            "failed_error_rules": len(failed_errors),
            "failed_warning_rules": len(failed_warnings),
            "constraint_violations": sum(
                check.status == "fail" and check.rule.startswith("C") for check in checks
            ),
        }
    )

    return TrajectoryEvaluation(
        schema_version="1.0",
        evaluator="rule_based_trajectory_v1",
        score=score,
        passed=not failed_errors,
        summary={
            "errors": len(failed_errors),
            "warnings": len(failed_warnings),
            "passed": sum(check.status == "pass" for check in checks),
            "skipped": len(skipped),
            "total": len(checks),
        },
        metrics=metrics,
        sections=sections,
        checks=checks,
        llm_judge=None,
    )
