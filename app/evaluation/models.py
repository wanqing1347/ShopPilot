from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RuleStatus = Literal["pass", "fail", "skipped"]
RuleSeverity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class RuleCheck:
    rule: str
    title: str
    section: str
    severity: RuleSeverity
    status: RuleStatus
    message: str
    max_points: float
    details: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.status == "pass"
        return payload


@dataclass(frozen=True)
class SectionScore:
    key: str
    label: str
    score: float
    max_score: float
    passed_checks: int
    failed_checks: int
    skipped_checks: int

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryEvaluation:
    schema_version: str
    evaluator: str
    score: float
    passed: bool
    summary: dict[str, int]
    metrics: dict[str, Any]
    sections: dict[str, SectionScore]
    checks: list[RuleCheck]
    llm_judge: dict[str, Any] | None = None

    def model_dump(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluator": self.evaluator,
            "score": self.score,
            "passed": self.passed,
            "summary": dict(self.summary),
            "metrics": dict(self.metrics),
            "sections": {
                key: section.model_dump() for key, section in self.sections.items()
            },
            "checks": [check.model_dump() for check in self.checks],
            # Reserved for the next evaluation layer. Keeping it in the schema now
            # means the history UI and API do not need to change shape later.
            "llm_judge": self.llm_judge,
        }
