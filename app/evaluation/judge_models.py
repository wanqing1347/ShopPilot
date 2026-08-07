from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JudgeDimension(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1, max_length=1200)


class JudgeAssessment(BaseModel):
    planning_quality: JudgeDimension
    tool_selection: JudgeDimension
    trajectory_efficiency: JudgeDimension
    final_answer_quality: JudgeDimension
    strengths: list[str] = Field(default_factory=list, max_length=6)
    issues: list[str] = Field(default_factory=list, max_length=6)
    suggestions: list[str] = Field(default_factory=list, max_length=6)
    verdict: Literal["excellent", "good", "mixed", "poor"]
