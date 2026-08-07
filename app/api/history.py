from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation import evaluate_trajectory
from app.evaluation.judge import evaluate_with_llm_judge
from app.evaluation.judge_context import JUDGE_CONTEXT_VERSION
from app.evaluation.judge_prompt import JUDGE_VERSION
from app.utils.runtime import OUTPUT_ROOT, safe_join


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _completed_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _plan_summary(payload: dict[str, Any]) -> dict[str, Any]:
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        return {}
    return {
        "category": plan.get("category"),
        "category_key": plan.get("category_key"),
        "budget_cny": plan.get("budget_cny"),
        "platforms": plan.get("platforms") or [],
        "hard_constraints": plan.get("hard_constraints") or [],
        "soft_preferences": plan.get("soft_preferences") or [],
    }


def _existing_files(session_dir: Path) -> list[str]:
    return [
        name
        for name in ("shopping-list.md", "result.json", "trace.json", "evaluation.json")
        if (session_dir / name).is_file()
    ]


def list_task_history(
    *,
    user_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if not OUTPUT_ROOT.exists():
        return []

    rows: list[dict[str, Any]] = []
    for session_dir in OUTPUT_ROOT.iterdir():
        if not session_dir.is_dir():
            continue
        result_path = session_dir / "result.json"
        if not result_path.is_file():
            continue
        payload = _read_json(result_path)
        if payload is None:
            continue
        if user_id is not None and str(payload.get("user_id") or "") != user_id:
            continue

        query = str(payload.get("query") or "").strip()
        final_answer = str(payload.get("agent_final_message") or "").strip()
        thread_id = str(payload.get("thread_id") or session_dir.name)
        # Browser tasks created by POST /api/task use uuid.uuid4().hex. Exclude
        # cli-/pytest- demo runs that may share the same demo user id.
        if re.fullmatch(r"[0-9a-f]{32}", thread_id) is None:
            continue
        rows.append(
            {
                "thread_id": thread_id,
                "user_id": payload.get("user_id"),
                "query": query,
                "final_preview": final_answer[:240],
                "completed_at": _completed_at(result_path),
                "plan": _plan_summary(payload),
                "files": _existing_files(session_dir),
            }
        )

    rows.sort(key=lambda row: str(row.get("completed_at") or ""), reverse=True)
    return rows[: max(1, min(limit, 200))]


def _evaluation_payload(
    session_dir: Path,
    trace: list[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    evaluation = evaluate_trajectory(trace, payload).model_dump()
    cached = _read_json(session_dir / "evaluation.json")
    if cached is not None and isinstance(cached.get("llm_judge"), dict):
        judge = dict(cached["llm_judge"])
        judge["stale"] = bool(
            judge.get("judge_version") != JUDGE_VERSION
            or judge.get("context_version") != JUDGE_CONTEXT_VERSION
        )
        evaluation["llm_judge"] = judge
    return evaluation


async def run_task_judge(
    thread_id: str,
    *,
    user_id: str | None = None,
    force: bool = False,
) -> dict[str, Any] | None:
    try:
        session_dir = safe_join(OUTPUT_ROOT, thread_id)
    except ValueError:
        return None
    result_path = session_dir / "result.json"
    trace_path = session_dir / "trace.json"
    if not result_path.is_file() or not trace_path.is_file():
        return None

    payload = _read_json(result_path)
    if payload is None:
        return None
    if user_id is not None and str(payload.get("user_id") or "") != user_id:
        return None

    try:
        raw_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw_trace = []
    trace = [item for item in raw_trace if isinstance(item, dict)] if isinstance(raw_trace, list) else []
    evaluation = _evaluation_payload(session_dir, trace, payload)
    existing = evaluation.get("llm_judge")
    if (
        isinstance(existing, dict)
        and existing.get("status") == "completed"
        and not existing.get("stale")
        and not force
    ):
        return {"evaluation": evaluation, "cached": True}

    judge = await evaluate_with_llm_judge(
        trace=trace,
        result=payload,
        rule_evaluation=evaluation,
    )
    evaluation["llm_judge"] = judge
    (session_dir / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"evaluation": evaluation, "cached": False}


def read_task_history(
    thread_id: str,
    *,
    user_id: str | None = None,
) -> dict[str, Any] | None:
    try:
        session_dir = safe_join(OUTPUT_ROOT, thread_id)
    except ValueError:
        return None
    result_path = session_dir / "result.json"
    if not result_path.is_file():
        return None

    payload = _read_json(result_path)
    if payload is None:
        return None
    if user_id is not None and str(payload.get("user_id") or "") != user_id:
        return None

    trace: list[dict[str, Any]] = []
    trace_path = session_dir / "trace.json"
    if trace_path.is_file():
        try:
            raw_trace = json.loads(trace_path.read_text(encoding="utf-8"))
            if isinstance(raw_trace, list):
                trace = [item for item in raw_trace if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            pass

    final_answer = str(payload.get("agent_final_message") or "")
    if not final_answer:
        markdown_path = session_dir / "shopping-list.md"
        if markdown_path.is_file():
            try:
                final_answer = markdown_path.read_text(encoding="utf-8")
            except OSError:
                pass

    return {
        "thread_id": str(payload.get("thread_id") or thread_id),
        "user_id": payload.get("user_id"),
        "query": str(payload.get("query") or ""),
        "completed_at": _completed_at(result_path),
        "plan": _plan_summary(payload),
        "final_answer": final_answer,
        "files": _existing_files(session_dir),
        "events": trace,
        "evaluation": _evaluation_payload(session_dir, trace, payload),
    }
