from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

JUDGE_CONTEXT_VERSION = "judge_context_v2"


def _event_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    return data if isinstance(data, dict) else {}


def _compact_main_trajectory(
    trace: list[dict[str, Any]],
    root_thread_id: str,
) -> list[dict[str, Any]]:
    successes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    starts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    retries: dict[tuple[str, str], int] = defaultdict(int)
    failures: dict[tuple[str, str], int] = defaultdict(int)
    for event in trace:
        data = _event_data(event)
        actor = str(data.get("actor_thread_id") or root_thread_id)
        tool_name = str(data.get("tool_name") or "")
        if not tool_name:
            continue
        key = (actor, tool_name)
        if event.get("event") == "tool_start":
            raw_args = data.get("args")
            args = raw_args if isinstance(raw_args, dict) else {}
            starts[key].append(
                {
                    field: args.get(field)
                    for field in (
                        "platform",
                        "category",
                        "category_key",
                        "budget_cny",
                        "top_k",
                        "query",
                        "demands",
                    )
                    if args.get(field) is not None
                }
            )
        elif event.get("event") == "tool_success":
            successes[key].append(
                {
                    "duration_ms": data.get("duration_ms"),
                    "attempts": data.get("attempts"),
                    "source": data.get("source"),
                }
            )
        elif event.get("event") == "tool_retry":
            retries[key] += 1
        elif event.get("event") == "tool_failure":
            failures[key] += 1

    steps: list[dict[str, Any]] = []
    seen_success_index: dict[tuple[str, str], int] = defaultdict(int)
    seen_start_index: dict[tuple[str, str], int] = defaultdict(int)
    for event in trace:
        if event.get("event") != "assistant_call":
            continue
        data = _event_data(event)
        actor = str(data.get("actor_thread_id") or root_thread_id)
        if actor != root_thread_id:
            continue
        tool_calls = [str(name) for name in data.get("tool_calls") or [] if name]
        step_payload: dict[str, Any] = {
            "step": data.get("step"),
            "tool_calls": tool_calls,
        }
        call_details: list[dict[str, Any]] = []
        for tool_name in tool_calls:
            key = (actor, tool_name)
            index = seen_success_index[key]
            success = successes.get(key, [])
            detail: dict[str, Any] = {"tool": tool_name}
            start_index = seen_start_index[key]
            start_events = starts.get(key, [])
            if start_index < len(start_events):
                args = dict(start_events[start_index])
                if isinstance(args.get("query"), str):
                    args["query"] = args["query"][:300]
                if isinstance(args.get("demands"), str):
                    args["demands"] = args["demands"][:300]
                if args:
                    detail["args"] = args
                seen_start_index[key] += 1
            if index < len(success):
                detail.update(success[index])
                detail["status"] = "success"
                seen_success_index[key] += 1
            elif failures.get(key, 0):
                detail["status"] = "failure_or_recovered"
            else:
                detail["status"] = "no_terminal_event"
            if retries.get(key, 0):
                detail["retry_events_for_tool"] = retries[key]
            call_details.append(detail)
        if call_details:
            step_payload["calls"] = call_details
        preview = str(data.get("preview") or "").strip()
        if preview:
            step_payload["preview"] = preview[:600]
        steps.append(step_payload)
    return steps


def _fork_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    forks: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    deduplicated = 0
    for event in trace:
        data = _event_data(event)
        kind = event.get("event")
        if kind == "fork":
            forks.append(
                {
                    "sub_thread_id": data.get("sub_thread_id"),
                    "demands": str(data.get("demands") or "")[:500],
                }
            )
        elif kind == "fork_rejected":
            rejected.append(
                {
                    "reason": data.get("reason"),
                    "message": str(event.get("message") or "")[:300],
                }
            )
        elif kind == "fork_deduplicated":
            deduplicated += 1
    return {
        "forks": forks[:8],
        "rejected": rejected[:8],
        "deduplicated_count": deduplicated,
    }


def _final_picks(result: dict[str, Any]) -> list[dict[str, Any]]:
    picker = result.get("picker")
    if not isinstance(picker, dict):
        return []
    picks = picker.get("picks")
    if not isinstance(picks, list):
        return []
    compact: list[dict[str, Any]] = []
    for raw in picks[:10]:
        if not isinstance(raw, dict):
            continue
        compact.append(
            {
                "item_id": raw.get("item_id"),
                "platform": raw.get("platform"),
                "title": raw.get("title"),
                "landed_cny": raw.get("landed_cny"),
                "score": raw.get("score"),
                "reasons": raw.get("reasons") or [],
                "flags": raw.get("flags") or [],
                "data_origin": raw.get("data_origin"),
                "verification_status": raw.get("verification_status"),
            }
        )
    return compact


def build_judge_context(
    trace: list[dict[str, Any]],
    result: dict[str, Any],
    rule_evaluation: dict[str, Any],
) -> str:
    root_thread_id = str(result.get("thread_id") or "main")
    plan = result.get("plan") if isinstance(result.get("plan"), dict) else {}
    failed_checks = []
    for check in rule_evaluation.get("checks") or []:
        if not isinstance(check, dict) or check.get("status") != "fail":
            continue
        failed_checks.append(
            {
                "rule": check.get("rule"),
                "severity": check.get("severity"),
                "message": check.get("message"),
            }
        )

    summary = result.get("summary")
    summary_final = summary.get("final_text") if isinstance(summary, dict) else ""
    final_answer = str(result.get("agent_final_message") or summary_final or "")
    payload = {
        "user_request": str(result.get("query") or ""),
        "plan": {
            "category": plan.get("category"),
            "category_key": plan.get("category_key"),
            "budget_cny": plan.get("budget_cny"),
            "platforms": plan.get("platforms") or [],
            "hard_constraints": plan.get("hard_constraints") or [],
            "soft_preferences": plan.get("soft_preferences") or [],
        },
        "main_agent_trajectory": _compact_main_trajectory(trace, root_thread_id),
        "fork_summary": _fork_summary(trace),
        "rule_evaluation": {
            "score": rule_evaluation.get("score"),
            "passed": rule_evaluation.get("passed"),
            "summary": rule_evaluation.get("summary") or {},
            "metrics": rule_evaluation.get("metrics") or {},
            "failed_checks": failed_checks,
        },
        "final_picks": _final_picks(result),
        "final_answer": final_answer[:7000],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
