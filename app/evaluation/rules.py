from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from app.agent.settings import max_model_steps
from app.evaluation.models import RuleCheck


def _data(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("data")
    return value if isinstance(value, dict) else {}


def _actor(event: dict[str, Any]) -> str:
    return str(_data(event).get("actor_thread_id") or "unknown")


def _tool(event: dict[str, Any]) -> str:
    return str(_data(event).get("tool_name") or "")


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _main_actor(result: dict[str, Any], trace: list[dict[str, Any]]) -> str:
    thread_id = str(result.get("thread_id") or "")
    if thread_id:
        return thread_id
    for event in trace:
        actor = _actor(event)
        if actor != "unknown" and not actor.startswith("sub-"):
            return actor
    return "unknown"


def _events_of(trace: list[dict[str, Any]], name: str) -> list[tuple[int, dict[str, Any]]]:
    return [(index, event) for index, event in enumerate(trace) if event.get("event") == name]


def _duplicate_executions(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts: Counter[tuple[str, str, str]] = Counter()
    for event in trace:
        if event.get("event") != "tool_attempt":
            continue
        data = _data(event)
        if int(data.get("attempt") or 0) != 1:
            continue
        key = (
            _actor(event),
            _tool(event),
            str(data.get("idempotency_key") or ""),
        )
        starts[key] += 1
    return [
        {
            "actor_thread_id": actor,
            "tool_name": tool,
            "idempotency_key": key,
            "executions": count,
        }
        for (actor, tool, key), count in starts.items()
        if key and count > 1
    ]


def _slow_tools(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_timeout: dict[tuple[str, str], float] = {}
    slow: list[dict[str, Any]] = []
    for event in trace:
        name = str(event.get("event") or "")
        actor_tool = (_actor(event), _tool(event))
        data = _data(event)
        if name == "tool_attempt":
            try:
                latest_timeout[actor_tool] = float(data.get("timeout_sec") or 0.0)
            except (TypeError, ValueError):
                latest_timeout[actor_tool] = 0.0
            continue
        if name != "tool_success":
            continue
        try:
            duration_ms = int(data.get("duration_ms") or 0)
        except (TypeError, ValueError):
            duration_ms = 0
        timeout_sec = latest_timeout.get(actor_tool, 0.0)
        near_timeout = timeout_sec > 0 and duration_ms >= timeout_sec * 1000 * 0.90
        very_slow = duration_ms >= 60_000
        if near_timeout or very_slow:
            slow.append(
                {
                    "actor_thread_id": actor_tool[0],
                    "tool_name": actor_tool[1],
                    "duration_ms": duration_ms,
                    "timeout_sec": timeout_sec,
                }
            )
    return slow


def evaluate_trace_rules(
    trace: list[dict[str, Any]],
    result: dict[str, Any],
) -> list[RuleCheck]:
    checks: list[RuleCheck] = []
    main_actor = _main_actor(result, trace)

    # ----- Termination -----
    terminated = bool(result.get("terminated"))
    checks.append(
        RuleCheck(
            rule="T001_TERMINATED",
            title="Agent 正常终止",
            section="termination",
            severity="error",
            status="pass" if terminated else "fail",
            message="checkpoint state 标记为 terminated=true。" if terminated else "任务有结果文件，但 state 未标记为 terminated。",
            max_points=5,
        )
    )

    terminal_tool = str(result.get("terminal_tool") or "")
    valid_terminal_tools = {"shopping_summary", "chat_fallback"}
    checks.append(
        RuleCheck(
            rule="T002_TERMINAL_TOOL",
            title="终止工具合法",
            section="termination",
            severity="error",
            status="pass" if terminal_tool in valid_terminal_tools else "fail",
            message=(
                f"终止工具为 {terminal_tool}。"
                if terminal_tool in valid_terminal_tools
                else f"terminal_tool={terminal_tool or '<empty>'}，不在允许的终止工具集合中。"
            ),
            max_points=5,
            details={"allowed": sorted(valid_terminal_tools), "terminal_tool": terminal_tool},
        )
    )

    task_results = [event for event in trace if event.get("event") == "task_result"]
    checks.append(
        RuleCheck(
            rule="T003_SINGLE_TASK_RESULT",
            title="最终结果事件唯一",
            section="termination",
            severity="error",
            status="pass" if len(task_results) == 1 else "fail",
            message=(
                "trace 中恰好存在 1 个 task_result。"
                if len(task_results) == 1
                else f"trace 中存在 {len(task_results)} 个 task_result，期望恰好 1 个。"
            ),
            max_points=5,
            details={"task_result_count": len(task_results)},
        )
    )

    # ----- Lifecycle invariants -----
    steps_by_actor: dict[str, list[int]] = defaultdict(list)
    for event in trace:
        if event.get("event") != "assistant_call":
            continue
        try:
            step = int(_data(event).get("step") or 0)
        except (TypeError, ValueError):
            step = 0
        steps_by_actor[_actor(event)].append(step)
    bad_steps: dict[str, list[int]] = {}
    for actor, steps in steps_by_actor.items():
        expected = list(range(1, len(steps) + 1))
        if steps != expected:
            bad_steps[actor] = steps
    checks.append(
        RuleCheck(
            rule="L001_STEP_SEQUENCE",
            title="Agent step 连续递增",
            section="lifecycle",
            severity="error",
            status="fail" if bad_steps else "pass",
            message=(
                f"发现 {len(bad_steps)} 个 Agent 的 step 序列异常。"
                if bad_steps
                else "主 Agent 与子 Agent 的 assistant_call step 均从 1 连续递增。"
            ),
            max_points=5,
            details={"invalid_actors": bad_steps},
        )
    )

    invalid_attempts: list[dict[str, Any]] = []
    for index, event in _events_of(trace, "tool_attempt"):
        data = _data(event)
        try:
            attempt = int(data.get("attempt") or 0)
            maximum = int(data.get("max_attempts") or 0)
        except (TypeError, ValueError):
            attempt, maximum = 0, 0
        if not (1 <= attempt <= maximum):
            invalid_attempts.append(
                {
                    "index": index,
                    "actor_thread_id": _actor(event),
                    "tool_name": _tool(event),
                    "attempt": attempt,
                    "max_attempts": maximum,
                }
            )
    checks.append(
        RuleCheck(
            rule="L002_ATTEMPT_BOUNDS",
            title="工具重试次数合法",
            section="lifecycle",
            severity="error",
            status="fail" if invalid_attempts else "pass",
            message=(
                f"发现 {len(invalid_attempts)} 个 attempt 越界事件。"
                if invalid_attempts
                else "所有 tool_attempt 均满足 1 <= attempt <= max_attempts。"
            ),
            max_points=5,
            details={"violations": invalid_attempts},
        )
    )

    retry_violations: list[dict[str, Any]] = []
    for index, event in _events_of(trace, "tool_retry"):
        data = _data(event)
        actor = _actor(event)
        tool = _tool(event)
        previous: dict[str, Any] | None = None
        following: dict[str, Any] | None = None
        for candidate in reversed(trace[:index]):
            if candidate.get("event") == "tool_attempt" and _actor(candidate) == actor and _tool(candidate) == tool:
                previous = candidate
                break
        for candidate in trace[index + 1 :]:
            if candidate.get("event") == "tool_attempt" and _actor(candidate) == actor and _tool(candidate) == tool:
                following = candidate
                break
        try:
            retry_attempt = int(data.get("attempt") or 0)
            next_attempt = int(data.get("next_attempt") or 0)
            previous_attempt = int(_data(previous or {}).get("attempt") or 0)
            following_attempt = int(_data(following or {}).get("attempt") or 0)
        except (TypeError, ValueError):
            retry_attempt = next_attempt = previous_attempt = following_attempt = 0
        if (
            previous is None
            or following is None
            or retry_attempt != previous_attempt
            or next_attempt != retry_attempt + 1
            or following_attempt != next_attempt
        ):
            retry_violations.append(
                {
                    "index": index,
                    "actor_thread_id": actor,
                    "tool_name": tool,
                    "retry_attempt": retry_attempt,
                    "declared_next_attempt": next_attempt,
                    "previous_attempt": previous_attempt,
                    "following_attempt": following_attempt,
                }
            )
    checks.append(
        RuleCheck(
            rule="L003_RETRY_SEQUENCE",
            title="Retry 生命周期连续",
            section="lifecycle",
            severity="error",
            status="fail" if retry_violations else "pass",
            message=(
                f"发现 {len(retry_violations)} 个 retry 序列异常。"
                if retry_violations
                else "所有 retry 都能关联到前一 attempt 与下一 attempt。"
            ),
            max_points=5,
            details={"violations": retry_violations},
        )
    )

    pending_attempts: Counter[tuple[str, str]] = Counter()
    terminal_without_attempt: list[dict[str, Any]] = []
    for index, event in enumerate(trace):
        name = event.get("event")
        key = (_actor(event), _tool(event))
        if name == "tool_attempt":
            pending_attempts[key] += 1
        elif name in {"tool_success", "tool_failure"}:
            if pending_attempts[key] <= 0:
                terminal_without_attempt.append(
                    {
                        "index": index,
                        "actor_thread_id": key[0],
                        "tool_name": key[1],
                        "event": name,
                    }
                )
            pending_attempts[key] = 0
    checks.append(
        RuleCheck(
            rule="L004_TERMINAL_HAS_ATTEMPT",
            title="工具结果可追溯到 attempt",
            section="lifecycle",
            severity="error",
            status="fail" if terminal_without_attempt else "pass",
            message=(
                f"发现 {len(terminal_without_attempt)} 个 success/failure 缺少对应 attempt。"
                if terminal_without_attempt
                else "所有工具最终 success/failure 都能追溯到 tool_attempt。"
            ),
            max_points=5,
            details={"violations": terminal_without_attempt},
        )
    )

    result_indices = [index for index, event in _events_of(trace, "task_result")]
    after_result: list[dict[str, Any]] = []
    business_events = {
        "assistant_call",
        "tool_attempt",
        "tool_start",
        "tool_end",
        "tool_success",
        "tool_failure",
        "tool_retry",
        "fork",
    }
    if result_indices:
        final_index = result_indices[-1]
        for index, event in enumerate(trace[final_index + 1 :], start=final_index + 1):
            if event.get("event") in business_events:
                after_result.append({"index": index, "event": event.get("event")})
    checks.append(
        RuleCheck(
            rule="L005_NO_WORK_AFTER_RESULT",
            title="最终结果后无继续执行",
            section="lifecycle",
            severity="error",
            status="fail" if after_result else "pass",
            message=(
                f"task_result 后仍出现 {len(after_result)} 个业务执行事件。"
                if after_result
                else "task_result 之后没有继续调用模型或工具。"
            ),
            max_points=5,
            details={"violations": after_result},
        )
    )

    # ----- Tool dependency invariants on the root agent -----
    root_successes: list[tuple[int, str]] = [
        (index, _tool(event))
        for index, event in _events_of(trace, "tool_success")
        if _actor(event) == main_actor
    ]

    def positions(tool_name: str) -> list[int]:
        return [index for index, name in root_successes if name == tool_name]

    def has_before(tool_names: set[str], position: int) -> bool:
        return any(index < position and name in tool_names for index, name in root_successes)

    search_positions = sorted(positions("item_search") + positions("dispatch_tool"))
    search_bad = [position for position in search_positions if not has_before({"planner"}, position)]
    checks.append(
        RuleCheck(
            rule="D001_PLANNER_BEFORE_SEARCH",
            title="搜索前完成 Planner",
            section="tool_correctness",
            severity="error",
            status="skipped" if not search_positions else ("fail" if search_bad else "pass"),
            message=(
                "没有发生商品搜索，跳过该依赖检查。"
                if not search_positions
                else (f"发现 {len(search_bad)} 次搜索发生在 planner 成功之前。" if search_bad else "所有商品搜索均发生在 planner 成功之后。")
            ),
            max_points=5,
            details={"violations": search_bad},
        )
    )

    compare_positions = positions("price_compare")
    compare_bad = [position for position in compare_positions if not has_before({"item_search", "dispatch_tool"}, position)]
    checks.append(
        RuleCheck(
            rule="D002_COMPARE_AFTER_SEARCH",
            title="比价依赖搜索结果",
            section="tool_correctness",
            severity="error",
            status="skipped" if not compare_positions else ("fail" if compare_bad else "pass"),
            message=(
                "没有调用 price_compare，跳过该依赖检查。"
                if not compare_positions
                else (f"发现 {len(compare_bad)} 次 price_compare 前没有成功搜索。" if compare_bad else "price_compare 均建立在搜索步骤之后。")
            ),
            max_points=5,
            details={"violations": compare_bad},
        )
    )

    shipping_positions = positions("shipping_calc")
    shipping_bad = [position for position in shipping_positions if not has_before({"price_compare"}, position)]
    checks.append(
        RuleCheck(
            rule="D003_SHIPPING_AFTER_COMPARE",
            title="到手价计算依赖比价",
            section="tool_correctness",
            severity="error",
            status="skipped" if not shipping_positions else ("fail" if shipping_bad else "pass"),
            message=(
                "没有调用 shipping_calc，跳过该依赖检查。"
                if not shipping_positions
                else (f"发现 {len(shipping_bad)} 次 shipping_calc 前没有 price_compare。" if shipping_bad else "shipping_calc 均建立在 price_compare 之后。")
            ),
            max_points=5,
            details={"violations": shipping_bad},
        )
    )

    picker_positions = positions("item_picker")
    picker_bad = [position for position in picker_positions if not has_before({"shipping_calc"}, position)]
    checks.append(
        RuleCheck(
            rule="D004_PICKER_AFTER_SHIPPING",
            title="精排依赖到手价",
            section="tool_correctness",
            severity="error",
            status="skipped" if not picker_positions else ("fail" if picker_bad else "pass"),
            message=(
                "没有调用 item_picker，跳过该依赖检查。"
                if not picker_positions
                else (f"发现 {len(picker_bad)} 次 item_picker 前没有 shipping_calc。" if picker_bad else "item_picker 均建立在 shipping_calc 之后。")
            ),
            max_points=5,
            details={"violations": picker_bad},
        )
    )

    summary_positions = positions("shopping_summary")
    summary_bad = [position for position in summary_positions if not has_before({"item_picker"}, position)]
    checks.append(
        RuleCheck(
            rule="D005_SUMMARY_AFTER_PICKER",
            title="总结依赖精排结果",
            section="tool_correctness",
            severity="error",
            status="skipped" if not summary_positions else ("fail" if summary_bad else "pass"),
            message=(
                "没有调用 shopping_summary，跳过该依赖检查。"
                if not summary_positions
                else (f"发现 {len(summary_bad)} 次 shopping_summary 前没有 item_picker。" if summary_bad else "shopping_summary 建立在 item_picker 之后。")
            ),
            max_points=5,
            details={"violations": summary_bad},
        )
    )

    # ----- Efficiency (warning-level; these reduce score but do not make passed=false) -----
    main_steps = steps_by_actor.get(main_actor, [])
    main_step_count = max(main_steps, default=0)
    soft_step_limit = max(12, int(max_model_steps() * 0.60))
    checks.append(
        RuleCheck(
            rule="E001_MODEL_STEPS",
            title="AgentLoop 步数合理",
            section="efficiency",
            severity="warning",
            status="fail" if main_step_count > soft_step_limit else "pass",
            message=(
                f"主 Agent 使用 {main_step_count} 步，超过效率阈值 {soft_step_limit}。"
                if main_step_count > soft_step_limit
                else f"主 Agent 使用 {main_step_count} 步，未超过效率阈值 {soft_step_limit}。"
            ),
            max_points=2.5,
            details={"model_steps": main_step_count, "soft_limit": soft_step_limit},
        )
    )

    retries = len(_events_of(trace, "tool_retry"))
    logical_calls = sum(
        1
        for _, event in _events_of(trace, "tool_attempt")
        if int(_data(event).get("attempt") or 0) == 1
    )
    retry_rate = retries / logical_calls if logical_calls else 0.0
    checks.append(
        RuleCheck(
            rule="E002_RETRY_RATE",
            title="工具重试率合理",
            section="efficiency",
            severity="warning",
            status="fail" if retry_rate > 0.25 else "pass",
            message=(
                f"工具重试率 {retry_rate:.0%}，高于 25% 阈值。"
                if retry_rate > 0.25
                else f"工具重试率 {retry_rate:.0%}。"
            ),
            max_points=2.5,
            details={"retries": retries, "logical_tool_calls": logical_calls, "retry_rate": round(retry_rate, 4)},
        )
    )

    duplicates = _duplicate_executions(trace)
    checks.append(
        RuleCheck(
            rule="E003_DUPLICATE_EXECUTION",
            title="无重复等价工具执行",
            section="efficiency",
            severity="warning",
            status="fail" if duplicates else "pass",
            message=(
                f"发现 {len(duplicates)} 组相同幂等键被重复从 attempt=1 执行。"
                if duplicates
                else "没有发现相同幂等键的重复完整执行。"
            ),
            max_points=2.5,
            details={"duplicates": duplicates},
        )
    )

    slow_tools = _slow_tools(trace)
    checks.append(
        RuleCheck(
            rule="E004_SLOW_TOOL",
            title="工具耗时未逼近超时",
            section="efficiency",
            severity="warning",
            status="fail" if slow_tools else "pass",
            message=(
                f"发现 {len(slow_tools)} 次工具执行超过 60s 或达到超时预算的 90%。"
                if slow_tools
                else "没有工具执行超过 60s 或逼近自身超时预算。"
            ),
            max_points=2.5,
            details={"slow_tools": slow_tools},
        )
    )

    return checks


def collect_metrics(
    trace: list[dict[str, Any]],
    result: dict[str, Any],
) -> dict[str, Any]:
    main_actor = _main_actor(result, trace)
    timestamps = [
        parsed
        for parsed in (_parse_timestamp(event.get("timestamp")) for event in trace)
        if parsed is not None
    ]
    duration_ms = 0
    if len(timestamps) >= 2:
        duration_ms = max(0, int((max(timestamps) - min(timestamps)).total_seconds() * 1000))

    main_steps = [
        int(_data(event).get("step") or 0)
        for event in trace
        if event.get("event") == "assistant_call" and _actor(event) == main_actor
    ]
    tool_attempt_events = [event for _, event in _events_of(trace, "tool_attempt")]
    logical_tool_calls = sum(
        1 for event in tool_attempt_events if int(_data(event).get("attempt") or 0) == 1
    )

    success_durations: list[dict[str, Any]] = []
    for _, event in _events_of(trace, "tool_success"):
        data = _data(event)
        try:
            duration = int(data.get("duration_ms") or 0)
        except (TypeError, ValueError):
            duration = 0
        success_durations.append(
            {
                "tool_name": _tool(event),
                "actor_thread_id": _actor(event),
                "duration_ms": duration,
            }
        )
    longest_tool = max(success_durations, key=lambda row: row["duration_ms"], default=None)

    picker = result.get("picker")
    picks = picker.get("picks") if isinstance(picker, dict) else []
    actors = sorted({_actor(event) for event in trace if _actor(event) != "unknown"})

    return {
        "duration_ms": duration_ms,
        "model_steps": max(main_steps, default=0),
        "model_calls_all_agents": len(_events_of(trace, "assistant_call")),
        "tool_calls": logical_tool_calls,
        "tool_attempts": len(tool_attempt_events),
        "tool_successes": len(_events_of(trace, "tool_success")),
        "tool_failures": len(_events_of(trace, "tool_failure")),
        "tool_retries": len(_events_of(trace, "tool_retry")),
        "idempotency_hits": len(_events_of(trace, "tool_idempotency_hit")),
        "fork_count": len(_events_of(trace, "fork")),
        "fork_rejected": len(_events_of(trace, "fork_rejected")),
        "actor_count": len(actors),
        "actors": actors,
        "picked_items": len(picks or []),
        "terminal_tool": result.get("terminal_tool"),
        "task_result_count": len(_events_of(trace, "task_result")),
        "duplicate_executions": len(_duplicate_executions(trace)),
        "slow_tool_count": len(_slow_tools(trace)),
        "longest_tool": longest_tool,
    }
