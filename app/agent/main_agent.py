from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from app.agent.checkpoint import delete_thread_checkpoint, read_thread_checkpoint
from app.agent.fork_scheduler import reset_fork_scope
from app.agent.prompts import get_system_prompt
from app.agent.settings import main_timeout_sec, retrieval_warmup_timeout_sec
from app.agent.state import initial_state, state_payload
from app.api.connection import manager
from app.api.monitor import monitor
from app.evaluation import evaluate_trajectory
from app.memory.store import store
from app.models import AgentResult, QueryPlan, ShoppingSummaryOutput
from app.utils.runtime import ensure_session_dir, thread_scope


async def _persist_outputs(
    thread_id: str,
    session_dir: Path,
    final_text: str,
    result_payload: dict[str, object],
) -> list[str]:
    markdown_path = session_dir / "shopping-list.md"
    json_path = session_dir / "result.json"
    trace_path = session_dir / "trace.json"
    evaluation_path = session_dir / "evaluation.json"

    markdown_path.write_text(final_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    files = [
        path.name
        for path in (markdown_path, json_path, trace_path, evaluation_path)
    ]
    await monitor.report_task_result(final_text, files)
    trace = manager.get_history(thread_id)
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    evaluation = evaluate_trajectory(trace, result_payload)
    evaluation_path.write_text(
        json.dumps(evaluation.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return files


def _learned_preferences(final_state: dict[str, Any]) -> list[str]:
    summary_value = final_state.get("summary")
    if isinstance(summary_value, ShoppingSummaryOutput):
        return summary_value.learned_preferences
    if isinstance(summary_value, dict):
        return [str(value) for value in summary_value.get("learned_preferences", [])]
    return []


def _plan_from_state(final_state: dict[str, Any]) -> QueryPlan | None:
    plan_value = final_state.get("plan")
    if isinstance(plan_value, QueryPlan):
        return plan_value
    if plan_value is not None:
        return QueryPlan.model_validate(plan_value)
    return None


async def _finalize_run(
    *,
    thread_id: str,
    user_id: str | None,
    final_text: str,
    final_state: dict[str, Any],
    resumed: bool,
) -> AgentResult:
    session_dir = ensure_session_dir(thread_id)
    plan = _plan_from_state(final_state)
    memory_report = await store.write_many(
        user_id,
        _learned_preferences(final_state),
        source_session=thread_id,
        category=plan.category if plan is not None else None,
    )
    if memory_report.upserted or memory_report.superseded_ids:
        await monitor.report_memory_updated(
            upserted=len(memory_report.upserted),
            superseded=len(memory_report.superseded_ids),
            unchanged=len(memory_report.unchanged_ids),
        )
    payload = {
        "runtime": "langgraph-agentloop",
        "resumed_from_checkpoint": resumed,
        **state_payload(final_state),
        "memory_write_report": memory_report.model_dump(mode="json"),
        "agent_final_message": final_text,
    }
    files = await _persist_outputs(
        thread_id=thread_id,
        session_dir=session_dir,
        final_text=final_text,
        result_payload=payload,
    )
    return AgentResult(
        status="ok",
        thread_id=thread_id,
        plan=plan,
        final=final_text,
        output_files=files,
    )


async def _run_agent_loop(
    query: str,
    thread_id: str,
    user_id: str | None,
) -> AgentResult:
    from app.agent.graph_runtime import ainvoke_agent

    session_dir = ensure_session_dir(thread_id)
    with thread_scope(thread_id, session_dir, root_thread_id=thread_id):
        # POST /api/task means a fresh task. Recovery uses the dedicated resume path.
        await delete_thread_checkpoint(thread_id)
        memory_search = await store.read_relevant(user_id=user_id, query=query)
        stored_preferences = memory_search.prompt_texts
        structured_memory = [
            entry.model_dump(mode="json") for entry in memory_search.entries
        ]
        await monitor.report_memory_retrieved(
            count=len(structured_memory),
            query=query,
        )
        state = initial_state(
            query=query,
            thread_id=thread_id,
            user_id=user_id,
            long_term_preferences=stored_preferences,
            long_term_memory=structured_memory,
        )
        run = await ainvoke_agent(
            query=query,
            thread_id=thread_id,
            system_prompt=get_system_prompt(stored_preferences, structured_memory),
            initial=state,
        )
        return await _finalize_run(
            thread_id=thread_id,
            user_id=user_id,
            final_text=run.final_text,
            final_state=dict(run.state),
            resumed=False,
        )


async def _resume_agent_loop(thread_id: str) -> AgentResult:
    from app.agent.graph_runtime import aresume_agent

    saved = await read_thread_checkpoint(thread_id)
    if saved is None:
        raise LookupError(f"checkpoint 不存在：{thread_id}")

    user_id_value = saved.get("user_id")
    user_id = str(user_id_value) if user_id_value is not None else None
    preferences = [str(value) for value in saved.get("long_term_preferences", [])]
    structured_memory = [
        value
        for value in saved.get("long_term_memory", [])
        if isinstance(value, dict)
    ]
    session_dir = ensure_session_dir(thread_id)
    with thread_scope(thread_id, session_dir, root_thread_id=thread_id):
        run = await aresume_agent(
            thread_id=thread_id,
            system_prompt=get_system_prompt(preferences, structured_memory),
        )
        return await _finalize_run(
            thread_id=thread_id,
            user_id=user_id,
            final_text=run.final_text,
            final_state=dict(run.state),
            resumed=True,
        )


async def _warm_retrieval_resources(thread_id: str, session_dir: Path) -> None:
    """Initialize the local hybrid retriever before the main Agent timeout starts.

    SentenceTransformer document encoding is a one-time startup cost and can take
    far longer than an ordinary search. Keeping it inside item_search's 30-second
    reliability timeout causes retries to spawn more non-cancellable worker
    threads. Warm it once, separately, then let every tool call use the hot cache.
    """

    from app.recall.hybrid import get_hybrid_retriever

    with thread_scope(thread_id, session_dir, root_thread_id=thread_id):
        await monitor.report_stage("prepare", "正在预热商品检索索引")
    await asyncio.wait_for(
        asyncio.to_thread(get_hybrid_retriever),
        timeout=retrieval_warmup_timeout_sec(),
    )


async def run_agent(
    query: str,
    thread_id: str,
    user_id: str | None = None,
) -> AgentResult:
    session_dir = ensure_session_dir(thread_id)
    try:
        await _warm_retrieval_resources(thread_id, session_dir)
    except asyncio.TimeoutError:
        message = f"商品检索预热超过 {retrieval_warmup_timeout_sec():g}s"
        with thread_scope(thread_id, session_dir, root_thread_id=thread_id):
            await monitor.report_error("retrieval_warmup_timeout", message)
        return AgentResult(status="timeout", thread_id=thread_id, error=message)
    except Exception as exc:
        message = f"检索预热失败：{type(exc).__name__}: {exc}"
        with thread_scope(thread_id, session_dir, root_thread_id=thread_id):
            await monitor.report_error("retrieval_warmup_error", message)
        return AgentResult(status="error", thread_id=thread_id, error=message)

    try:
        return await asyncio.wait_for(
            _run_agent_loop(query, thread_id, user_id),
            timeout=main_timeout_sec(),
        )
    except asyncio.TimeoutError:
        with thread_scope(thread_id, session_dir, root_thread_id=thread_id):
            await monitor.report_error(
                "timeout",
                f"主 AgentLoop 超过 {main_timeout_sec()}s",
            )
        return AgentResult(status="timeout", thread_id=thread_id)
    except Exception as exc:
        with thread_scope(thread_id, session_dir, root_thread_id=thread_id):
            await monitor.report_error("internal_error", f"{type(exc).__name__}: {exc}")
        return AgentResult(
            status="error",
            thread_id=thread_id,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        await reset_fork_scope(thread_id)


async def resume_agent(thread_id: str) -> AgentResult:
    session_dir = ensure_session_dir(thread_id)
    try:
        await _warm_retrieval_resources(thread_id, session_dir)
    except asyncio.TimeoutError:
        message = f"商品检索预热超过 {retrieval_warmup_timeout_sec():g}s"
        with thread_scope(thread_id, session_dir, root_thread_id=thread_id):
            await monitor.report_error("retrieval_warmup_timeout", message)
        return AgentResult(status="timeout", thread_id=thread_id, error=message)
    except Exception as exc:
        message = f"检索预热失败：{type(exc).__name__}: {exc}"
        with thread_scope(thread_id, session_dir, root_thread_id=thread_id):
            await monitor.report_error("retrieval_warmup_error", message)
        return AgentResult(status="error", thread_id=thread_id, error=message)

    try:
        return await asyncio.wait_for(
            _resume_agent_loop(thread_id),
            timeout=main_timeout_sec(),
        )
    except asyncio.TimeoutError:
        with thread_scope(thread_id, session_dir, root_thread_id=thread_id):
            await monitor.report_error(
                "resume_timeout",
                f"恢复后的 AgentLoop 超过 {main_timeout_sec()}s",
            )
        return AgentResult(status="timeout", thread_id=thread_id)
    except Exception as exc:
        with thread_scope(thread_id, session_dir, root_thread_id=thread_id):
            await monitor.report_error("resume_error", f"{type(exc).__name__}: {exc}")
        return AgentResult(
            status="error",
            thread_id=thread_id,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        await reset_fork_scope(thread_id)
