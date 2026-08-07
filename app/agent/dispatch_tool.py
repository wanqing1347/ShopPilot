from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.agent.fork_guard import ForkLimitExceeded, enter_fork
from app.agent.fork_scheduler import (
    ForkBudgetExceeded,
    ForkQueueFull,
    ForkQueueTimeout,
    get_fork_scheduler,
)
from app.agent.prompts import get_scoped_search_prompt, get_sub_agent_prompt
from app.agent.settings import sub_agent_timeout_sec
from app.agent.state import ShopPilotState, initial_state, state_payload
from app.api.context import get_root_thread_id, get_session_dir
from app.api.monitor import monitor
from app.models import Platform, QueryPlan
from app.utils.runtime import thread_scope


@dataclass
class SubAgentRun:
    sub_thread_id: str
    final_answer: str
    state: ShopPilotState | dict[str, Any]

    def artifact(self) -> dict[str, Any]:
        return {
            "sub_thread_id": self.sub_thread_id,
            "final_answer": self.final_answer,
            "state": state_payload(self.state),
        }


def _build_child_query(
    demands: str,
    platform: Platform | None,
    category: str | None,
) -> str:
    constraints: list[str] = []
    if platform is not None:
        constraints.append(f"只允许检索平台：{platform}")
    if category:
        constraints.append(f"目标商品品类：{category}")
    if not constraints:
        return demands
    return demands + "\n\n子任务硬范围：\n- " + "\n- ".join(constraints)


def _scoped_parent_plan(
    parent_state: ShopPilotState | dict[str, Any],
    *,
    platform: Platform | None,
    category: str | None,
) -> QueryPlan | None:
    """Reuse the parent's validated plan for one-platform leaf retrieval."""

    raw_plan = parent_state.get("plan")
    if raw_plan is None:
        return None
    plan = raw_plan if isinstance(raw_plan, QueryPlan) else QueryPlan.model_validate(raw_plan)
    updates: dict[str, Any] = {}
    if platform is not None:
        updates["platforms"] = [platform]
    if category:
        updates["category"] = category
    return plan.model_copy(update=updates) if updates else plan


async def run_sub_agent(
    *,
    demands: str,
    platform: Platform | None,
    category: str | None,
    parent_state: ShopPilotState | dict[str, Any],
) -> SubAgentRun:
    """Run a homogeneous child AgentLoop with checkpointed scoped state."""

    session_dir = get_session_dir()
    root_thread_id = get_root_thread_id()
    if session_dir is None or root_thread_id is None:
        raise RuntimeError("dispatch_tool 只能在主 Agent 上下文中调用")

    with enter_fork() as depth:
        sub_thread_id = f"sub-{uuid4().hex[:8]}-d{depth}"
        child_query = _build_child_query(demands, platform, category)
        await monitor.report_fork(sub_thread_id, child_query)

        allowed_platforms = [platform] if platform is not None else None
        scoped_plan = _scoped_parent_plan(
            parent_state,
            platform=platform,
            category=category,
        )
        scoped_category = category or (scoped_plan.category if scoped_plan is not None else None)
        leaf_search_mode = platform is not None and scoped_plan is not None
        child_state = initial_state(
            query=child_query,
            thread_id=sub_thread_id,
            user_id=parent_state.get("user_id"),
            long_term_preferences=parent_state.get("long_term_preferences") or [],
            long_term_memory=parent_state.get("long_term_memory") or [],
            allowed_platforms=allowed_platforms,
            allowed_category=scoped_category,
            is_sub_agent=True,
        )
        if scoped_plan is not None:
            child_state["plan"] = scoped_plan
        parent_insight = parent_state.get("insight")
        if parent_insight is not None:
            child_state["insight"] = parent_insight

        prompt = (
            get_scoped_search_prompt(child_query)
            if leaf_search_mode
            else get_sub_agent_prompt(
                demands=child_query,
                long_term_preferences=child_state.get("long_term_preferences") or [],
                long_term_memory=child_state.get("long_term_memory") or [],
            )
        )

        # Delayed import avoids graph_runtime -> tool_registry -> dispatch_tool cycle.
        from app.agent.graph_runtime import ainvoke_agent

        with thread_scope(sub_thread_id, session_dir, root_thread_id=root_thread_id):
            result = await asyncio.wait_for(
                ainvoke_agent(
                    query=child_query,
                    thread_id=sub_thread_id,
                    system_prompt=prompt,
                    initial=child_state,
                    leaf_sub_agent=leaf_search_mode,
                ),
                timeout=sub_agent_timeout_sec(),
            )
        return SubAgentRun(
            sub_thread_id=sub_thread_id,
            final_answer=result.final_text,
            state=result.state,
        )


async def safe_run_sub_agent(
    *,
    demands: str,
    platform: Platform | None,
    category: str | None,
    parent_state: ShopPilotState | dict[str, Any],
) -> SubAgentRun | str:
    """Convert fork failures into model-readable results without crashing the parent."""

    root_thread_id = get_root_thread_id()
    if root_thread_id is None:
        return "[dispatch_tool 失败] 缺少主任务 thread_id。"

    try:
        scheduler = get_fork_scheduler()
        return await scheduler.run(
            root_thread_id=root_thread_id,
            demands=demands,
            platform=platform,
            category=category,
            operation=lambda: run_sub_agent(
                demands=demands,
                platform=platform,
                category=category,
                parent_state=parent_state,
            ),
        )
    except ForkBudgetExceeded as exc:
        await monitor.report_fork_rejected("fork_budget", str(exc))
        return f"[dispatch_tool 拒绝] {exc}。请复用已有结果或由当前 Agent 处理。"
    except ForkQueueFull as exc:
        await monitor.report_fork_rejected("fork_queue_full", str(exc))
        return f"[dispatch_tool 队列已满] {exc}。请基于已有结果继续。"
    except ForkQueueTimeout as exc:
        await monitor.report_fork_rejected("fork_queue_timeout", str(exc))
        return f"[dispatch_tool 排队超时] {exc}。请基于已有结果继续。"
    except ForkLimitExceeded as exc:
        await monitor.report_fork_rejected("fork_limit", str(exc))
        return f"[dispatch_tool 拒绝] {exc}。请由当前 Agent 自己处理。"
    except asyncio.TimeoutError:
        message = f"子 Agent 超过 {sub_agent_timeout_sec()}s"
        await monitor.report_fork_rejected("sub_agent_timeout", message)
        return f"[dispatch_tool 超时] {message}。请基于已有结果继续。"
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        await monitor.report_fork_rejected("sub_agent_error", message)
        return f"[dispatch_tool 失败] {message}"
