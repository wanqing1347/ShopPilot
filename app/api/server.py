from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent.checkpoint import (
    checkpoint_description,
    cleanup_expired_checkpoints,
    close_checkpointer,
    delete_thread_checkpoint,
    get_checkpointer,
    read_thread_checkpoint,
)
from app.agent.fork_scheduler import close_fork_scheduler
from app.agent.main_agent import resume_agent, run_agent
from app.agent.settings import (
    checkpoint_cleanup_interval_sec,
    checkpoint_cleanup_on_start,
)
from app.agent.state import state_payload
from app.agent.tool_reliability import reset_tool_reliability
from app.api.connection import manager
from app.api.history import list_task_history, read_task_history, run_task_judge
from app.api.monitor import monitor
from app.evaluation.judge import JudgeConfigurationError
from app.memory.store import store
from app.models import AgentResult
from app.utils.runtime import (
    OUTPUT_ROOT,
    ensure_session_dir,
    ensure_upload_dir,
    safe_join,
    thread_scope,
)

logger = logging.getLogger(__name__)
active_tasks: dict[str, asyncio.Task[None]] = {}
results: dict[str, AgentResult] = {}
_cleanup_task: asyncio.Task[None] | None = None


def _cors_origins() -> list[str]:
    configured = os.getenv(
        "SHOPPILOT_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


async def _run_checkpoint_cleanup() -> dict:
    report = await cleanup_expired_checkpoints(
        active_thread_ids={
            thread_id
            for thread_id, task in active_tasks.items()
            if not task.done()
        }
    )
    for thread_id in report.deleted_threads:
        results.pop(thread_id, None)
    return report.model_dump()


async def _checkpoint_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(checkpoint_cleanup_interval_sec())
        try:
            await _run_checkpoint_cleanup()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Checkpoint retention cleanup failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _cleanup_task

    await get_checkpointer()
    if checkpoint_cleanup_on_start():
        try:
            await _run_checkpoint_cleanup()
        except Exception:
            logger.exception("Checkpoint startup cleanup failed")
    _cleanup_task = asyncio.create_task(
        _checkpoint_cleanup_loop(),
        name="shoppilot-checkpoint-cleanup",
    )
    try:
        yield
    finally:
        if _cleanup_task is not None:
            _cleanup_task.cancel()
            await asyncio.gather(_cleanup_task, return_exceptions=True)
            _cleanup_task = None
        pending = [task for task in active_tasks.values() if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await close_fork_scheduler()
        await reset_tool_reliability()
        await close_checkpointer()


app = FastAPI(
    title="ShopPilot Agent API",
    version="0.12.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None
    user_id: str | None = None


class MemoryResolveRequest(BaseModel):
    memory_id: str = Field(min_length=1)


async def _start_background_task(
    thread_id: str,
    coroutine,
) -> None:
    old = active_tasks.get(thread_id)
    if old and not old.done():
        old.cancel()
        await asyncio.gather(old, return_exceptions=True)

    async def runner() -> None:
        try:
            results[thread_id] = await coroutine
        except asyncio.CancelledError:
            results[thread_id] = AgentResult(
                status="error", thread_id=thread_id, error="cancelled"
            )
            session_dir = ensure_session_dir(thread_id)
            with thread_scope(thread_id, session_dir, root_thread_id=thread_id):
                await monitor.report_task_cancelled()
            raise
        finally:
            active_tasks.pop(thread_id, None)

    task = asyncio.create_task(runner(), name=f"shoppilot-{thread_id}")
    active_tasks[thread_id] = task


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "runtime": "langgraph-agentloop",
        **checkpoint_description(),
    }


@app.post("/api/task")
async def create_task(req: TaskRequest) -> dict[str, str]:
    thread_id = req.thread_id or uuid.uuid4().hex
    await _start_background_task(
        thread_id,
        run_agent(req.query, thread_id, req.user_id),
    )
    return {
        "status": "started",
        "thread_id": thread_id,
        "runtime": "langgraph-agentloop",
    }


@app.post("/api/task/{thread_id}/resume")
async def resume_task(thread_id: str) -> dict[str, str]:
    if await read_thread_checkpoint(thread_id) is None:
        raise HTTPException(404, f"checkpoint 不存在：{thread_id}")
    await _start_background_task(thread_id, resume_agent(thread_id))
    return {"status": "resuming", "thread_id": thread_id}


@app.get("/api/task/{thread_id}")
async def get_task(thread_id: str) -> dict:
    task = active_tasks.get(thread_id)
    if task and not task.done():
        return {"status": "running", "thread_id": thread_id}
    result = results.get(thread_id)
    if result is not None:
        return result.model_dump(mode="json")
    checkpoint = await read_thread_checkpoint(thread_id)
    if checkpoint is not None:
        return {
            "status": "checkpointed",
            "thread_id": thread_id,
            "terminated": bool(checkpoint.get("terminated")),
            "terminal_tool": checkpoint.get("terminal_tool"),
        }
    raise HTTPException(404, f"任务 {thread_id} 不存在")


@app.get("/api/task/{thread_id}/checkpoint")
async def get_checkpoint(thread_id: str) -> dict:
    state = await read_thread_checkpoint(thread_id)
    if state is None:
        raise HTTPException(404, f"checkpoint 不存在：{thread_id}")
    return {
        "thread_id": thread_id,
        "checkpoint": state_payload(state),
    }


@app.delete("/api/task/{thread_id}/checkpoint")
async def delete_checkpoint(thread_id: str) -> dict[str, str]:
    task = active_tasks.get(thread_id)
    if task and not task.done():
        raise HTTPException(409, "任务运行中，不能删除 checkpoint")
    if not await delete_thread_checkpoint(thread_id):
        raise HTTPException(404, f"checkpoint 不存在：{thread_id}")
    results.pop(thread_id, None)
    return {"status": "deleted", "thread_id": thread_id}


@app.post("/api/checkpoints/cleanup")
async def cleanup_checkpoints() -> dict:
    return await _run_checkpoint_cleanup()


@app.get("/api/users/{user_id}/memories")
async def get_user_memories(
    user_id: str,
    query: str = "",
    include_inactive: bool = True,
) -> dict:
    if query.strip():
        result = await store.read_relevant(user_id=user_id, query=query)
        entries = result.entries
    else:
        entries = await store.list_entries(
            user_id,
            include_inactive=include_inactive,
        )
    return {
        "user_id": user_id,
        "query": query,
        "memories": [entry.model_dump(mode="json") for entry in entries],
    }


@app.delete("/api/users/{user_id}/memories/{memory_id}")
async def delete_user_memory(user_id: str, memory_id: str) -> dict[str, str]:
    if not await store.delete_entry(user_id, memory_id):
        raise HTTPException(404, f"memory 不存在：{memory_id}")
    return {"status": "deleted", "user_id": user_id, "memory_id": memory_id}


@app.post("/api/users/{user_id}/memories/resolve")
async def resolve_user_memory(
    user_id: str,
    request: MemoryResolveRequest,
) -> dict:
    selected = await store.resolve_entry(user_id, request.memory_id)
    if selected is None:
        raise HTTPException(404, f"memory 不存在：{request.memory_id}")
    return {
        "status": "resolved",
        "user_id": user_id,
        "memory": selected.model_dump(mode="json"),
    }


@app.get("/api/history")
async def get_history(user_id: str | None = None, limit: int = 50) -> dict:
    return {
        "user_id": user_id,
        "items": list_task_history(user_id=user_id, limit=limit),
    }


@app.get("/api/history/{thread_id}")
async def get_history_detail(thread_id: str, user_id: str | None = None) -> dict:
    history = read_task_history(thread_id, user_id=user_id)
    if history is None:
        raise HTTPException(404, f"历史任务不存在：{thread_id}")
    return history


@app.get("/api/history/{thread_id}/evaluation")
async def get_history_evaluation(thread_id: str, user_id: str | None = None) -> dict:
    history = read_task_history(thread_id, user_id=user_id)
    if history is None:
        raise HTTPException(404, f"历史任务不存在：{thread_id}")
    return {
        "thread_id": thread_id,
        "evaluation": history["evaluation"],
    }


@app.post("/api/history/{thread_id}/judge")
async def run_history_judge(
    thread_id: str,
    user_id: str | None = None,
    force: bool = False,
) -> dict:
    try:
        result = await run_task_judge(
            thread_id,
            user_id=user_id,
            force=force,
        )
    except JudgeConfigurationError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.exception("LLM Judge failed for thread %s", thread_id)
        raise HTTPException(502, f"LLM Judge 调用失败：{type(exc).__name__}: {exc}") from exc
    if result is None:
        raise HTTPException(404, f"历史任务不存在：{thread_id}")
    return {"thread_id": thread_id, **result}


@app.get("/api/task/{thread_id}/events")
async def get_events(thread_id: str) -> dict:
    return {"thread_id": thread_id, "events": manager.get_history(thread_id)}


@app.post("/api/task/{thread_id}/cancel")
async def cancel_task(thread_id: str) -> dict[str, str]:
    task = active_tasks.get(thread_id)
    if not task or task.done():
        raise HTTPException(404, f"任务 {thread_id} 不存在或已结束")
    task.cancel()
    return {"status": "cancelling", "thread_id": thread_id}


@app.websocket("/ws/{thread_id}")
async def ws_endpoint(websocket: WebSocket, thread_id: str) -> None:
    await manager.connect(websocket, thread_id)
    try:
        await websocket.send_json(
            {
                "type": "monitor_event",
                "event": "session_created",
                "message": "会话已创建",
                "data": {"thread_id": thread_id},
            }
        )
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await manager.disconnect(websocket, thread_id)


@app.get("/api/files/{thread_id}/{filename}")
async def download_file(thread_id: str, filename: str) -> FileResponse:
    session_dir = OUTPUT_ROOT / thread_id
    if not session_dir.exists():
        raise HTTPException(404, "会话不存在")
    try:
        target = safe_join(session_dir, filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not target.is_file():
        raise HTTPException(404, f"文件不存在：{filename}")
    return FileResponse(target, filename=target.name)


@app.post("/api/upload")
async def upload_file(thread_id: str, file: UploadFile = File(...)) -> dict[str, str]:
    upload_dir = ensure_upload_dir(thread_id)
    safe_name = Path(file.filename or "upload.bin").name
    target = safe_join(upload_dir, safe_name)
    target.write_bytes(await file.read())
    return {"status": "ok", "path": str(target)}
