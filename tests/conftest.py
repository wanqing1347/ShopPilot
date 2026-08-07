from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
async def isolated_checkpoint_backend(monkeypatch: pytest.MonkeyPatch):
    """Keep unit tests isolated; SQLite tests opt in explicitly."""

    from app.agent.checkpoint import close_checkpointer
    from app.agent.fork_scheduler import close_fork_scheduler
    from app.agent.tool_reliability import reset_tool_reliability

    monkeypatch.setenv("SHOPPILOT_CHECKPOINT_BACKEND", "memory")
    monkeypatch.setenv("SHOPPILOT_RETRIEVAL_EMBEDDING_PROVIDER", "hashing")
    monkeypatch.setenv("SHOPPILOT_KNOWLEDGE_SYNTHESIS_ENABLED", "false")
    await close_fork_scheduler()
    await reset_tool_reliability()
    await close_checkpointer()
    try:
        yield
    finally:
        await close_fork_scheduler()
        await reset_tool_reliability()
        await close_checkpointer()
