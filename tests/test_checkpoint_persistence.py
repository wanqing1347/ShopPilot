from __future__ import annotations

from pathlib import Path

import pytest
from langchain.messages import AIMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

from app.agent import graph_runtime
from app.agent.checkpoint import (
    close_checkpointer,
    delete_thread_checkpoint,
    read_thread_checkpoint,
)
from app.agent.state import initial_state


class _ToolCapableFakeChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@pytest.mark.asyncio
async def test_sqlite_checkpoint_survives_connection_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "shoppilot-checkpoints.sqlite3"
    monkeypatch.setenv("SHOPPILOT_CHECKPOINT_BACKEND", "sqlite")
    monkeypatch.setenv("SHOPPILOT_CHECKPOINT_DB", str(db_path))
    monkeypatch.setattr(
        graph_runtime,
        "get_llm",
        lambda: _ToolCapableFakeChatModel(
            responses=[AIMessage(content="持久化完成")]
        ),
    )
    await close_checkpointer()

    thread_id = "sqlite-restart-test"
    result = await graph_runtime.ainvoke_agent(
        query="保存这个会话",
        thread_id=thread_id,
        system_prompt="You are a checkpoint test agent.",
        initial=initial_state(
            query="保存这个会话",
            thread_id=thread_id,
            user_id="checkpoint-user",
        ),
    )
    assert result.final_text == "持久化完成"
    assert db_path.is_file()

    # Close and reopen the SQLite connection to simulate a process restart.
    await close_checkpointer()
    recovered_run = await graph_runtime.aresume_agent(
        thread_id=thread_id,
        system_prompt="You are a checkpoint test agent.",
    )
    assert recovered_run.resumed is True
    assert recovered_run.final_text == "持久化完成"

    recovered = await read_thread_checkpoint(thread_id)
    assert recovered is not None
    assert recovered["thread_id"] == thread_id
    assert recovered["query"] == "保存这个会话"
    assert recovered["user_id"] == "checkpoint-user"
    assert recovered["messages"][-1].content == "持久化完成"

    assert await delete_thread_checkpoint(thread_id) is True
    assert await read_thread_checkpoint(thread_id) is None
    await close_checkpointer()
