from __future__ import annotations

import asyncio
import os

os.environ.setdefault("SHOPPILOT_CHECKPOINT_BACKEND", "memory")
os.environ.setdefault("LLM_BASE_URL", "http://127.0.0.1:9999/v1")
os.environ.setdefault("LLM_API_KEY", "EMPTY")

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent.checkpoint import close_checkpointer
from app.agent.graph_runtime import build_agent
from app.agent.llm import clear_llm_cache
from app.agent.tool_registry import get_full_tool_set


async def main() -> None:
    clear_llm_cache()
    try:
        agent = await build_agent("verify")
        print(type(agent).__name__)
        print([tool.name for tool in get_full_tool_set()])
        print(AsyncPostgresSaver.__name__)
    finally:
        await close_checkpointer()


if __name__ == "__main__":
    asyncio.run(main())
