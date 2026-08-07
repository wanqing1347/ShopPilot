from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.agent.checkpoint import close_checkpointer
from app.agent.fork_scheduler import close_fork_scheduler
from app.agent.main_agent import run_agent
from app.agent.tool_reliability import reset_tool_reliability


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ShopPilot LangGraph AgentLoop")
    parser.add_argument(
        "query",
        nargs="?",
        default="想买便宜又抗造的旅行收纳三件套，预算300，不要塑料，偏好小众",
    )
    parser.add_argument("--user-id", default="demo-user")
    args = parser.parse_args()

    thread_id = f"cli-{uuid4().hex[:8]}"
    try:
        result = await run_agent(args.query, thread_id, args.user_id)
        print(result.final or result.error or result.status)
        print(f"\nthread_id={thread_id}")
        if result.output_files:
            print("files=" + ", ".join(result.output_files))
    finally:
        await close_fork_scheduler()
        await reset_tool_reliability()
        await close_checkpointer()


if __name__ == "__main__":
    asyncio.run(main())
