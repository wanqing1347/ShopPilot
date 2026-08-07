from __future__ import annotations

import os
from functools import lru_cache

from app.agent.settings import llm_timeout_sec


class AgentConfigurationError(RuntimeError):
    """Raised when the LangGraph runtime is enabled without a usable model."""


@lru_cache(maxsize=1)
def get_llm():
    """Build one OpenAI-compatible chat model shared by main and sub agents."""

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - only possible before dependency sync
        raise AgentConfigurationError(
            "缺少 langchain-openai，请先执行 `uv sync --extra dev`"
        ) from exc

    base_url = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or ""
    ).strip()
    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ("EMPTY" if base_url.startswith(("http://127.0.0.1", "http://localhost")) else "")
    ).strip()
    model = (
        os.getenv("LLM_MODEL_NAME")
        or os.getenv("LLM_MAIN")
        or "qwen-max"
    ).strip()

    if not api_key:
        raise AgentConfigurationError(
            "运行 ShopPilot AgentLoop 必须配置 LLM_API_KEY（或 OPENAI_API_KEY）。"
        )

    try:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    except ValueError:
        temperature = 0.1

    kwargs: dict[str, object] = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "timeout": llm_timeout_sec(),
        "max_retries": 2,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def clear_llm_cache() -> None:
    """Allow tests or configuration reloads to rebuild the model instance."""

    get_llm.cache_clear()
