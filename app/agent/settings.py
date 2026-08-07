from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def project_env(name: str, default: str | None = None) -> str | None:
    """Read a ShopPilot project setting from the environment."""

    return _env(name, default)


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(_env(name, str(default)) or str(default))
    except ValueError:
        return default
    return max(minimum, value)


def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(_env(name, str(default)) or str(default))
    except ValueError:
        return default
    return max(minimum, value)


def _bool_env(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def dataset_dir() -> str:
    return (_env("SHOPPILOT_DATASET_DIR", "data/merged_catalog") or "data/merged_catalog").strip()


def dataset_schema_version() -> int:
    return _int_env("SHOPPILOT_DATASET_SCHEMA_VERSION", 2)


def public_demo_enabled() -> bool:
    return _bool_env("SHOPPILOT_PUBLIC_DEMO_ENABLED", False)


def public_demo_data_file() -> str:
    return (
        _env("SHOPPILOT_PUBLIC_DEMO_DATA_FILE", "data/public_demo/products.jsonl")
        or "data/public_demo/products.jsonl"
    ).strip()


def checkpoint_backend() -> str:
    value = (_env("SHOPPILOT_CHECKPOINT_BACKEND", "sqlite") or "sqlite").strip().lower()
    if value not in {"memory", "sqlite", "postgres"}:
        raise ValueError(
            "SHOPPILOT_CHECKPOINT_BACKEND 只能是 memory、sqlite 或 postgres，"
            f"当前值为 {value!r}"
        )
    return value


def checkpoint_db_file() -> str:
    return (
        _env("SHOPPILOT_CHECKPOINT_DB", "data/checkpoints.sqlite3")
        or "data/checkpoints.sqlite3"
    ).strip()


def checkpoint_postgres_dsn() -> str:
    return (_env("SHOPPILOT_CHECKPOINT_POSTGRES_DSN", "") or "").strip()


def checkpoint_auto_setup() -> bool:
    return _bool_env("SHOPPILOT_CHECKPOINT_AUTO_SETUP", True)


def checkpoint_retention_days() -> int:
    return _int_env("SHOPPILOT_CHECKPOINT_RETENTION_DAYS", 30, minimum=0)


def checkpoint_cleanup_interval_sec() -> float:
    return _float_env("SHOPPILOT_CHECKPOINT_CLEANUP_INTERVAL_SEC", 3600.0, minimum=10.0)


def checkpoint_cleanup_batch_size() -> int:
    return _int_env("SHOPPILOT_CHECKPOINT_CLEANUP_BATCH_SIZE", 100)


def checkpoint_cleanup_scan_limit() -> int:
    return _int_env("SHOPPILOT_CHECKPOINT_CLEANUP_SCAN_LIMIT", 5000)


def checkpoint_cleanup_on_start() -> bool:
    return _bool_env("SHOPPILOT_CHECKPOINT_CLEANUP_ON_START", True)


def max_fork_depth() -> int:
    return _int_env("SHOPPILOT_MAX_FORK_DEPTH", 2, minimum=0)


def max_concurrent_forks() -> int:
    return _int_env("SHOPPILOT_MAX_CONCURRENT_FORKS", 8)


def max_concurrent_forks_per_task() -> int:
    return _int_env("SHOPPILOT_MAX_CONCURRENT_FORKS_PER_TASK", 4)


def max_forks_per_task() -> int:
    return _int_env("SHOPPILOT_MAX_FORKS_PER_TASK", 12)


def max_fork_queue_size() -> int:
    return _int_env("SHOPPILOT_MAX_FORK_QUEUE_SIZE", 64, minimum=0)


def fork_queue_timeout_sec() -> float:
    return _float_env("SHOPPILOT_FORK_QUEUE_TIMEOUT_SEC", 30.0, minimum=0.1)


def fork_dedup_ttl_sec() -> float:
    return _float_env("SHOPPILOT_FORK_DEDUP_TTL_SEC", 300.0, minimum=0.0)


def max_tool_calls() -> int:
    return _int_env("SHOPPILOT_MAX_TOOL_CALLS", 30)


def max_model_steps() -> int:
    return _int_env("SHOPPILOT_MAX_MODEL_STEPS", 30)


def main_timeout_sec() -> int:
    return _int_env("SHOPPILOT_MAIN_TIMEOUT_SEC", 300)


def sub_agent_timeout_sec() -> int:
    return _int_env("SHOPPILOT_SUB_AGENT_TIMEOUT_SEC", 90)


def tool_result_max_chars() -> int:
    return _int_env("SHOPPILOT_TOOL_RESULT_MAX_CHARS", 16_000, minimum=1_000)


def tool_max_retries() -> int:
    return _int_env("SHOPPILOT_TOOL_MAX_RETRIES", 2, minimum=0)


def tool_retry_initial_delay_sec() -> float:
    return _float_env("SHOPPILOT_TOOL_RETRY_INITIAL_DELAY_SEC", 0.5, minimum=0.0)


def tool_retry_max_delay_sec() -> float:
    return _float_env("SHOPPILOT_TOOL_RETRY_MAX_DELAY_SEC", 5.0, minimum=0.0)


def tool_retry_jitter() -> bool:
    return _bool_env("SHOPPILOT_TOOL_RETRY_JITTER", True)


def tool_circuit_failure_threshold() -> int:
    return _int_env("SHOPPILOT_TOOL_CIRCUIT_FAILURE_THRESHOLD", 3)


def tool_circuit_reset_sec() -> float:
    return _float_env("SHOPPILOT_TOOL_CIRCUIT_RESET_SEC", 30.0, minimum=0.1)


def tool_idempotency_ttl_sec() -> float:
    return _float_env("SHOPPILOT_TOOL_IDEMPOTENCY_TTL_SEC", 3600.0, minimum=0.0)


def tool_timeout_llm_sec() -> float:
    return _float_env("SHOPPILOT_TOOL_TIMEOUT_LLM_SEC", 75.0, minimum=1.0)


def tool_timeout_search_sec() -> float:
    return _float_env("SHOPPILOT_TOOL_TIMEOUT_SEARCH_SEC", 30.0, minimum=1.0)


def tool_timeout_compute_sec() -> float:
    return _float_env("SHOPPILOT_TOOL_TIMEOUT_COMPUTE_SEC", 15.0, minimum=1.0)


def tool_timeout_web_sec() -> float:
    return _float_env("SHOPPILOT_TOOL_TIMEOUT_WEB_SEC", 25.0, minimum=1.0)


def tool_timeout_sub_agent_sec() -> float:
    return _float_env(
        "SHOPPILOT_TOOL_TIMEOUT_SUB_AGENT_SEC",
        float(sub_agent_timeout_sec()) + 5.0,
        minimum=1.0,
    )


def context_compaction_trigger_messages() -> int:
    legacy = _env("SHOPPILOT_SUMMARY_TRIGGER_MESSAGES", "40") or "40"
    return _int_env(
        "SHOPPILOT_CONTEXT_COMPACTION_TRIGGER_MESSAGES",
        int(legacy) if legacy.isdigit() else 40,
        minimum=10,
    )


def context_compaction_trigger_chars() -> int:
    return _int_env("SHOPPILOT_CONTEXT_COMPACTION_TRIGGER_CHARS", 48_000, minimum=4_000)


def context_keep_recent_tool_calls() -> int:
    return _int_env("SHOPPILOT_CONTEXT_KEEP_RECENT_TOOL_CALLS", 3, minimum=0)


def context_keep_recent_messages() -> int:
    legacy = _env("SHOPPILOT_SUMMARY_KEEP_MESSAGES", "12") or "12"
    return _int_env(
        "SHOPPILOT_CONTEXT_KEEP_RECENT_MESSAGES",
        int(legacy) if legacy.isdigit() else 12,
        minimum=4,
    )


def context_compaction_min_messages() -> int:
    return _int_env("SHOPPILOT_CONTEXT_COMPACTION_MIN_MESSAGES", 8, minimum=2)


def context_summary_max_chars() -> int:
    return _int_env("SHOPPILOT_CONTEXT_SUMMARY_MAX_CHARS", 6_000, minimum=1_000)


def context_tool_message_max_chars() -> int:
    return _int_env("SHOPPILOT_CONTEXT_TOOL_MESSAGE_MAX_CHARS", 4_000, minimum=500)


def prompt_cache_key_enabled() -> bool:
    return (_env("SHOPPILOT_PROMPT_CACHE_KEY_ENABLED", "false") or "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def prompt_cache_key_prefix() -> str:
    # The value itself is branding-sensitive, so do not inherit a pre-rename
    # cache-key value from the compatibility namespace.
    return (
        os.getenv("SHOPPILOT_PROMPT_CACHE_KEY_PREFIX", "shoppilot-agent-v1")
        or "shoppilot-agent-v1"
    ).strip()


def retrieval_backend() -> str:
    value = (_env("SHOPPILOT_RETRIEVAL_BACKEND", "hybrid") or "hybrid").strip().lower()
    if value not in {"lexical", "vector", "hybrid"}:
        raise ValueError(
            "SHOPPILOT_RETRIEVAL_BACKEND 只能是 lexical、vector 或 hybrid，"
            f"当前值为 {value!r}"
        )
    return value


def retrieval_embedding_provider() -> str:
    value = (
        _env("SHOPPILOT_RETRIEVAL_EMBEDDING_PROVIDER", "hashing") or "hashing"
    ).strip().lower()
    if value not in {"hashing", "sentence_transformers"}:
        raise ValueError(
            "SHOPPILOT_RETRIEVAL_EMBEDDING_PROVIDER 只能是 hashing 或 "
            f"sentence_transformers，当前值为 {value!r}"
        )
    return value


def retrieval_embedding_model() -> str:
    return (
        _env("SHOPPILOT_RETRIEVAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        or "BAAI/bge-small-zh-v1.5"
    ).strip()


def retrieval_embedding_dimension() -> int:
    return _int_env("SHOPPILOT_RETRIEVAL_EMBEDDING_DIMENSION", 384, minimum=64)


def retrieval_embedding_query_prompt() -> str:
    configured = _env("SHOPPILOT_RETRIEVAL_EMBEDDING_QUERY_PROMPT")
    # Blank means "use the provider/model default instruction" rather than
    # disabling it. This also prevents a blank pre-rename local env value from
    # shadowing the ShopPilot BGE default.
    if configured is not None and configured.strip():
        return configured.strip()
    model = retrieval_embedding_model().lower()
    if "bge" in model and "zh" in model:
        return "为这个句子生成表示以用于检索相关文章："
    return ""


def retrieval_index_dir() -> str:
    return (
        _env("SHOPPILOT_RETRIEVAL_INDEX_DIR", "data/retrieval") or "data/retrieval"
    ).strip()


def retrieval_warmup_timeout_sec() -> float:
    return _float_env("SHOPPILOT_RETRIEVAL_WARMUP_TIMEOUT_SEC", 240.0, minimum=30.0)


def retrieval_candidate_pool() -> int:
    return _int_env("SHOPPILOT_RETRIEVAL_CANDIDATE_POOL", 80, minimum=10)


def retrieval_rrf_k() -> int:
    return _int_env("SHOPPILOT_RETRIEVAL_RRF_K", 60, minimum=1)


def retrieval_bm25_weight() -> float:
    return _float_env("SHOPPILOT_RETRIEVAL_BM25_WEIGHT", 1.5, minimum=0.0)


def retrieval_vector_weight() -> float:
    return _float_env("SHOPPILOT_RETRIEVAL_VECTOR_WEIGHT", 0.25, minimum=0.0)


def retrieval_hnsw_m() -> int:
    return _int_env("SHOPPILOT_RETRIEVAL_HNSW_M", 32, minimum=4)


def retrieval_hnsw_ef_construction() -> int:
    return _int_env("SHOPPILOT_RETRIEVAL_HNSW_EF_CONSTRUCTION", 80, minimum=8)


def retrieval_hnsw_ef_search() -> int:
    return _int_env("SHOPPILOT_RETRIEVAL_HNSW_EF_SEARCH", 64, minimum=8)


def retrieval_faiss_threads() -> int:
    return _int_env("SHOPPILOT_RETRIEVAL_FAISS_THREADS", 1, minimum=1)


def retrieval_rerank_weight() -> float:
    return min(
        1.0,
        _float_env("SHOPPILOT_RETRIEVAL_RERANK_WEIGHT", 0.25, minimum=0.0),
    )


def retrieval_reranker() -> str:
    value = (_env("SHOPPILOT_RETRIEVAL_RERANKER", "auto") or "auto").strip().lower()
    if value not in {"auto", "none", "rules", "learned"}:
        raise ValueError(
            "SHOPPILOT_RETRIEVAL_RERANKER 只能是 auto、none、rules 或 learned，"
            f"当前值为 {value!r}"
        )
    return value


def retrieval_reranker_model_file() -> str:
    return (
        _env("SHOPPILOT_RETRIEVAL_RERANKER_MODEL", "app/recall/artifacts/ltr-v1.json")
        or "app/recall/artifacts/ltr-v1.json"
    ).strip()


def retrieval_rerank_top_n() -> int:
    return _int_env("SHOPPILOT_RETRIEVAL_RERANK_TOP_N", 60, minimum=5)


def knowledge_top_k() -> int:
    return _int_env("SHOPPILOT_KNOWLEDGE_TOP_K", 5, minimum=1)


def knowledge_candidate_pool() -> int:
    return _int_env("SHOPPILOT_KNOWLEDGE_CANDIDATE_POOL", 20, minimum=5)


def knowledge_bm25_weight() -> float:
    return _float_env("SHOPPILOT_KNOWLEDGE_BM25_WEIGHT", 1.0, minimum=0.0)


def knowledge_vector_weight() -> float:
    return _float_env("SHOPPILOT_KNOWLEDGE_VECTOR_WEIGHT", 1.0, minimum=0.0)


def knowledge_rrf_k() -> int:
    return _int_env("SHOPPILOT_KNOWLEDGE_RRF_K", 20, minimum=1)


def knowledge_min_evidence() -> int:
    return _int_env("SHOPPILOT_KNOWLEDGE_MIN_EVIDENCE", 2, minimum=1)


def knowledge_synthesis_enabled() -> bool:
    configured = _env("SHOPPILOT_KNOWLEDGE_SYNTHESIS_ENABLED")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    return bool((os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip())


def knowledge_synthesis_max_attempts() -> int:
    return _int_env("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MAX_ATTEMPTS", 2, minimum=1)


def knowledge_synthesis_min_claims() -> int:
    return _int_env("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MIN_CLAIMS", 3, minimum=1)


def knowledge_synthesis_max_claims() -> int:
    return _int_env("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MAX_CLAIMS", 6, minimum=1)


def knowledge_synthesis_min_token_overlap() -> float:
    return min(
        1.0,
        _float_env("SHOPPILOT_KNOWLEDGE_SYNTHESIS_MIN_TOKEN_OVERLAP", 0.05, minimum=0.0),
    )


def memory_file() -> str:
    return (_env("SHOPPILOT_MEMORY_FILE", "data/preferences.json") or "data/preferences.json").strip()


def memory_retrieval_limit() -> int:
    return _int_env("SHOPPILOT_MEMORY_RETRIEVAL_LIMIT", 20)


def memory_min_relevance() -> float:
    return min(
        1.0,
        _float_env("SHOPPILOT_MEMORY_MIN_RELEVANCE", 0.35, minimum=0.0),
    )


def memory_max_entries_per_user() -> int:
    return _int_env("SHOPPILOT_MEMORY_MAX_ENTRIES_PER_USER", 200)


def memory_confidence_increment() -> float:
    return min(
        0.25,
        _float_env("SHOPPILOT_MEMORY_CONFIDENCE_INCREMENT", 0.06, minimum=0.0),
    )


def llm_timeout_sec() -> float:
    return _float_env("LLM_TIMEOUT_SEC", 60.0, minimum=1.0)
