from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from app.agent.settings import project_env
from app.api.context import (
    _root_thread_id_var,
    _session_dir_var,
    _thread_id_var,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = (PROJECT_ROOT / (project_env("SHOPPILOT_OUTPUT_DIR", "output") or "output")).resolve()
UPLOAD_ROOT = (PROJECT_ROOT / (project_env("SHOPPILOT_UPLOAD_DIR", "uploaded") or "uploaded")).resolve()
DATA_ROOT = PROJECT_ROOT / "data"


def ensure_session_dir(thread_id: str) -> Path:
    path = OUTPUT_ROOT / thread_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_upload_dir(thread_id: str) -> Path:
    path = UPLOAD_ROOT / thread_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_join(base: Path, *parts: str) -> Path:
    base = base.resolve()
    target = (base / Path(*parts)).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"路径越权: {target}") from exc
    return target


@contextmanager
def thread_scope(
    thread_id: str,
    session_dir: Path,
    root_thread_id: str | None = None,
):
    token_t = _thread_id_var.set(thread_id)
    token_r = _root_thread_id_var.set(root_thread_id or thread_id)
    token_s = _session_dir_var.set(session_dir)
    try:
        yield
    finally:
        _thread_id_var.reset(token_t)
        _root_thread_id_var.reset(token_r)
        _session_dir_var.reset(token_s)
