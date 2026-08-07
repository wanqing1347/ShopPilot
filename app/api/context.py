from contextvars import ContextVar
from pathlib import Path

_thread_id_var: ContextVar[str | None] = ContextVar("shoppilot_thread_id", default=None)
_root_thread_id_var: ContextVar[str | None] = ContextVar("shoppilot_root_thread_id", default=None)
_session_dir_var: ContextVar[Path | None] = ContextVar("shoppilot_session_dir", default=None)


def set_thread_context(
    thread_id: str,
    session_dir: Path,
    root_thread_id: str | None = None,
) -> None:
    _thread_id_var.set(thread_id)
    _root_thread_id_var.set(root_thread_id or thread_id)
    _session_dir_var.set(session_dir)


def get_thread_id() -> str | None:
    return _thread_id_var.get()


def get_root_thread_id() -> str | None:
    return _root_thread_id_var.get() or _thread_id_var.get()


def get_session_dir() -> Path | None:
    return _session_dir_var.get()
