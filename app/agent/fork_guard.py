from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

from app.agent.settings import max_fork_depth

_fork_depth: ContextVar[int] = ContextVar("shoppilot_fork_depth", default=0)


class ForkLimitExceeded(RuntimeError):
    pass


@contextmanager
def enter_fork():
    current = _fork_depth.get()
    limit = max_fork_depth()
    if current >= limit:
        raise ForkLimitExceeded(f"fork 深度超过上限 {limit}")
    token = _fork_depth.set(current + 1)
    try:
        yield current + 1
    finally:
        _fork_depth.reset(token)


def current_fork_depth() -> int:
    return _fork_depth.get()
