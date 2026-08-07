from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    """thread_id -> WebSocket router with a small replay buffer."""

    def __init__(self, history_size: int = 200) -> None:
        self.active: dict[str, WebSocket] = {}
        self.history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, thread_id: str) -> None:
        await websocket.accept()
        async with self._lock:
            self.active[thread_id] = websocket
            replay = list(self.history.get(thread_id, ()))
        for payload in replay:
            await websocket.send_json(payload)

    async def disconnect(self, websocket: WebSocket, thread_id: str) -> None:
        async with self._lock:
            if self.active.get(thread_id) is websocket:
                del self.active[thread_id]

    async def send_to_thread(
        self,
        payload: dict[str, Any],
        thread_id: str,
        *,
        record_history: bool = True,
    ) -> None:
        if record_history:
            self.history[thread_id].append(payload)
        ws = self.active.get(thread_id)
        if ws is None:
            return
        try:
            await ws.send_json(payload)
        except Exception:
            await self.disconnect(ws, thread_id)

    def get_history(self, thread_id: str) -> list[dict[str, Any]]:
        return list(self.history.get(thread_id, ()))


manager = ConnectionManager()
