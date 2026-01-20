import asyncio
from typing import List
from fastapi import WebSocket
import logging

logger = logging.getLogger("mgsa-server")

class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.connections.append(websocket)
        logger.info(f"WebSocket connected. Total={len(self.connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.connections:
                self.connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total={len(self.connections)}")

    async def broadcast(self, message: dict):
        dead: List[WebSocket] = []
        async with self._lock:
            conns = list(self.connections)
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)

manager = ConnectionManager()
