import asyncio
import json
from uuid import UUID

import redis.asyncio as redis
from fastapi import APIRouter
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from config import get_settings


settings = get_settings()

router = APIRouter()


redis_client = redis.from_url(
    settings.REDIS_URL,
    decode_responses=True
)


class ConnectionManager:

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(
        self,
        project_id: str,
        websocket: WebSocket
    ) -> None:
        await websocket.accept()

        self.active_connections[project_id] = websocket

    def disconnect(
        self,
        project_id: str
    ) -> None:
        if project_id in self.active_connections:
            del self.active_connections[project_id]

    async def send(
        self,
        project_id: str,
        message: dict
    ) -> None:
        websocket = self.active_connections.get(project_id)

        if not websocket:
            return

        await websocket.send_json(message)


manager = ConnectionManager()


async def redis_listener(
    project_id: str,
    manager: ConnectionManager
) -> None:
    pubsub = redis_client.pubsub()

    channel = f"project:{project_id}:events"

    await pubsub.subscribe(channel)

    try:
        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=1.0
            )

            if not message:
                await asyncio.sleep(0.1)
                continue

            payload = json.loads(message["data"])

            await manager.send(
                project_id,
                payload
            )

    finally:
        await pubsub.unsubscribe(channel)

        await pubsub.close()


@router.websocket("/{project_id}")
async def ws_endpoint(
    websocket: WebSocket,
    project_id: UUID
) -> None:
    project_key = str(project_id)

    await manager.connect(
        project_key,
        websocket
    )

    listener_task = asyncio.create_task(
        redis_listener(
            project_key,
            manager
        )
    )

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        manager.disconnect(project_key)

    except Exception:
        manager.disconnect(project_key)

    finally:
        listener_task.cancel()

        try:
            await listener_task
        except asyncio.CancelledError:
            pass