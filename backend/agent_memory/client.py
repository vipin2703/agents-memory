"""
Thin facade — routes yahi se service call karti hain.
Stores alag packages me: sql / elasticsearch / knowledge_graph.
"""

from .bridge import persist_chat_turn
from .schemas import (
    MemoryHealth,
    MemoryRecallRequest,
    MemoryRecallResponse,
    MemoryWriteRequest,
    MemoryWriteResult,
)
from .service import AgentMemoryService, get_memory_service


async def write_memory(req: MemoryWriteRequest) -> MemoryWriteResult:
    return await get_memory_service().write_turn(req)


async def recall_memory(req: MemoryRecallRequest) -> MemoryRecallResponse:
    return await get_memory_service().recall(
        user_id=req.user_id,
        session_id=req.session_id,
        query=req.query,
        recent_limit=req.recent_limit,
        search_limit=req.search_limit,
        graph_limit=req.graph_limit,
    )


async def memory_health() -> MemoryHealth:
    return await get_memory_service().health()


async def memory_startup() -> None:
    await get_memory_service().startup()


async def memory_shutdown() -> None:
    await get_memory_service().shutdown()


async def clear_session(user_id: str, session_id: str) -> dict:
    return await get_memory_service().clear_session(user_id, session_id)


async def clear_user(user_id: str) -> dict:
    return await get_memory_service().clear_user(user_id)


__all__ = [
    "AgentMemoryService",
    "get_memory_service",
    "write_memory",
    "recall_memory",
    "memory_health",
    "memory_startup",
    "memory_shutdown",
    "clear_session",
    "clear_user",
    "persist_chat_turn",
]
