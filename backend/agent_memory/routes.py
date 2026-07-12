"""
Agent memory HTTP API.

  GET    /memory/health
  POST   /memory/write
  POST   /memory/recall
  DELETE /memory/session
  DELETE /memory/user/{user_id}
"""

from fastapi import APIRouter, HTTPException, Query

from . import client
from .schemas import (
    MemoryHealth,
    MemoryRecallRequest,
    MemoryRecallResponse,
    MemoryWriteRequest,
    MemoryWriteResult,
)

router = APIRouter(prefix="/memory", tags=["agent-memory"])


@router.get("/health", response_model=MemoryHealth)
async def health():
    return await client.memory_health()


@router.post("/write", response_model=MemoryWriteResult)
async def write(req: MemoryWriteRequest):
    """
    Ek turn: SQL append + Elasticsearch index + KG upsert.
    Partial success allowed — sql_ok / search_ok / graph_ok dekho.
    """
    try:
        result = await client.write_memory(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    if not (result.sql_ok or result.search_ok or result.graph_ok):
        raise HTTPException(
            status_code=502,
            detail={"message": "all memory stores failed", "errors": result.errors},
        )
    return result


@router.post("/recall", response_model=MemoryRecallResponse)
async def recall(req: MemoryRecallRequest):
    """Recent SQL + optional Elasticsearch query + KG facts → memory_block."""
    try:
        return await client.recall_memory(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/session")
async def delete_session(
    user_id: str = Query(...),
    session_id: str = Query(...),
):
    """Session transcript clear (SQL + Elasticsearch). KG long-term facts rehte hain."""
    try:
        return await client.clear_session(user_id, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/user/{user_id}")
async def delete_user(user_id: str):
    """Poora user wipe: SQL + Elasticsearch + Knowledge Graph."""
    try:
        return await client.clear_user(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
