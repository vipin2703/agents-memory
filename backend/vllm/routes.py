"""
vLLM routes — structured stream only.

Agent memory + tools always enabled on the backend (not client-controlled).
"""

from __future__ import annotations

import json
import logging
import os
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .client import get_health_info, run_chat_structured_stream
from .schemas import ChatRequest, MemoryStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vllm"])

# Backend policy — not from client flags
DEFAULT_USER_ID = os.getenv("MEMORY_DEFAULT_USER_ID", "default-user")


@router.get("/health")
def health_check():
    return get_health_info()


def _resolve_identity(request: ChatRequest) -> tuple[str, str]:
    """
    Who is chatting. Client may pass user_id/session_id as identity only.
    Backend always runs memory/tools — never a client on/off switch.
    """
    user_id = (request.user_id or "").strip() or DEFAULT_USER_ID
    session_id = (request.session_id or "").strip() or str(uuid.uuid4())
    return user_id, session_id


async def _persist(
    *,
    user_id: str,
    session_id: str,
    messages_dicts: list[dict],
    answer: str,
    extracted_facts,
    mem_status: MemoryStatus,
) -> MemoryStatus:
    from agent_memory.bridge import latest_user_text, persist_chat_turn

    persist = await persist_chat_turn(
        user_id=user_id,
        session_id=session_id,
        user_text=latest_user_text(messages_dicts),
        assistant_text=answer,
        extracted_facts=extracted_facts,
    )
    mem_status.wrote_user = bool(persist.get("wrote_user"))
    mem_status.wrote_assistant = bool(persist.get("wrote_assistant"))
    mem_status.errors.extend(persist.get("errors") or [])
    return mem_status


@router.post("/chat/structured/stream")
async def chat_structured_stream(request: ChatRequest):
    """
    SSE: answer_delta | tool_call | tool_result | final | error | [DONE]

    Memory tools + persist always on (backend).
    """
    messages_dicts = [m.model_dump() for m in request.messages]
    user_id, session_id = _resolve_identity(request)

    async def _gen():
        # Always enabled — model decides whether to CALL tools, not client.
        mem_status = MemoryStatus(enabled=True)
        try:
            final_payload = None
            async for event in run_chat_structured_stream(
                messages_dicts,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools_enabled=True,
                user_id=user_id,
                session_id=session_id,
            ):
                if event.get("type") == "final":
                    final_payload = event
                    continue
                if event.get("type") == "error":
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if final_payload is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'no final payload'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            facts = final_payload.get("extracted_facts") or {}
            tools_used = list(final_payload.get("tools_used") or [])
            mem_status.tools_used = tools_used
            mem_status.recalled = bool(final_payload.get("recalled")) or any(
                t in tools_used
                for t in ("search_conversation", "search_context", "search_memory")
            )
            mem_status = await _persist(
                user_id=user_id,
                session_id=session_id,
                messages_dicts=messages_dicts,
                answer=final_payload.get("answer") or "",
                extracted_facts=facts,
                mem_status=mem_status,
            )
            out = {
                "type": "final",
                "answer": final_payload.get("answer") or "",
                "extracted_facts": facts,
                "finish_reason": final_payload.get("finish_reason"),
                "tools_used": tools_used,
                "user_id": user_id,
                "session_id": session_id,
                "memory_status": mem_status.model_dump(),
            }
            yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("chat_structured_stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")
