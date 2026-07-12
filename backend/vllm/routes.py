"""
vllm routes — chat + structured (+ live structured stream) + agent memory.
"""

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .client import (
    get_health_info,
    run_chat,
    run_chat_stream,
    run_chat_structured,
    run_chat_structured_stream,
)
from .schemas import ChatRequest, ChatResponse, MemoryStatus, StructuredChatOutput

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vllm"])


@router.get("/health")
def health_check():
    return get_health_info()


def _wants_agent_memory(request: ChatRequest) -> bool:
    return bool(
        request.use_agent_memory
        and request.user_id
        and request.session_id
    )


async def _prepare_memory(request: ChatRequest, messages_dicts: list[dict]):
    memory = request.memory.model_dump() if request.memory else None
    extra_block = None
    mem_status = MemoryStatus(enabled=False)
    if not _wants_agent_memory(request):
        return memory, extra_block, mem_status

    from agent_memory.bridge import prepare_chat_memory

    mem_status.enabled = True
    prep = await prepare_chat_memory(
        user_id=request.user_id,  # type: ignore[arg-type]
        session_id=request.session_id,  # type: ignore[arg-type]
        messages=messages_dicts,
        client_memory=memory,
    )
    memory = prep["memory_dict"]
    extra_block = prep.get("memory_block") or None
    mem_status.recalled = bool(prep.get("recalled"))
    mem_status.errors.extend(prep.get("errors") or [])
    return memory, extra_block, mem_status


async def _persist(
    request: ChatRequest,
    messages_dicts: list[dict],
    answer: str,
    extracted_facts,
    mem_status: MemoryStatus,
) -> MemoryStatus:
    if not _wants_agent_memory(request):
        return mem_status
    from agent_memory.bridge import latest_user_text, persist_chat_turn

    persist = await persist_chat_turn(
        user_id=request.user_id,  # type: ignore[arg-type]
        session_id=request.session_id,  # type: ignore[arg-type]
        user_text=latest_user_text(messages_dicts),
        assistant_text=answer,
        extracted_facts=extracted_facts,
    )
    mem_status.wrote_user = bool(persist.get("wrote_user"))
    mem_status.wrote_assistant = bool(persist.get("wrote_assistant"))
    mem_status.errors.extend(persist.get("errors") or [])
    return mem_status


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        messages_dicts = [m.model_dump() for m in request.messages]
        memory, extra_block, mem_status = await _prepare_memory(
            request, messages_dicts
        )
        response_text = await run_chat(
            messages_dicts,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            memory=memory,
            extra_memory_block=extra_block,
        )
        await _persist(request, messages_dicts, response_text, None, mem_status)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM upstream error: {e}")


@router.post("/chat/structured", response_model=StructuredChatOutput)
async def chat_structured(request: ChatRequest):
    """Non-stream structured: full answer + facts (1 LLM call)."""
    try:
        messages_dicts = [m.model_dump() for m in request.messages]
        memory, extra_block, mem_status = await _prepare_memory(
            request, messages_dicts
        )
        result = await run_chat_structured(
            messages_dicts,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            memory=memory,
            extra_memory_block=extra_block,
        )
        mem_status = await _persist(
            request,
            messages_dicts,
            result.answer,
            result.extracted_facts,
            mem_status,
        )
        result.memory_status = mem_status
        return result
    except Exception as e:
        logger.exception("chat_structured failed")
        raise HTTPException(status_code=502, detail=f"LLM upstream error: {e}")


@router.post("/chat/structured/stream")
async def chat_structured_stream(request: ChatRequest):
    """
    Live answer tokens + final facts (still 1 LLM call).

    SSE events (JSON lines):
      {"type":"answer_delta","text":"..."}
      {"type":"final","answer":"...","extracted_facts":{...},"memory_status":{...}}
      {"type":"error","message":"..."}
      [DONE]
    """
    messages_dicts = [m.model_dump() for m in request.messages]

    async def _gen():
        try:
            memory, extra_block, mem_status = await _prepare_memory(
                request, messages_dicts
            )
            final_payload = None
            async for event in run_chat_structured_stream(
                messages_dicts,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                memory=memory,
                extra_memory_block=extra_block,
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
            mem_status = await _persist(
                request,
                messages_dicts,
                final_payload.get("answer") or "",
                facts,
                mem_status,
            )
            out = {
                "type": "final",
                "answer": final_payload.get("answer") or "",
                "extracted_facts": facts,
                "finish_reason": final_payload.get("finish_reason"),
                "memory_status": mem_status.model_dump(),
            }
            yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.exception("chat_structured_stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    messages_dicts = [m.model_dump() for m in request.messages]

    async def _gen():
        memory, extra_block, mem_status = await _prepare_memory(
            request, messages_dicts
        )
        full: list[str] = []
        async for chunk in run_chat_stream(
            messages_dicts,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            memory=memory,
            extra_memory_block=extra_block,
        ):
            if (
                chunk.startswith("data: ")
                and not chunk.startswith("data: [DONE]")
                and not chunk.startswith("data: [ERROR]")
            ):
                full.append(chunk[len("data: ") :].rstrip("\n"))
            yield chunk

        if full:
            await _persist(
                request, messages_dicts, "".join(full).strip(), None, mem_status
            )

    return StreamingResponse(_gen(), media_type="text/event-stream")
