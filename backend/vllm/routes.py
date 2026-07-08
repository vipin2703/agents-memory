
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .client import get_health_info, run_chat, run_chat_stream
from .schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["vllm"])


@router.get("/health")
def health_check():
    return get_health_info()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        messages_dicts = [m.model_dump() for m in request.messages]
        response_text = await run_chat(
            messages_dicts,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM upstream error: {e}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    print(request.messages)
    messages_dicts = [m.model_dump() for m in request.messages]
    return StreamingResponse(
        run_chat_stream(
            messages_dicts,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ),
        media_type="text/event-stream",
    )