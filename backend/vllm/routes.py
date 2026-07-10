
# from fastapi import APIRouter, HTTPException
# from fastapi.responses import StreamingResponse

# from .client import get_health_info, run_chat, run_chat_stream
# from .schemas import ChatRequest, ChatResponse

# router = APIRouter(tags=["vllm"])


# @router.get("/health")
# def health_check():
#     return get_health_info()


# @router.post("/chat", response_model=ChatResponse)
# async def chat(request: ChatRequest):
#     try:
#         messages_dicts = [m.model_dump() for m in request.messages]
#         response_text = await run_chat(
#             messages_dicts,
#             temperature=request.temperature,
#             max_tokens=request.max_tokens,
#         )
#         return ChatResponse(response=response_text)
#     except Exception as e:
#         raise HTTPException(status_code=502, detail=f"LLM upstream error: {e}")


# @router.post("/chat/stream")
# async def chat_stream(request: ChatRequest):
#     print(request.messages)
#     messages_dicts = [m.model_dump() for m in request.messages]
#     return StreamingResponse(
#         run_chat_stream(
#             messages_dicts,
#             temperature=request.temperature,
#             max_tokens=request.max_tokens,
#         ),
#         media_type="text/event-stream",
#     )







"""
vllm_service/routes.py -- vLLM service ke saare API endpoints yahi honge.
Business logic client.py se import hota hai, models schemas.py se.
Naya vLLM-related endpoint add karna ho to bas yaha ek naya @router.<method> likho.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .client import get_health_info, run_chat, run_chat_stream, run_chat_structured
from .schemas import ChatRequest, ChatResponse, StructuredChatOutput

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


@router.post("/chat/structured", response_model=StructuredChatOutput)
async def chat_structured(request: ChatRequest):
    """
    Guided-decoding endpoint -- ek hi LLM call se user-facing 'answer' aur
    is turn ka compressed 'summary_fact' dono milte hain. summary_fact ko
    caller (chat_client.py ya koi frontend) apni khud ki persistent
    facts-list me store kar sakta hai, taaki lambi conversation me bhi
    important cheezein na bhoole bina poori history baar-baar bheje.
    """
    try:
        messages_dicts = [m.model_dump() for m in request.messages]
        result = await run_chat_structured(
            messages_dicts,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM upstream error: {e}")


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    messages_dicts = [m.model_dump() for m in request.messages]
    return StreamingResponse(
        run_chat_stream(
            messages_dicts,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ),
        media_type="text/event-stream",
    )