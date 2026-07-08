
import os
from langfuse import observe
from langfuse.openai import AsyncOpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

llm_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


def get_health_info() -> dict:
    return {"status": "ok", "base_url": BASE_URL, "model": MODEL_NAME}


@observe()
async def run_chat(messages: list[dict], temperature: float, max_tokens: int) -> str:
    """Non-streaming chat completion. messages = list of {"role", "content"} dicts."""
    completion = await llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


@observe()
async def run_chat_stream(messages: list[dict], temperature: float, max_tokens: int):
    """Streaming chat completion generator. Yields SSE-formatted string chunks."""
    try:
        stream = await llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:

            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {delta}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: [ERROR] {e}\n\n"