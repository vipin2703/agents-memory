
# import os
# from langfuse import observe
# from langfuse.openai import AsyncOpenAI

# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass


# BASE_URL = os.getenv("BASE_URL")
# API_KEY = os.getenv("API_KEY")
# MODEL_NAME = os.getenv("MODEL_NAME")

# llm_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


# def get_health_info() -> dict:
#     return {"status": "ok", "base_url": BASE_URL, "model": MODEL_NAME}


# @observe()
# async def run_chat(messages: list[dict], temperature: float, max_tokens: int) -> str:
#     """Non-streaming chat completion. messages = list of {"role", "content"} dicts."""
#     completion = await llm_client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=messages,
#         temperature=temperature,
#         max_tokens=max_tokens,
#     )
#     return completion.choices[0].message.content


# @observe()
# async def run_chat_stream(messages: list[dict], temperature: float, max_tokens: int):
#     """Streaming chat completion generator. Yields SSE-formatted string chunks."""
#     try:
#         stream = await llm_client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=messages,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             stream=True,
#             stream_options={"include_usage": True},
#         )
#         async for chunk in stream:

#             if chunk.choices:
#                 delta = chunk.choices[0].delta.content
#                 if delta:
#                     yield f"data: {delta}\n\n"
#         yield "data: [DONE]\n\n"
#     except Exception as e:
#         yield f"data: [ERROR] {e}\n\n"






# """
# vllm_service/client.py -- Local vLLM ke sath saara interaction yahi handle karega.
# Sirf business logic. Koi FastAPI route yaha nahi hoga.

# LANGFUSE TRACING:
#   - `langfuse.openai` ka AsyncOpenAI drop-in wrapper use kar rahe hain --
#     isse har LLM call (prompt, response, tokens, latency, cost) automatically
#     Langfuse me trace ho jaata hai.
#   - @observe() decorator function-level trace banata hai.

# GUIDED DECODING (structured output):
#   - `run_chat_structured()` vLLM ke response_format={"type": "json_schema"}
#     feature use karta hai -- ye vLLM ko token-level pe FORCE karta hai ki
#     output hamesha StructuredChatOutput schema follow kare (answer +
#     summary_fact). Isse ek hi LLM call se:
#       1. User ko dikhane wala normal jawab (answer)
#       2. Is turn ka compressed fact/summary (summary_fact) -- jo history
#          compress karne ke liye baad me use hoga
#     dono milte hain, bina extra LLM call ke aur bina format-break hone ke
#     dar ke (jaisa prompt-based JSON asking me hota hai).

# NOTE: temperature/max_tokens ki default value sirf schemas.py (ChatRequest)
# me hai -- yahi single source of truth hai.
# """

# import os
# import json
# from langfuse import observe
# from langfuse.openai import AsyncOpenAI

# from .schemas import StructuredChatOutput

# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass

# # -----------------------------------------------------------
# # Local vLLM config -- seedha localhost:8000 pe already chal
# # raha vLLM server hit karega.
# # -----------------------------------------------------------
# BASE_URL = os.getenv("BASE_URL")
# API_KEY = os.getenv("API_KEY")
# MODEL_NAME = os.getenv("MODEL_NAME")

# # Langfuse apna config in env vars se khud utha leta hai:
# #   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

# llm_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


# def get_health_info() -> dict:
#     return {"status": "ok", "base_url": BASE_URL, "model": MODEL_NAME}


# @observe()
# async def run_chat(messages: list[dict], temperature: float, max_tokens: int) -> str:
#     """Non-streaming chat completion (plain text, no structured output). messages = list of {"role", "content"} dicts."""
#     completion = await llm_client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=messages,
#         temperature=temperature,
#         max_tokens=max_tokens,
#     )
#     return completion.choices[0].message.content


# @observe()
# async def run_chat_structured(messages: list[dict], temperature: float, max_tokens: int) -> StructuredChatOutput:
#     """
#     Non-streaming chat completion JISME guided decoding lagi hai -- vLLM
#     hamesha valid {"answer": ..., "extracted_facts": {...}} JSON return karega.

#     SIRF EK INSTRUCTION SOURCE: pehle system-message + user-suffix-reminder
#     + schema teeno se overlapping instruction ja rahi thi -- chhote model
#     (gemma4) ke liye ye confusing tha aur safe-fallback me jaake sirf input
#     echo kar raha tha. Ab sirf response_format (JSON schema) hi instruction
#     ka source hai -- guided decoding khud token-level pe enforce karta hai,
#     alag se prompt-instruction ki zarurat hi nahi.

#     Returns: StructuredChatOutput (parsed Pydantic object, .answer aur
#              .extracted_facts attributes se access karo)
#     """

#     print(json.dumps({"model": MODEL_NAME,
#             "messages": messages,
#               "temperature": temperature, 
#               "max_tokens": max_tokens, 
#               "repetition_penalty": 1.3,
#                 "response_format": 
#                 {"type": "json_schema", 
#                  "json_schema": 
#                  {"name": "structured_chat_output", 
#                   "schema": StructuredChatOutput.model_json_schema()}}},
#                     indent=2, ensure_ascii=False))
#     completion = await llm_client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=messages,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         extra_body={"repetition_penalty": 1.3},  # repetition-loop se bachne ke liye
#         response_format={
#             "type": "json_schema",
#             "json_schema": {
#                 "name": "structured_chat_output",
#                 "schema": StructuredChatOutput.model_json_schema(),
#             },
#         },
#     )
#     # print(llm_client.chat.completions.create)
#     print(completion)
#     raw_json = completion.choices[0].message.content
#     try:
#         parsed = json.loads(raw_json)
#     except json.JSONDecodeError as e:
#         # Model ka output beech me hi kat gaya (max_tokens khatam ho gaya
#         # JSON complete hone se pehle) -- ye truncation hai, koi aur bug
#         # nahi. Fix: is call ke max_tokens ko badhao (caller/schemas.py se),
#         # ya answer ki length ko chhota rakhne ka instruction do.
#         raise ValueError(
#             f"Model returned incomplete/invalid JSON (likely truncated by "
#             f"max_tokens={max_tokens}). Raw output length: {len(raw_json)} chars. "
#             f"Original error: {e}"
#         )
#     return StructuredChatOutput(**parsed)


# @observe()
# async def run_chat_stream(messages: list[dict], temperature: float, max_tokens: int):
#     """
#     Streaming chat completion generator. Yields SSE-formatted string chunks.

#     NOTE: Guided decoding (structured JSON output) streaming ke saath is
#     file me use nahi kiya -- token-by-token JSON chunks ko live parse karna
#     complex hai aur user ko half-JSON dikhta agar bina buffering ke stream
#     kiya jaaye. Structured output sirf run_chat_structured() (non-streaming)
#     me use hota hai. Agar streaming + structured dono chahiye ho future me,
#     poora JSON buffer karke ek hi baar client ko bhejna padega (ab streaming
#     ka fayda hi khatam ho jaata hai us case me).
#     """
#     try:
#         stream = await llm_client.chat.completions.create(
#             model=MODEL_NAME,
#             messages=messages,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             stream=True,
#             stream_options={"include_usage": True},
#         )
#         async for chunk in stream:
#             if chunk.choices:
#                 delta = chunk.choices[0].delta.content
#                 if delta:
#                     yield f"data: {delta}\n\n"
#         yield "data: [DONE]\n\n"
#     except Exception as e:
#         yield f"data: [ERROR] {e}\n\n"







"""
vllm_service/client.py -- Local vLLM ke sath saara interaction yahi handle karega.
Sirf business logic. Koi FastAPI route yaha nahi hoga.

LANGFUSE TRACING:
  - `langfuse.openai` ka AsyncOpenAI drop-in wrapper use kar rahe hain --
    isse har LLM call (prompt, response, tokens, latency, cost) automatically
    Langfuse me trace ho jaata hai.
  - @observe() decorator function-level trace banata hai.

GUIDED DECODING (structured output):
  - `run_chat_structured()` vLLM ke response_format={"type": "json_schema"}
    feature use karta hai -- ye vLLM ko token-level pe FORCE karta hai ki
    output hamesha StructuredChatOutput schema follow kare (answer +
    summary_fact). Isse ek hi LLM call se:
      1. User ko dikhane wala normal jawab (answer)
      2. Is turn ka compressed fact/summary (summary_fact) -- jo history
         compress karne ke liye baad me use hoga
    dono milte hain, bina extra LLM call ke aur bina format-break hone ke
    dar ke (jaisa prompt-based JSON asking me hota hai).

NOTE: temperature/max_tokens ki default value sirf schemas.py (ChatRequest)
me hai -- yahi single source of truth hai.
"""

import os
import json
import re
from langfuse import observe
from langfuse.openai import AsyncOpenAI

from .schemas import StructuredChatOutput

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# -----------------------------------------------------------
# Local vLLM config -- seedha localhost:8000 pe already chal
# raha vLLM server hit karega.
# -----------------------------------------------------------
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Langfuse apna config in env vars se khud utha leta hai:
#   LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

llm_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)


def get_health_info() -> dict:
    return {"status": "ok", "base_url": BASE_URL, "model": MODEL_NAME}


@observe()
async def run_chat(messages: list[dict], temperature: float, max_tokens: int) -> str:
    """Non-streaming chat completion (plain text, no structured output). messages = list of {"role", "content"} dicts."""
    completion = await llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


# @observe()     
# async def run_chat_structured(messages: list[dict], temperature: float, max_tokens: int) -> StructuredChatOutput:
#     """
#     Non-streaming chat completion JISME guided decoding lagi hai -- vLLM
#     hamesha valid {"answer": ..., "extracted_facts": {...}} JSON return karega.

#     SIRF EK INSTRUCTION SOURCE: pehle system-message + user-suffix-reminder
#     + schema teeno se overlapping instruction ja rahi thi -- chhote model
#     (gemma4) ke liye ye confusing tha aur safe-fallback me jaake sirf input
#     echo kar raha tha. Ab sirf response_format (JSON schema) hi instruction
#     ka source hai -- guided decoding khud token-level pe enforce karta hai,
#     alag se prompt-instruction ki zarurat hi nahi.

#     Returns: StructuredChatOutput (parsed Pydantic object, .answer aur
#              .extracted_facts attributes se access karo)
#     """

#     print(json.dumps({"model": MODEL_NAME,
#             "messages": messages,
#               "temperature": temperature, 
#               "max_tokens": max_tokens, 
#               "repetition_penalty": 1.3,
#                 "response_format": 
#                 {"type": "json_schema", 
#                  "json_schema": 
#                  {"name": "structured_chat_output", 
#                   "schema": StructuredChatOutput.model_json_schema()}}},
#                     indent=2, ensure_ascii=False))

#     completion = await llm_client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=messages,
#         temperature=temperature,
#         max_tokens=max_tokens,

#         extra_body={
#     "repetition_penalty": 1.05,  # 1.3 se 1.05 — grammar ke saath itna zyada penalty na do
#     "structured_outputs": {
#         "json": StructuredChatOutput.model_json_schema(),
#         "whitespace_pattern": " ",  # empty string ki jagah single space — official docs/bug-repro isi ka use karte hain
#                 },
#             },
        
#     )

    # print(completion)
    # raw_json = completion.choices[0].message.content
    # try:
    #     parsed = json.loads(raw_json)
    # except json.JSONDecodeError as e:
    #     # Model ka output beech me hi kat gaya (max_tokens khatam ho gaya
    #     # JSON complete hone se pehle) -- ye truncation hai, koi aur bug
    #     # nahi. Fix: is call ke max_tokens ko badhao (caller/schemas.py se),
    #     # ya answer ki length ko chhota rakhne ka instruction do.
    #     raise ValueError(
    #         f"Model returned incomplete/invalid JSON (likely truncated by "
    #         f"max_tokens={max_tokens}). Raw output length: {len(raw_json)} chars. "
    #         f"Original error: {e}"
    #     )
    # return StructuredChatOutput(**parsed)




import re

LANGUAGE_PATTERN = r"LANGUAGE: (english|hindi|hinglish)\nANSWER: [^\n]+\nENTITIES: [^\n]*"

@observe()
async def run_chat_structured(messages: list[dict], temperature: float, max_tokens: int) -> StructuredChatOutput:
    completion = await llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={
            "repetition_penalty": 1.05,
            "structured_outputs": {
                "regex": LANGUAGE_PATTERN,
            },
            "stop": ["\n\n"],  # extra safety, do newline ke baad ruk jao
        },
    )
    raw_text = completion.choices[0].message.content
    print(raw_text)

    match = re.match(
        r"LANGUAGE: (\w+)\nANSWER: (.+)\nENTITIES: (.*)",
        raw_text.strip(),
        re.DOTALL,
    )
    if not match:
        raise ValueError(f"Model output didn't match expected format: {raw_text}")

    language, answer, entities_raw = match.groups()
    entities = [e.strip() for e in entities_raw.split(",") if e.strip()]

    return StructuredChatOutput(
        detected_language=language,
        answer=answer.strip(),
        extracted_facts=ExtractedFacts(entities=entities),
    )


@observe()
async def run_chat_stream(messages: list[dict], temperature: float, max_tokens: int):
    """
    Streaming chat completion generator. Yields SSE-formatted string chunks.

    NOTE: Guided decoding (structured JSON output) streaming ke saath is
    file me use nahi kiya -- token-by-token JSON chunks ko live parse karna
    complex hai aur user ko half-JSON dikhta agar bina buffering ke stream
    kiya jaaye. Structured output sirf run_chat_structured() (non-streaming)
    me use hota hai. Agar streaming + structured dono chahiye ho future me,
    poora JSON buffer karke ek hi baar client ko bhejna padega (ab streaming
    ka fayda hi khatam ho jaata hai us case me).
    """
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