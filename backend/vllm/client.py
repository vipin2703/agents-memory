"""
vllm_service/client.py -- Local vLLM interaction.

Single-call structured JSON:
  - answer (complete, streamed live when using stream path)
  - extracted_facts (from latest user message only; verified in code)

No answer.maxLength hard cap (that was cutting replies).
Facts stay bounded with maxItems so JSON can finish.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import AsyncIterator
from typing import Any

from langfuse import observe
from langfuse.openai import AsyncOpenAI

from .schemas import ExtractedFacts, StructuredChatOutput

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

llm_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# Floor so answer + short facts both fit (client max_tokens se max liya jata hai)
STRUCTURED_MIN_MAX_TOKENS = 2048
FACT_ARRAY_MAX_ITEMS = 8

# answer pe maxLength NAHI — complete answer cut na ho.
# facts maxItems se JSON finish hota hai; pad risk prompt + disable_any_whitespace se.
_RELATION_ITEM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "predicate": {"type": "string"},
        "object": {"type": "string"},
    },
    "required": ["subject", "predicate", "object"],
    "additionalProperties": False,
}

GUIDED_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "extracted_facts": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": FACT_ARRAY_MAX_ITEMS,
                },
                "facts_about_user": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": FACT_ARRAY_MAX_ITEMS,
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": FACT_ARRAY_MAX_ITEMS,
                },
                "relations": {
                    "type": "array",
                    "items": _RELATION_ITEM_SCHEMA,
                    "maxItems": FACT_ARRAY_MAX_ITEMS,
                },
            },
            "required": [
                "entities",
                "facts_about_user",
                "constraints",
                "relations",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["answer", "extracted_facts"],
    "additionalProperties": False,
}

CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Reply in the same language the user used (Hindi/Hinglish/English). "
    "Copy names and company names exactly as the user wrote them — do not misspell. "
    "When MEMORY is provided, treat it as true for this conversation and use it. "
    "Be clear and complete — do not cut the answer short."
)

STRUCTURED_SYSTEM_PROMPT = (
    CHAT_SYSTEM_PROMPT
    + " "
    "You MUST reply with ONE JSON object only (no markdown, no text outside JSON). "
    "Keys in order: "
    '(1) "answer" — your FULL complete reply to the user; finish the whole thought; '
    "do NOT shorten the answer to make room for facts; "
    "do NOT pad with spaces/newlines after the last sentence; close the string immediately. "
    '(2) "extracted_facts" — object with: '
    "entities, facts_about_user, constraints (string arrays), "
    "and relations (array of {subject, predicate, object} triples). "
    f"Max {FACT_ARRAY_MAX_ITEMS} items per array. "
    "IMPORTANT: BOTH answer and extracted_facts must be present and finished. "
    "extracted_facts ONLY from the latest user message (not from your answer, not invented). "
    "relations: how entities connect, e.g. subject=Rahul predicate=LIVES_IN object=Pune; "
    "predicate UPPER_SNAKE_CASE; only if clearly stated; else relations=[]. "
    "If nothing to extract, use empty arrays []. "
    "After answer is complete, fill facts+relations briefly, then close all braces."
)


def get_health_info() -> dict:
    return {"status": "ok", "base_url": BASE_URL, "model": MODEL_NAME}


def _format_memory_block(memory: dict | ExtractedFacts | None) -> str:
    if not memory:
        return ""
    if isinstance(memory, ExtractedFacts):
        data = memory.model_dump()
    else:
        data = memory
    lines: list[str] = []
    for key in ("entities", "facts_about_user", "constraints"):
        items = data.get(key) or []
        if items:
            lines.append(f"- {key}: {', '.join(str(x) for x in items)}")
    rels = data.get("relations") or []
    if rels:
        bits = []
        for r in rels:
            if isinstance(r, dict):
                bits.append(
                    f"{r.get('subject', '')} -[{r.get('predicate', '')}]-> {r.get('object', '')}"
                )
            else:
                bits.append(str(r))
        lines.append("- relations: " + "; ".join(bits))
    if not lines:
        return ""
    return (
        "STRUCTURED FACTS (use them; do not forget or contradict):\n"
        + "\n".join(lines)
    )


def _with_system_and_memory(
    messages: list[dict],
    memory: dict | ExtractedFacts | None = None,
    *,
    system_prompt: str = CHAT_SYSTEM_PROMPT,
    extra_memory_block: str | None = None,
) -> list[dict]:
    """
    final_messages banata hai (vLLM ko yahi jati hai).

    File: backend/vllm/client.py
    Variable: final_messages
    """
    parts = [system_prompt]
    rich = (extra_memory_block or "").strip()
    if rich:
        parts.append(rich)
    fact_block = _format_memory_block(memory)
    if fact_block:
        parts.append(fact_block)

    system_content = "\n\n".join(parts)
    rest = [m for m in messages if m.get("role") != "system"]
    final_messages = [{"role": "system", "content": system_content}, *rest]
    return final_messages


def _debug_print_final_messages(final_messages: list[dict], where: str) -> None:
    """Simple debug — vLLM call se pehle. Variable name = final_messages."""
    print("\n========== FINAL vLLM INPUT ==========", flush=True)
    print("file: backend/vllm/client.py", flush=True)
    print(f"where: {where}", flush=True)
    print("variable: final_messages", flush=True)
    print(f"count: {len(final_messages)}", flush=True)
    print(json.dumps(final_messages, indent=2, ensure_ascii=False), flush=True)
    print("========== END FINAL vLLM INPUT ==========\n", flush=True)
    sys.stdout.flush()


def _debug_print_final_output(payload: dict | str, where: str, *, variable: str = "result") -> None:
    """Simple debug — vLLM se aane ke baad. Variable name default = result."""
    print("\n========== FINAL vLLM OUTPUT ==========", flush=True)
    print("file: backend/vllm/client.py", flush=True)
    print(f"where: {where}", flush=True)
    print(f"variable: {variable}", flush=True)
    if isinstance(payload, str):
        print(f"len: {len(payload)}", flush=True)
        print(payload, flush=True)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), flush=True)
    print("========== END FINAL vLLM OUTPUT ==========\n", flush=True)
    sys.stdout.flush()


def _dedupe_list(items: list, *, max_items: int = FACT_ARRAY_MAX_ITEMS) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items or []:
        s = str(raw).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def _item_grounded_in_user(item: str, user_lower: str) -> bool:
    """Cheap anti-hallucination: fact must appear in / overlap user text."""
    s = (item or "").strip()
    if not s:
        return False
    sl = s.lower()
    if sl in user_lower:
        return True
    words = [w for w in re.findall(r"\w+", sl, flags=re.UNICODE) if len(w) > 2]
    if not words:
        return sl in user_lower
    hits = sum(1 for w in words if w in user_lower)
    return hits >= max(1, (len(words) + 1) // 2)


def _normalize_predicate(pred: str) -> str:
    p = re.sub(r"[^A-Za-z0-9]+", "_", (pred or "").strip()).strip("_")
    return (p or "RELATED_TO").upper()[:64]


def _normalize_relations(raw_rels: list, *, max_items: int = FACT_ARRAY_MAX_ITEMS) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for r in raw_rels or []:
        if isinstance(r, dict):
            sub = str(r.get("subject") or "").strip()
            pred = _normalize_predicate(str(r.get("predicate") or ""))
            obj = str(r.get("object") or "").strip()
        else:
            continue
        if not sub or not obj:
            continue
        key = f"{sub.lower()}|{pred}|{obj.lower()}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"subject": sub, "predicate": pred, "object": obj})
        if len(out) >= max_items:
            break
    return out


def filter_facts_against_user_text(facts: dict, user_text: str) -> dict:
    """Drop ungrounded extracts (no extra LLM cost)."""
    user_lower = (user_text or "").lower()
    empty = {
        "entities": [],
        "facts_about_user": [],
        "constraints": [],
        "relations": [],
    }
    if not user_lower.strip():
        return empty
    out: dict = {}
    for key in ("entities", "facts_about_user", "constraints"):
        kept: list[str] = []
        for item in facts.get(key) or []:
            if _item_grounded_in_user(str(item), user_lower):
                kept.append(str(item).strip())
        out[key] = _dedupe_list(kept)
    rels_kept: list[dict] = []
    for r in facts.get("relations") or []:
        if not isinstance(r, dict):
            continue
        sub = str(r.get("subject") or "").strip()
        obj = str(r.get("object") or "").strip()
        pred = _normalize_predicate(str(r.get("predicate") or ""))
        # subject + object dono user text me grounded
        if _item_grounded_in_user(sub, user_lower) and _item_grounded_in_user(
            obj, user_lower
        ):
            rels_kept.append({"subject": sub, "predicate": pred, "object": obj})
    out["relations"] = _normalize_relations(rels_kept)
    return out


def _normalize_structured_dict(
    data: dict, *, user_text: str | None = None
) -> dict:
    answer = data.get("answer")
    if answer is None:
        answer = ""
    elif not isinstance(answer, str):
        answer = str(answer)
    answer = answer.strip()

    facts_raw = data.get("extracted_facts") or {}
    if not isinstance(facts_raw, dict):
        facts_raw = {}

    facts = {
        "entities": _dedupe_list(facts_raw.get("entities") or []),
        "facts_about_user": _dedupe_list(facts_raw.get("facts_about_user") or []),
        "constraints": _dedupe_list(facts_raw.get("constraints") or []),
        "relations": _normalize_relations(facts_raw.get("relations") or []),
    }
    if user_text is not None:
        facts = filter_facts_against_user_text(facts, user_text)

    return {"answer": answer, "extracted_facts": facts}


def _extract_answer_from_broken_json(text: str) -> str:
    key = '"answer"'
    idx = text.find(key)
    if idx < 0:
        return ""
    after = text[idx + len(key) :]
    colon = after.find(":")
    if colon < 0:
        return ""
    after = after[colon + 1 :].lstrip()
    if not after.startswith('"'):
        return ""
    i = 1
    chars: list[str] = []
    while i < len(after):
        c = after[i]
        if c == "\\" and i + 1 < len(after):
            chars.append(after[i : i + 2])
            i += 2
            continue
        if c == '"':
            break
        chars.append(c)
        i += 1
    try:
        return json.loads('"' + "".join(chars).replace("\n", "\\n") + '"').strip()
    except json.JSONDecodeError:
        return "".join(chars).strip()


def partial_answer_from_raw_json(raw: str) -> str:
    """Live stream: partial/complete answer string from incomplete JSON buffer."""
    m = re.search(r'"answer"\s*:\s*"', raw)
    if not m:
        return ""
    i = m.end()
    out: list[str] = []
    escape = False
    while i < len(raw):
        c = raw[i]
        if escape:
            if c == "n":
                out.append("\n")
            elif c == "t":
                out.append("\t")
            elif c == "r":
                out.append("\r")
            elif c == '"':
                out.append('"')
            elif c == "\\":
                out.append("\\")
            elif c == "/":
                out.append("/")
            elif c == "u" and i + 4 < len(raw):
                hexpart = raw[i + 1 : i + 5]
                try:
                    out.append(chr(int(hexpart, 16)))
                    i += 4
                except ValueError:
                    out.append(c)
            else:
                out.append(c)
            escape = False
            i += 1
            continue
        if c == "\\":
            escape = True
            i += 1
            continue
        if c == '"':
            break
        out.append(c)
        i += 1
    return "".join(out)


def _try_repair_truncated_json(text: str) -> dict | None:
    answer = _extract_answer_from_broken_json(text)

    candidate = text.rstrip()
    in_string = False
    escape = False
    for ch in candidate:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        candidate += '"'

    candidate = candidate.rstrip()
    if candidate.endswith(","):
        candidate = candidate[:-1]

    opens = candidate.count("{") - candidate.count("}")
    opens_arr = candidate.count("[") - candidate.count("]")
    if opens >= 0 and opens_arr >= 0:
        candidate += "]" * opens_arr + "}" * opens
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                if not (data.get("answer") or "").strip() and answer:
                    data["answer"] = answer
                if "extracted_facts" not in data or not isinstance(
                    data.get("extracted_facts"), dict
                ):
                    data["extracted_facts"] = {
                        "entities": [],
                        "facts_about_user": [],
                        "constraints": [],
                        "relations": [],
                    }
                else:
                    data["extracted_facts"].setdefault("relations", [])
                return data
        except json.JSONDecodeError:
            pass

    if answer:
        return {
            "answer": answer,
            "extracted_facts": {
                "entities": [],
                "facts_about_user": [],
                "constraints": [],
                "relations": [],
            },
        }
    return None


def _parse_structured_output(
    raw: str,
    finish_reason: str | None,
    max_tokens: int,
    *,
    user_text: str | None = None,
) -> StructuredChatOutput:
    text = (raw or "").strip()
    if not text:
        raise ValueError(
            f"Empty structured output (finish_reason={finish_reason!r}, max_tokens={max_tokens})"
        )

    if "```" in text:
        start = text.find("```")
        rest = text[start + 3 :]
        if rest.lstrip().lower().startswith("json"):
            rest = rest.lstrip()[4:]
        end = rest.find("```")
        text = (rest[:end] if end >= 0 else rest).strip()

    data: dict | None = None
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Structured output root must be object, got {type(parsed).__name__}"
            )
        data = parsed
    except json.JSONDecodeError as e:
        repaired = _try_repair_truncated_json(text)
        if repaired is not None and (repaired.get("answer") or "").strip():
            logger.warning(
                "Structured JSON incomplete (finish_reason=%r); salvaged. err=%s",
                finish_reason,
                e,
            )
            data = repaired
        else:
            snippet = text[:240].replace("\n", "\\n")
            hint = ""
            if finish_reason == "length":
                hint = " Generation hit max_tokens mid-JSON."
            raise ValueError(
                f"Invalid structured JSON (finish_reason={finish_reason!r}, "
                f"max_tokens={max_tokens}, len={len(text)}).{hint} "
                f"Snippet: {snippet!r}. Error: {e}"
            ) from e

    assert data is not None
    return StructuredChatOutput(
        **_normalize_structured_dict(data, user_text=user_text)
    )


def _latest_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _structured_request_kwargs(
    final_messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    stream: bool = False,
) -> dict[str, Any]:
    structured_temperature = min(temperature, 0.5)
    structured_max_tokens = max(max_tokens, STRUCTURED_MIN_MAX_TOKENS)
    kwargs: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": final_messages,
        "temperature": structured_temperature,
        "max_tokens": structured_max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_chat_output",
                "schema": GUIDED_JSON_SCHEMA,
                "strict": True,
            },
        },
        "extra_body": {
            "structured_outputs": {
                "json": GUIDED_JSON_SCHEMA,
                "disable_any_whitespace": True,
                "disable_additional_properties": True,
                "whitespace_pattern": "",
            },
        },
    }
    if stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
    return kwargs


@observe()
async def run_chat(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    memory: dict | ExtractedFacts | None = None,
    extra_memory_block: str | None = None,
) -> str:
    final_messages = _with_system_and_memory(
        messages,
        memory=memory,
        system_prompt=CHAT_SYSTEM_PROMPT,
        extra_memory_block=extra_memory_block,
    )
    _debug_print_final_messages(final_messages, "run_chat → create")

    completion = await llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=final_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = completion.choices[0].message.content
    text = (content or "").strip()
    _debug_print_final_output(
        {
            "content": text,
            "finish_reason": completion.choices[0].finish_reason,
        },
        "run_chat ← response",
        variable="content",
    )
    return text


@observe()
async def run_chat_structured(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    memory: dict | ExtractedFacts | None = None,
    extra_memory_block: str | None = None,
) -> StructuredChatOutput:
    """Single LLM call: complete answer + extracted_facts (user-grounded filter)."""
    final_messages = _with_system_and_memory(
        messages,
        memory=memory,
        system_prompt=STRUCTURED_SYSTEM_PROMPT,
        extra_memory_block=extra_memory_block,
    )
    _debug_print_final_messages(final_messages, "run_chat_structured → create")
    kwargs = _structured_request_kwargs(
        final_messages, temperature=temperature, max_tokens=max_tokens, stream=False
    )
    structured_max_tokens = kwargs["max_tokens"]

    completion = await llm_client.chat.completions.create(**kwargs)
    choice = completion.choices[0]
    raw = choice.message.content or ""
    finish_reason = choice.finish_reason
    user_text = _latest_user_text(messages)

    _debug_print_final_output(
        {
            "raw_content": raw,
            "finish_reason": finish_reason,
        },
        "run_chat_structured ← raw model content",
        variable="raw",
    )

    result = _parse_structured_output(
        raw, finish_reason, structured_max_tokens, user_text=user_text
    )
    _debug_print_final_output(
        {
            "answer": result.answer,
            "extracted_facts": result.extracted_facts.model_dump(),
            "finish_reason": finish_reason,
        },
        "run_chat_structured ← parsed result",
        variable="result",
    )
    if finish_reason == "length":
        logger.warning(
            "Structured hit max_tokens=%s; answer_len=%s facts=%s",
            structured_max_tokens,
            len(result.answer or ""),
            result.extracted_facts.model_dump(),
        )
    return result


@observe()
async def run_chat_structured_stream(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    memory: dict | ExtractedFacts | None = None,
    extra_memory_block: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Single structured call with LIVE answer tokens.

    Yields:
      {"type": "answer_delta", "text": "..."}
      {"type": "final", "answer": "...", "extracted_facts": {...}, "finish_reason": "..."}
      {"type": "error", "message": "..."}
    """
    final_messages = _with_system_and_memory(
        messages,
        memory=memory,
        system_prompt=STRUCTURED_SYSTEM_PROMPT,
        extra_memory_block=extra_memory_block,
    )
    
    _debug_print_final_messages(final_messages, "run_chat_structured_stream → create")
    kwargs = _structured_request_kwargs(
        final_messages, temperature=temperature, max_tokens=max_tokens, stream=True
    )
    structured_max_tokens = kwargs["max_tokens"]
    user_text = _latest_user_text(messages)

    raw_parts: list[str] = []
    emitted_answer_len = 0
    finish_reason: str | None = None

    try:
        stream = await llm_client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta.content if choice.delta else None
            if not delta:
                continue
            raw_parts.append(delta)
            raw_so_far = "".join(raw_parts)
            partial = partial_answer_from_raw_json(raw_so_far)
            if len(partial) > emitted_answer_len:
                new_text = partial[emitted_answer_len:]
                emitted_answer_len = len(partial)
                yield {"type": "answer_delta", "text": new_text}

        raw = "".join(raw_parts)
        _debug_print_final_output(
            {
                "raw_content": raw,
                "finish_reason": finish_reason,
            },
            "run_chat_structured_stream ← raw model content",
            variable="raw",
        )
        result = _parse_structured_output(
            raw, finish_reason, structured_max_tokens, user_text=user_text
        )
        _debug_print_final_output(
            {
                "answer": result.answer,
                "extracted_facts": result.extracted_facts.model_dump(),
                "finish_reason": finish_reason,
            },
            "run_chat_structured_stream ← parsed result",
            variable="result",
        )
        # agar stream me answer incomplete tha, final se catch-up
        if len(result.answer) > emitted_answer_len:
            yield {
                "type": "answer_delta",
                "text": result.answer[emitted_answer_len:],
            }
        yield {
            "type": "final",
            "answer": result.answer,
            "extracted_facts": result.extracted_facts.model_dump(),
            "finish_reason": finish_reason,
        }
    except Exception as e:
        logger.exception("structured stream failed")
        yield {"type": "error", "message": str(e)}


@observe()
async def run_chat_stream(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    memory: dict | ExtractedFacts | None = None,
    extra_memory_block: str | None = None,
):
    """Streaming plain text (no guided JSON)."""
    final_messages = _with_system_and_memory(
        messages,
        memory=memory,
        system_prompt=CHAT_SYSTEM_PROMPT,
        extra_memory_block=extra_memory_block,
    )
    _debug_print_final_messages(final_messages, "run_chat_stream → create")
    try:
        stream = await llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=final_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        parts: list[str] = []
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    yield f"data: {delta}\n\n"
        full = "".join(parts)
        _debug_print_final_output(
            {"content": full},
            "run_chat_stream ← full text",
            variable="full",
        )
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: [ERROR] {e}\n\n"
