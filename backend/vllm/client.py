# """
# vllm client — only structured STREAM path.

# Entry: run_chat_structured_stream
#   tool_calls (optional) → answer + extracted_facts (streamed)
# """

# from __future__ import annotations

# import json
# import logging
# import os
# import re
# import sys
# from collections.abc import AsyncIterator
# from typing import Any

# from langfuse import observe
# from langfuse.openai import AsyncOpenAI

# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     pass

# logger = logging.getLogger(__name__)

# BASE_URL = os.getenv("BASE_URL")
# API_KEY = os.getenv("API_KEY")
# MODEL_NAME = os.getenv("MODEL_NAME")

# llm_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

# # answer FIRST, then extracted_facts (user requirement).
# # No hard "short answer" cap — only strip trailing pad after the real reply.
# STRUCTURED_MIN_MAX_TOKENS = 1024
# STRUCTURED_MAX_MAX_TOKENS = 2048
# FACT_ARRAY_MAX_ITEMS = 8
# MAX_TOOL_ROUNDS = 2

# # Small models sometimes emit an endless run of whitespace between JSON tokens
# # (typically right after the answer string closes), burning tokens until
# # finish_reason='length' and truncating the JSON — the answer then arrives
# # incomplete. This many consecutive whitespace chars while streaming is never
# # legitimate formatting; abort the round early and let salvage recover.
# PAD_RUN_ABORT = 48

# _RELATION_ITEM_SCHEMA: dict = {
#     "type": "object",
#     "properties": {
#         "subject": {"type": "string"},
#         "predicate": {"type": "string"},
#         "object": {"type": "string"},
#     },
#     "required": ["subject", "predicate", "object"],
#     "additionalProperties": False,
# }

# _TOOL_CALL_ITEM_SCHEMA: dict = {
#     "type": "object",
#     "properties": {
#         "name": {"type": "string"},
#         "arguments": {"type": "string"},
#     },
#     "required": ["name", "arguments"],
#     "additionalProperties": False,
# }

# _EXTRACTED_FACTS_SCHEMA: dict = {
#     "type": "object",
#     "properties": {
#         "entities": {
#             "type": "array",
#             "items": {"type": "string"},
#             "maxItems": FACT_ARRAY_MAX_ITEMS,
#         },
#         "facts_about_user": {
#             "type": "array",
#             "items": {"type": "string"},
#             "maxItems": FACT_ARRAY_MAX_ITEMS,
#         },
#         "constraints": {
#             "type": "array",
#             "items": {"type": "string"},
#             "maxItems": FACT_ARRAY_MAX_ITEMS,
#         },
#         "relations": {
#             "type": "array",
#             "items": _RELATION_ITEM_SCHEMA,
#             "maxItems": FACT_ARRAY_MAX_ITEMS,
#         },
#     },
#     "required": [
#         "entities",
#         "facts_about_user",
#         "constraints",
#         "relations",
#     ],
#     "additionalProperties": False,
# }

# # Full answer allowed (no maxLength / no single-line pattern).
# _ANSWER_SCHEMA: dict = {
#     "type": "string",
# }

# # Order: answer first → then extract (facts)
# GUIDED_JSON_SCHEMA: dict = {
#     "type": "object",
#     "properties": {
#         "answer": _ANSWER_SCHEMA,
#         "extracted_facts": _EXTRACTED_FACTS_SCHEMA,
#     },
#     "required": ["answer", "extracted_facts"],
#     "additionalProperties": False,
# }

# GUIDED_JSON_SCHEMA_WITH_TOOLS: dict = {
#     "type": "object",
#     "properties": {
#         "tool_calls": {
#             "type": "array",
#             "items": _TOOL_CALL_ITEM_SCHEMA,
#             "maxItems": 2,
#         },
#         "answer": _ANSWER_SCHEMA,
#         "extracted_facts": _EXTRACTED_FACTS_SCHEMA,
#     },
#     "required": ["tool_calls", "answer", "extracted_facts"],
#     "additionalProperties": False,
# }

# _FACTS_EMPTY_RULE = (
#     "extracted_facts ONLY from the latest user message. "
#     "If nothing to extract, leave entities/facts/constraints/relations as empty [] — "
#     "do not invent. Empty is correct for chit-chat. "
#     'relations: link two named things as {"subject":"..","predicate":"UPPER_SNAKE","object":".."} '
#     "using the exact words in this message (a dedicated pass also backfills these)."
# )

# # Focused, single-purpose relation extractor. Small models fill the relations
# # array far more reliably when it is the ONLY task, at temperature 0, than when
# # it is one field buried in the big answer+facts prompt (gemma returns [] there).
# _RELATIONS_ONLY_SCHEMA: dict = {
#     "type": "object",
#     "properties": {
#         "relations": {
#             "type": "array",
#             "items": _RELATION_ITEM_SCHEMA,
#             "maxItems": FACT_ARRAY_MAX_ITEMS,
#         },
#     },
#     "required": ["relations"],
#     "additionalProperties": False,
# }

# _RELATION_EXTRACT_PROMPT = (
#     "Extract RELATIONSHIP TRIPLES from the user message. Output ONE JSON object only, "
#     'exactly {"relations":[{"subject":"..","predicate":"UPPER_SNAKE","object":".."}]}. '
#     "A triple links two named things the user stated (person-employer, person-city, "
#     "person-person, thing-thing). Examples:\n"
#     '"My name is Kapil, I work at Google and live in Bangalore" -> '
#     '{"relations":[{"subject":"Kapil","predicate":"WORKS_AT","object":"Google"},'
#     '{"subject":"Kapil","predicate":"LIVES_IN","object":"Bangalore"}]}\n'
#     '"Aman is the CEO of Acme" -> '
#     '{"relations":[{"subject":"Aman","predicate":"CEO_OF","object":"Acme"}]}\n'
#     '"hello there" -> {"relations":[]}\n'
#     "Use the exact names from the message; subject AND object must both be words that "
#     'appear in the message. If there is no clear link, output {"relations":[]}.'
# )

# STRUCTURED_SYSTEM_PROMPT = (
#     "Reply in the user's language. ONE JSON object only. ASCII quotes only. "
#     'Shape order: {"answer":"<full reply>","extracted_facts":{"entities":[],'
#     '"facts_about_user":[],"constraints":[],"relations":[]}}. '
#     "1) Write the FULL answer first (complete thought — do not cut short). "
#     "2) Close the answer string as soon as the reply is done — "
#     "do NOT pad with spaces or newlines after the last sentence. "
#     "3) Then fill extracted_facts. "
#     f"{_FACTS_EMPTY_RULE}"
# )

# STRUCTURED_SYSTEM_PROMPT_WITH_TOOLS = (
#     "Reply in the user's language. ONE JSON object only. ASCII quotes only.\n"
#     'Shape order: {"tool_calls":[],"answer":"<full reply>","extracted_facts":'
#     '{"entities":[],"facts_about_user":[],"constraints":[],"relations":[]}}\n'
#     "1) tool_calls if you need a tool (else []). "
#     "2) answer = FULL reply to the user (complete, not artificially short). "
#     "If calling a tool first, answer may be \"\". "
#     "3) Close answer quote immediately when the reply is finished — "
#     "NO spaces/newlines padding after the last word. "
#     "4) Then extracted_facts.\n"
#     "Tools: search_conversation (past chat / ES), search_context (graph facts).\n"
#     "DECIDE per message:\n"
#     "- Greeting / chit-chat / general knowledge (hello, hi, how are you, thanks, "
#     "jokes, facts) -> tool_calls=[] and write a normal friendly reply. Do NOT "
#     "search, do NOT mention names. A plain 'hello' just gets a plain greeting.\n"
#     "- If a 'KNOWN ABOUT THIS USER' block is present below, it is THIS user's own "
#     "saved data — retrieving it is expected, never a privacy issue. Answer the "
#     "question directly and confidently from that block (e.g. the person entity is "
#     "their name). Do NOT refuse and do NOT say you lack access.\n"
#     "- If the user asks about themselves / past info and there is NO such block, "
#     "you may call search_context or search_conversation; if still nothing, say you "
#     "don't have it stored yet. Never invent a name.\n"
#     f"{_FACTS_EMPTY_RULE}"
# )


# def get_health_info() -> dict:
#     return {"status": "ok", "base_url": BASE_URL, "model": MODEL_NAME}


# def _sanitize_chat_roles(messages: list[dict]) -> list[dict]:
#     """
#     vLLM/Gemma chat templates require:
#       system? → user → assistant → user → assistant → ...
#     Never start with assistant; never two same roles in a row.
#     """
#     rest = [
#         {"role": m.get("role"), "content": m.get("content") or ""}
#         for m in messages
#         if m.get("role") in ("user", "assistant")
#     ]
#     while rest and rest[0]["role"] != "user":
#         rest.pop(0)
#     cleaned: list[dict] = []
#     for m in rest:
#         role = m["role"]
#         if cleaned and cleaned[-1]["role"] == role:
#             # merge consecutive same-role (keep latest content)
#             cleaned[-1] = m
#             continue
#         cleaned.append(m)
#     if cleaned and cleaned[-1]["role"] != "user":
#         # request should end on user; drop trailing assistant orphan
#         # (only if last is assistant without following user — keep if odd history)
#         pass
#     # Ensure we end with user if possible (generation turn)
#     if cleaned and cleaned[-1]["role"] == "assistant":
#         cleaned.pop()
#     return cleaned


# def _with_system_and_memory(
#     messages: list[dict],
#     *,
#     system_prompt: str = STRUCTURED_SYSTEM_PROMPT,
# ) -> list[dict]:
#     """
#     final_messages banata hai (LLM ko yahi jati hai). system_prompt me tools /
#     recalled memory / tool-results already merged hote hain (_build_system_prompt).

#     File: backend/vllm/client.py
#     Variable: final_messages
#     """
#     rest = _sanitize_chat_roles(messages)
#     final_messages = [{"role": "system", "content": system_prompt}, *rest]
#     return final_messages


# def _debug_print_final_messages(final_messages: list[dict], where: str) -> None:
#     """Simple debug — vLLM call se pehle. Variable name = final_messages."""
#     print("\n========== FINAL vLLM INPUT ==========", flush=True)
#     print("file: backend/vllm/client.py", flush=True)
#     print(f"where: {where}", flush=True)
#     print("variable: final_messages", flush=True)
#     print(f"count: {len(final_messages)}", flush=True)
#     print(json.dumps(final_messages, indent=2, ensure_ascii=False), flush=True)
#     print("========== END FINAL vLLM INPUT ==========\n", flush=True)
#     sys.stdout.flush()


# def _debug_print_final_output(payload: dict | str, where: str, *, variable: str = "result") -> None:
#     """Simple debug — vLLM se aane ke baad. Variable name default = result."""
#     print("\n========== FINAL vLLM OUTPUT ==========", flush=True)
#     print("file: backend/vllm/client.py", flush=True)
#     print(f"where: {where}", flush=True)
#     print(f"variable: {variable}", flush=True)
#     if isinstance(payload, str):
#         print(f"len: {len(payload)}", flush=True)
#         print(payload, flush=True)
#     else:
#         print(json.dumps(payload, indent=2, ensure_ascii=False, default=str), flush=True)
#     print("========== END FINAL vLLM OUTPUT ==========\n", flush=True)
#     sys.stdout.flush()


# def _dedupe_list(items: list, *, max_items: int = FACT_ARRAY_MAX_ITEMS) -> list[str]:
#     out: list[str] = []
#     seen: set[str] = set()
#     for raw in items or []:
#         s = str(raw).strip()
#         if not s:
#             continue
#         key = s.lower()
#         if key in seen:
#             continue
#         seen.add(key)
#         out.append(s)
#         if len(out) >= max_items:
#             break
#     return out


# def _item_grounded_in_user(item: str, user_lower: str) -> bool:
#     """Cheap anti-hallucination: fact must appear in / overlap user text."""
#     s = (item or "").strip()
#     if not s:
#         return False
#     sl = s.lower()
#     if sl in user_lower:
#         return True
#     words = [w for w in re.findall(r"\w+", sl, flags=re.UNICODE) if len(w) > 2]
#     if not words:
#         return sl in user_lower
#     hits = sum(1 for w in words if w in user_lower)
#     return hits >= max(1, (len(words) + 1) // 2)


# def _normalize_predicate(pred: str) -> str:
#     p = re.sub(r"[^A-Za-z0-9]+", "_", (pred or "").strip()).strip("_")
#     return (p or "RELATED_TO").upper()[:64]


# def _normalize_relations(raw_rels: list, *, max_items: int = FACT_ARRAY_MAX_ITEMS) -> list[dict]:
#     out: list[dict] = []
#     seen: set[str] = set()
#     for r in raw_rels or []:
#         if isinstance(r, dict):
#             sub = str(r.get("subject") or "").strip()
#             pred = _normalize_predicate(str(r.get("predicate") or ""))
#             obj = str(r.get("object") or "").strip()
#         else:
#             continue
#         if not sub or not obj:
#             continue
#         key = f"{sub.lower()}|{pred}|{obj.lower()}"
#         if key in seen:
#             continue
#         seen.add(key)
#         out.append({"subject": sub, "predicate": pred, "object": obj})
#         if len(out) >= max_items:
#             break
#     return out


# def filter_facts_against_user_text(facts: dict, user_text: str) -> dict:
#     """Drop ungrounded extracts (no extra LLM cost)."""
#     user_lower = (user_text or "").lower()
#     empty = {
#         "entities": [],
#         "facts_about_user": [],
#         "constraints": [],
#         "relations": [],
#     }
#     if not user_lower.strip():
#         return empty
#     out: dict = {}
#     for key in ("entities", "facts_about_user", "constraints"):
#         kept: list[str] = []
#         for item in facts.get(key) or []:
#             if _item_grounded_in_user(str(item), user_lower):
#                 kept.append(str(item).strip())
#         out[key] = _dedupe_list(kept)
#     rels_kept: list[dict] = []
#     for r in facts.get("relations") or []:
#         if not isinstance(r, dict):
#             continue
#         sub = str(r.get("subject") or "").strip()
#         obj = str(r.get("object") or "").strip()
#         pred = _normalize_predicate(str(r.get("predicate") or ""))
#         # subject + object dono user text me grounded
#         if _item_grounded_in_user(sub, user_lower) and _item_grounded_in_user(
#             obj, user_lower
#         ):
#             rels_kept.append({"subject": sub, "predicate": pred, "object": obj})
#     out["relations"] = _normalize_relations(rels_kept)
#     return out


# def _normalize_tool_calls(raw: list | None) -> list[dict[str, str]]:
#     out: list[dict[str, str]] = []
#     for item in raw or []:
#         if not isinstance(item, dict):
#             continue
#         name = str(item.get("name") or "").strip()
#         if not name:
#             continue
#         args = item.get("arguments", "")
#         if isinstance(args, dict):
#             args_s = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
#         else:
#             args_s = str(args or "")
#         out.append({"name": name, "arguments": args_s})
#         if len(out) >= 2:
#             break
#     return out


# def _clean_answer_text(answer: str) -> str:
#     """
#     Strip trailing pad bombs only — do NOT shorten real answers.
#     Collapses runaway whitespace after content; keeps normal paragraphs.
#     """
#     if not answer:
#         return ""
#     # Collapse only extreme pad (3+ blank lines / long space runs at end)
#     text = re.sub(r"\n{3,}", "\n\n", answer)
#     text = re.sub(r"[ \t]{3,}", "  ", text)
#     text = text.rstrip()  # trailing pad after last sentence
#     # If model left a wall of newlines mid-string after a sentence, cut trailing junk
#     text = re.sub(r"([.!?…\"'])\s{10,}$", r"\1", text)
#     return text.strip()


# def _normalize_structured_dict(
#     data: dict, *, user_text: str | None = None, allow_empty_answer: bool = False
# ) -> dict:
#     answer = data.get("answer")
#     if answer is None:
#         answer = ""
#     elif not isinstance(answer, str):
#         answer = str(answer)
#     answer = _clean_answer_text(answer)

#     facts_raw = data.get("extracted_facts") or {}
#     if not isinstance(facts_raw, dict):
#         facts_raw = {}

#     facts = {
#         "entities": _dedupe_list(facts_raw.get("entities") or []),
#         "facts_about_user": _dedupe_list(facts_raw.get("facts_about_user") or []),
#         "constraints": _dedupe_list(facts_raw.get("constraints") or []),
#         "relations": _normalize_relations(facts_raw.get("relations") or []),
#     }
#     if user_text is not None:
#         facts = filter_facts_against_user_text(facts, user_text)

#     tool_calls = _normalize_tool_calls(data.get("tool_calls"))
#     _ = allow_empty_answer

#     return {
#         "answer": answer,
#         "extracted_facts": facts,
#         "tool_calls": tool_calls,
#     }


# def _extract_answer_from_broken_json(text: str) -> str:
#     key = '"answer"'
#     idx = text.find(key)
#     if idx < 0:
#         return ""
#     after = text[idx + len(key) :]
#     colon = after.find(":")
#     if colon < 0:
#         return ""
#     after = after[colon + 1 :].lstrip()
#     if not after.startswith('"'):
#         return ""
#     i = 1
#     chars: list[str] = []
#     while i < len(after):
#         c = after[i]
#         if c == "\\" and i + 1 < len(after):
#             chars.append(after[i : i + 2])
#             i += 2
#             continue
#         if c == '"':
#             break
#         chars.append(c)
#         i += 1
#     try:
#         raw_ans = json.loads('"' + "".join(chars).replace("\n", "\\n") + '"')
#     except json.JSONDecodeError:
#         raw_ans = "".join(chars)
#     return _clean_answer_text(raw_ans)


# def partial_answer_from_raw_json(raw: str) -> str:
#     """Live stream: partial/complete answer string from incomplete JSON buffer."""
#     m = re.search(r'"answer"\s*:\s*"', raw)
#     if not m:
#         return ""
#     i = m.end()
#     out: list[str] = []
#     escape = False
#     while i < len(raw):
#         c = raw[i]
#         if escape:
#             if c == "n":
#                 out.append("\n")
#             elif c == "t":
#                 out.append("\t")
#             elif c == "r":
#                 out.append("\r")
#             elif c == '"':
#                 out.append('"')
#             elif c == "\\":
#                 out.append("\\")
#             elif c == "/":
#                 out.append("/")
#             elif c == "u" and i + 4 < len(raw):
#                 hexpart = raw[i + 1 : i + 5]
#                 try:
#                     out.append(chr(int(hexpart, 16)))
#                     i += 4
#                 except ValueError:
#                     out.append(c)
#             else:
#                 out.append(c)
#             escape = False
#             i += 1
#             continue
#         if c == "\\":
#             escape = True
#             i += 1
#             continue
#         if c == '"':
#             break
#         out.append(c)
#         i += 1
#     return "".join(out)


# def _try_repair_truncated_json(text: str) -> dict | None:
#     answer = _extract_answer_from_broken_json(text)

#     candidate = text.rstrip()
#     in_string = False
#     escape = False
#     for ch in candidate:
#         if escape:
#             escape = False
#             continue
#         if ch == "\\":
#             escape = True
#             continue
#         if ch == '"':
#             in_string = not in_string
#     if in_string:
#         candidate += '"'

#     candidate = candidate.rstrip()
#     if candidate.endswith(","):
#         candidate = candidate[:-1]

#     opens = candidate.count("{") - candidate.count("}")
#     opens_arr = candidate.count("[") - candidate.count("]")
#     if opens >= 0 and opens_arr >= 0:
#         candidate += "]" * opens_arr + "}" * opens
#         try:
#             data = json.loads(candidate)
#             if isinstance(data, dict):
#                 if not (data.get("answer") or "").strip() and answer:
#                     data["answer"] = answer
#                 if "extracted_facts" not in data or not isinstance(
#                     data.get("extracted_facts"), dict
#                 ):
#                     data["extracted_facts"] = {
#                         "entities": [],
#                         "facts_about_user": [],
#                         "constraints": [],
#                         "relations": [],
#                     }
#                 else:
#                     data["extracted_facts"].setdefault("relations", [])
#                 return data
#         except json.JSONDecodeError:
#             pass

#     if answer:
#         return {
#             "answer": answer,
#             "extracted_facts": {
#                 "entities": [],
#                 "facts_about_user": [],
#                 "constraints": [],
#                 "relations": [],
#             },
#         }
#     return None


# def _sanitize_model_json_text(text: str) -> str:
#     """Small models emit curly/fullwidth punctuation that breaks JSON.parse."""
#     repl = {
#         "\u201c": '"',  # “
#         "\u201d": '"',  # ”
#         "\u2018": "'",  # ‘
#         "\u2019": "'",  # ’
#         "\uff02": '"',  # fullwidth "
#         "\uff0c": ",",  # ，
#         "\u3001": ",",
#         "\uff1a": ":",  # ：
#         "\u00a0": " ",
#     }
#     for a, b in repl.items():
#         text = text.replace(a, b)
#     # Trailing junk after a likely end of object
#     text = text.strip()
#     if text.startswith("{") and "}" in text:
#         # keep through last balanced-ish close if model appended garbage
#         pass
#     return text


# def _parse_structured_raw(
#     raw: str,
#     finish_reason: str | None,
#     max_tokens: int,
#     *,
#     user_text: str | None = None,
# ) -> dict[str, Any]:
#     """Parse model JSON → answer / extracted_facts / tool_calls dict."""
#     text = _sanitize_model_json_text((raw or "").strip())
#     if not text:
#         raise ValueError(
#             f"Empty structured output (finish_reason={finish_reason!r}, max_tokens={max_tokens})"
#         )

#     if "```" in text:
#         start = text.find("```")
#         rest = text[start + 3 :]
#         if rest.lstrip().lower().startswith("json"):
#             rest = rest.lstrip()[4:]
#         end = rest.find("```")
#         text = (rest[:end] if end >= 0 else rest).strip()

#     data: dict | None = None
#     last_err: Exception | None = None
#     for candidate in (text, _sanitize_model_json_text(text)):
#         try:
#             parsed = json.loads(candidate)
#             if not isinstance(parsed, dict):
#                 raise ValueError(
#                     f"Structured output root must be object, got {type(parsed).__name__}"
#                 )
#             data = parsed
#             break
#         except json.JSONDecodeError as e:
#             last_err = e
#             repaired = _try_repair_truncated_json(candidate)
#             if repaired is not None and (
#                 (repaired.get("answer") or "").strip()
#                 or repaired.get("tool_calls")
#             ):
#                 logger.warning(
#                     "Structured JSON incomplete (finish_reason=%r); salvaged. err=%s",
#                     finish_reason,
#                     e,
#                 )
#                 data = repaired
#                 break

#     if data is None:
#         snippet = text[:240].replace("\n", "\\n")
#         hint = ""
#         if finish_reason == "length":
#             hint = " Generation hit max_tokens mid-JSON."
#         raise ValueError(
#             f"Invalid structured JSON (finish_reason={finish_reason!r}, "
#             f"max_tokens={max_tokens}, len={len(text)}).{hint} "
#             f"Snippet: {snippet!r}. Error: {last_err}"
#         ) from last_err

#     return _normalize_structured_dict(data, user_text=user_text)


# def _latest_user_text(messages: list[dict]) -> str:
#     for m in reversed(messages):
#         if m.get("role") == "user":
#             return (m.get("content") or "").strip()
#     return ""


# # Recall-type turns: the user is asking about THEMSELVES or something stored
# # earlier. On these we inject memory; everything else (greetings, general chat)
# # gets nothing, so tokens stay low. Bilingual (English + Hinglish) patterns.
# _RECALL_INTENT_RE = re.compile(
#     r"\b(my name|your name|who am i|about me|about myself|know about me|"
#     r"remember|recall|earlier|before|last (chat|time|conversation)|"
#     r"what did i (say|tell)|where do i (live|work|stay)|"
#     r"my (name|city|job|work|company|address|number|email|age|birthday))\b"
#     r"|mera naam|mere baare|maine (kya|bataya|kaha)|yaad|pichl|"
#     r"kaha (rehta|kaam)|meri (city|job|company)",
#     re.IGNORECASE,
# )


# def _is_recall_intent(user_text: str) -> bool:
#     return bool(_RECALL_INTENT_RE.search(user_text or ""))


# # ===========================================================================
# # OLD — vLLM request payload (COMMENTED OUT).
# # vLLM-only: response_format.json_schema(strict) + extra_body guided_json /
# # xgrammar / structured_outputs / repetition_penalty. Gemini's OpenAI-compatible
# # endpoint does NOT understand these and would 400 or ignore them, so it is
# # replaced by the Gemini version below (same name + signature).
# # ---------------------------------------------------------------------------
# # def _structured_request_kwargs(
# #     final_messages: list[dict],
# #     *,
# #     temperature: float,
# #     max_tokens: int,
# #     stream: bool = False,
# #     schema: dict | None = None,
# # ) -> dict[str, Any]:
# #     # Low temp for JSON shape; room for full answer + facts (not a "short answer" policy).
# #     guided = schema or GUIDED_JSON_SCHEMA
# #     structured_temperature = min(float(temperature), 0.2)
# #     structured_max_tokens = min(
# #         STRUCTURED_MAX_MAX_TOKENS,
# #         max(
# #             STRUCTURED_MIN_MAX_TOKENS,
# #             int(max_tokens or STRUCTURED_MIN_MAX_TOKENS),
# #         ),
# #     )
# #     kwargs: dict[str, Any] = {
# #         "model": MODEL_NAME,
# #         "messages": final_messages,
# #         "temperature": structured_temperature,
# #         "max_tokens": structured_max_tokens,
# #         "response_format": {
# #             "type": "json_schema",
# #             "json_schema": {
# #                 "name": "structured_chat_output",
# #                 "schema": guided,
# #                 "strict": True,
# #             },
# #         },
# #         "extra_body": {
# #             "guided_json": guided,
# #             "guided_decoding_backend": "xgrammar:disable-any-whitespace",
# #             "structured_outputs": {
# #                 "json": guided,
# #                 "disable_any_whitespace": True,
# #                 "disable_additional_properties": True,
# #             },
# #             "repetition_penalty": 1.05,
# #         },
# #     }
# #     if stream:
# #         kwargs["stream"] = True
# #         kwargs["stream_options"] = {"include_usage": True}
# #     return kwargs


# # ===========================================================================
# # NEW — Gemini (OpenAI-compatible) request payload.
# # BASE_URL/API_KEY/MODEL_NAME come from backend/vllm/.env, e.g.
# #   BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
# #   MODEL_NAME=gemini-1.5-flash
# # Gemini speaks the OpenAI Chat Completions API, so we keep the SAME client,
# # the SAME agent loop, tools, streaming and parsing. Only the request body
# # changes: no vLLM extra_body; force valid JSON with response_format
# # json_object. The system prompt already spells out the exact JSON shape, and
# # the existing parser/repair handles anything the model gets slightly wrong.
# # `schema` is accepted for call-site compatibility but not sent to Gemini.
# # ===========================================================================
# def _structured_request_kwargs(
#     final_messages: list[dict],
#     *,
#     temperature: float,
#     max_tokens: int,
#     stream: bool = False,
#     schema: dict | None = None,
# ) -> dict[str, Any]:
#     _ = schema  # shape is enforced by the prompt, not a server-side grammar
#     structured_temperature = min(float(temperature), 0.3)
#     structured_max_tokens = min(
#         STRUCTURED_MAX_MAX_TOKENS,
#         max(
#             STRUCTURED_MIN_MAX_TOKENS,
#             int(max_tokens or STRUCTURED_MIN_MAX_TOKENS),
#         ),
#     )
#     kwargs: dict[str, Any] = {
#         "model": MODEL_NAME,
#         "messages": final_messages,
#         "temperature": structured_temperature,
#         "max_tokens": structured_max_tokens,
#         # Gemini honours json_object (valid-JSON guarantee); the shape itself
#         # is described in the system prompt.
#         "response_format": {"type": "json_object"},
#     }
#     if stream:
#         kwargs["stream"] = True
#     return kwargs


# async def _extract_relations(user_text: str) -> list[dict]:
#     """
#     Dedicated relation pass — backfills extracted_facts.relations, which the main
#     answer+facts call leaves empty on small models. Single task, temperature 0.
#     Only grounded (subject & object present in the message) triples are kept.
#     """
#     text = (user_text or "").strip()
#     if len(text.split()) < 3:  # greetings / one-word turns can't hold a relation
#         return []
#     messages = [
#         {"role": "system", "content": _RELATION_EXTRACT_PROMPT},
#         {"role": "user", "content": text},
#     ]
#     kwargs = _structured_request_kwargs(
#         messages,
#         temperature=0.0,
#         max_tokens=STRUCTURED_MIN_MAX_TOKENS,
#         stream=False,
#         schema=_RELATIONS_ONLY_SCHEMA,
#     )
#     try:
#         resp = await llm_client.chat.completions.create(**kwargs)
#         raw = (resp.choices[0].message.content or "") if resp.choices else ""
#     except Exception:
#         logger.exception("relation extraction call failed")
#         return []

#     data: dict | None = None
#     try:
#         parsed = json.loads(_sanitize_model_json_text(raw))
#         if isinstance(parsed, dict):
#             data = parsed
#     except json.JSONDecodeError:
#         data = _try_repair_truncated_json(raw)
#     if not isinstance(data, dict):
#         return []

#     rels = _normalize_relations(data.get("relations") or [])
#     # Reuse the same grounding rule the main path applies to relations.
#     grounded = filter_facts_against_user_text({"relations": rels}, text)
#     return grounded.get("relations") or []


# def _build_system_prompt(
#     *,
#     tools_block: str | None = None,
#     extra_memory_block: str | None = None,
#     base: str = STRUCTURED_SYSTEM_PROMPT,
# ) -> str:
#     parts = [base]
#     if tools_block and tools_block.strip():
#         parts.append(tools_block.strip())
#     if extra_memory_block and extra_memory_block.strip():
#         parts.append(extra_memory_block.strip())
#     return "\n\n".join(parts)


# async def _execute_tool_calls(
#     tool_calls: list[dict[str, str]],
#     *,
#     user_id: str,
#     session_id: str,
# ) -> tuple[str, list[str]]:
#     """Run registry tools; return (results text for next prompt, names used)."""
#     from agent_tools.registry import ToolContext, get_tool_registry

#     registry = get_tool_registry()
#     ctx = ToolContext(user_id=user_id, session_id=session_id)
#     used: list[str] = []
#     chunks: list[str] = []
#     for tc in tool_calls:
#         name = tc.get("name") or ""
#         res = await registry.execute(name, tc.get("arguments"), ctx)
#         used.append(name)
#         if res.ok:
#             chunks.append(f"[tool:{name}]\n{res.result}")
#         else:
#             chunks.append(f"[tool:{name} ERROR]\n{res.error}")
#     return "\n\n".join(chunks), used


# async def _stream_one_structured_round(
#     final_messages: list[dict],
#     *,
#     temperature: float,
#     max_tokens: int,
#     schema: dict,
#     user_text: str,
#     round_i: int,
#     emit_answer_deltas: bool,
# ) -> AsyncIterator[dict[str, Any]]:
#     """
#     One guided-JSON LLM call with live token stream.
#     Yields answer_delta events (if emit_answer_deltas), then a single
#     {"type":"_round_done", "norm":..., "finish_reason":..., "raw":...}.
#     """
#     kwargs = _structured_request_kwargs(
#         final_messages,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         stream=True,
#         schema=schema,
#     )
#     structured_max_tokens = kwargs["max_tokens"]
#     _debug_print_final_messages(
#         final_messages, f"agent_stream round={round_i} → create"
#     )

#     raw_parts: list[str] = []
#     emitted_answer_len = 0
#     finish_reason: str | None = None
#     trailing_ws = 0  # consecutive whitespace chars seen at the tail of the stream

#     stream = await llm_client.chat.completions.create(**kwargs)
#     async for chunk in stream:
#         if not chunk.choices:
#             continue
#         choice = chunk.choices[0]
#         if choice.finish_reason:
#             finish_reason = choice.finish_reason
#         delta = choice.delta.content if choice.delta else None
#         if not delta:
#             continue
#         raw_parts.append(delta)

#         # Runaway-pad guard: track the trailing whitespace run across deltas.
#         # A long run means the model is padding (not formatting) — stop the
#         # round now so the answer we already have isn't truncated by max_tokens.
#         stripped_delta = delta.rstrip()
#         if stripped_delta == "":
#             trailing_ws += len(delta)
#         else:
#             trailing_ws = len(delta) - len(stripped_delta)
#         if trailing_ws >= PAD_RUN_ABORT:
#             logger.warning(
#                 "Runaway whitespace pad detected (%d chars) at round=%d; "
#                 "aborting stream early.",
#                 trailing_ws,
#                 round_i,
#             )
#             finish_reason = finish_reason or "pad_abort"
#             try:
#                 await stream.close()
#             except Exception:  # noqa: BLE001 - best-effort close
#                 pass
#             break

#         if emit_answer_deltas:
#             raw_so_far = "".join(raw_parts)
#             partial = partial_answer_from_raw_json(raw_so_far)
#             if len(partial) > emitted_answer_len:
#                 new_text = partial[emitted_answer_len:]
#                 # Skip pure pad (only whitespace) so UI doesn't flood; keep real newlines
#                 if not new_text.strip():
#                     emitted_answer_len = len(partial)
#                     continue
#                 emitted_answer_len = len(partial)
#                 yield {"type": "answer_delta", "text": new_text}

#     raw = "".join(raw_parts)
#     _debug_print_final_output(
#         {
#             "raw_content": raw[:2000] + ("…" if len(raw) > 2000 else ""),
#             "finish_reason": finish_reason,
#             "round": round_i,
#         },
#         f"agent_stream round={round_i} ← raw",
#         variable="raw",
#     )
#     norm = _parse_structured_raw(
#         raw, finish_reason, structured_max_tokens, user_text=user_text
#     )
#     # Catch-up: if JSON closed with more answer text than we streamed mid-flight
#     final_answer = norm.get("answer") or ""
#     if emit_answer_deltas and final_answer:
#         streamed_partial = partial_answer_from_raw_json(raw)
#         # Prefer: if final cleaned answer is prefix-compatible with what user saw
#         seen = streamed_partial[:emitted_answer_len] if emitted_answer_len else ""
#         if final_answer.startswith(seen) and len(final_answer) > len(seen):
#             yield {"type": "answer_delta", "text": final_answer[len(seen) :]}
#         elif not seen and final_answer:
#             # streamed only pads / nothing useful — send clean answer once
#             yield {"type": "answer_delta", "text": final_answer}

#     yield {
#         "type": "_round_done",
#         "norm": norm,
#         "finish_reason": finish_reason,
#         "raw": raw,
#         "emitted_answer_len": emitted_answer_len,
#     }


# @observe()
# async def run_chat_structured_stream(
#     messages: list[dict],
#     temperature: float,
#     max_tokens: int,
#     *,
#     tools_enabled: bool = False,
#     user_id: str | None = None,
#     session_id: str | None = None,
# ) -> AsyncIterator[dict[str, Any]]:

#     from agent_tools.registry import get_tool_registry, tools_prompt_block

#     try:
#         user_text = _latest_user_text(messages)

#         # Memory is injected ONLY on recall-type turns (the backend detects the
#         # intent), never on greetings/normal chat — so tokens stay low. We do NOT
#         # rely on the small model to decide to call a search tool: both gemma and
#         # phi often refuse or skip it. When the user asks about themselves/their
#         # stored info, we fetch a compact memory block and inject it here.
#         memory_block = ""
#         recalled_flag = False
#         if user_id and session_id and _is_recall_intent(user_text):
#             from agent_memory.bridge import recall_memory_block

#             try:
#                 memory_block, recalled_flag = await recall_memory_block(
#                     user_id=user_id, session_id=session_id, messages=messages
#                 )
#             except Exception:
#                 logger.exception("recall injection failed")

#         tools_used: list[str] = []
#         tools_block = ""
#         schema = GUIDED_JSON_SCHEMA
#         base_prompt = STRUCTURED_SYSTEM_PROMPT
#         if tools_enabled and user_id and session_id:
#             specs = get_tool_registry().list_tools()
#             tools_block = tools_prompt_block(specs)
#             schema = GUIDED_JSON_SCHEMA_WITH_TOOLS
#             base_prompt = STRUCTURED_SYSTEM_PROMPT_WITH_TOOLS

#         work_messages = [m for m in messages if m.get("role") != "system"]
#         tool_results_block = ""
#         last_norm: dict[str, Any] = {
#             "answer": "",
#             "extracted_facts": {
#                 "entities": [],
#                 "facts_about_user": [],
#                 "constraints": [],
#                 "relations": [],
#             },
#             "tool_calls": [],
#         }
#         finish_reason: str | None = None

#         for round_i in range(MAX_TOOL_ROUNDS + 1):
#             extra_bits = []
#             if memory_block:
#                 extra_bits.append(memory_block)
#             if tool_results_block:
#                 extra_bits.append(
#                     "TOOL RESULTS (from your previous tool_calls — use to answer now):\n"
#                     + tool_results_block
#                 )
#             system_prompt = _build_system_prompt(
#                 tools_block=tools_block,
#                 extra_memory_block="\n\n".join(extra_bits) if extra_bits else None,
#                 base=base_prompt,
#             )
#             if tool_results_block:
#                 system_prompt += (
#                     "\n\nYou already called tools. Prefer tool_calls=[] and fill answer now."
#                 )

#             final_messages = _with_system_and_memory(
#                 work_messages,
#                 system_prompt=system_prompt,
#             )

#             # Always stream tokens live (answer field as it grows)
#             round_done: dict[str, Any] | None = None
#             async for ev in _stream_one_structured_round(
#                 final_messages,
#                 temperature=temperature,
#                 max_tokens=max_tokens,
#                 schema=schema,
#                 user_text=user_text,
#                 round_i=round_i,
#                 emit_answer_deltas=True,
#             ):
#                 if ev.get("type") == "_round_done":
#                     round_done = ev
#                     continue
#                 yield ev

#             if not round_done:
#                 yield {"type": "error", "message": "stream round produced no result"}
#                 return

#             last_norm = round_done["norm"]
#             finish_reason = round_done.get("finish_reason")
#             tcalls = last_norm.get("tool_calls") or []
#             answer_text = (last_norm.get("answer") or "").strip()

#             known_tool_names = (
#                 set(get_tool_registry().names()) if tools_enabled else set()
#             )
#             known_lower = {n.lower() for n in known_tool_names}
#             bare_tool_answer = bool(
#                 tools_enabled
#                 and not tcalls
#                 and answer_text
#                 and (
#                     answer_text in known_tool_names
#                     or answer_text.lower() in known_lower
#                     or answer_text.lower().replace(" ", "_") in known_lower
#                 )
#             )
#             if bare_tool_answer and round_i < MAX_TOOL_ROUNDS:
#                 tool_results_block = (
#                     (tool_results_block + "\n\n" if tool_results_block else "")
#                     + "SYSTEM NOTE: answer was only a tool name. "
#                     "Call via tool_calls with arguments, or write a full reply."
#                 )
#                 continue

#             if (
#                 tcalls
#                 and tools_enabled
#                 and user_id
#                 and session_id
#                 and round_i < MAX_TOOL_ROUNDS
#             ):
#                 for tc in tcalls:
#                     yield {
#                         "type": "tool_call",
#                         "name": tc.get("name") or "",
#                         "arguments": tc.get("arguments") or "",
#                     }
#                 results_text, used = await _execute_tool_calls(
#                     tcalls, user_id=user_id, session_id=session_id
#                 )
#                 tools_used.extend(used)
#                 for name in used:
#                     yield {"type": "tool_result", "name": name, "ok": True}
#                 tool_results_block = (
#                     (tool_results_block + "\n\n" if tool_results_block else "")
#                     + results_text
#                 )
#                 logger.info(
#                     "agent_stream tools executed: %s (round=%s)", used, round_i
#                 )
#                 # next round streams the real user-facing answer
#                 continue

#             break

#         answer = last_norm.get("answer") or ""
#         facts = last_norm.get("extracted_facts") or {
#             "entities": [],
#             "facts_about_user": [],
#             "constraints": [],
#             "relations": [],
#         }

#         # Backfill relations via the dedicated pass when the main call left them
#         # empty (the common case on small models). Merge into entities so the
#         # graph gets both the nodes and the RELATES_TO edges.
#         if not facts.get("relations"):
#             backfilled = await _extract_relations(user_text)
#             if backfilled:
#                 facts["relations"] = backfilled
#                 names = {e for e in (facts.get("entities") or [])}
#                 for r in backfilled:
#                     for side in (r.get("subject"), r.get("object")):
#                         if side and side not in names:
#                             names.add(side)
#                             facts.setdefault("entities", []).append(side)
#                 logger.info("relations backfilled: %d", len(backfilled))

#         _debug_print_final_output(
#             {
#                 "answer": answer,
#                 "extracted_facts": facts,
#                 "tools_used": tools_used,
#                 "finish_reason": finish_reason,
#             },
#             "agent_stream ← final result",
#             variable="result",
#         )
#         yield {
#             "type": "final",
#             "answer": answer,
#             "extracted_facts": facts,
#             "finish_reason": finish_reason,
#             "tools_used": tools_used,
#             "recalled": recalled_flag,
#         }
#     except Exception as e:
#         logger.exception("structured stream failed")
#         yield {"type": "error", "message": str(e)}









# ===========================================================================
# FUNCTION MAP — kiska kya kaam (active code neeche isi order me hai)
# ---------------------------------------------------------------------------
# 1) ENTRY POINTS (bahar se call hote):
#    get_health_info               -> /health ka jawab (base_url, model)
#    run_chat_structured_stream    -> MAIN: pura agent loop (tools + streaming + rounds)
#
# 2) PROMPT / MESSAGE taiyaar karna:
#    _latest_user_text             -> messages me se aakhri user message
#    _sanitize_chat_roles          -> role order theek (user/assistant alternate, user se shuru)
#    _build_system_prompt          -> base prompt + tools list + tool-results ek string me
#    _with_system_and_memory       -> wo system string ko messages ke aage lagata hai
#
# 3) FACTS saaf karna + GROUNDING (anti-hallucination):
#    _dedupe_list                  -> duplicate/khaali facts hatata hai
#    _item_grounded_in_user        -> fact sach me user ke message me hai? (warna drop)
#    _normalize_predicate          -> relation predicate -> UPPER_SNAKE
#    _normalize_relations          -> relation triples saaf/dedupe
#    filter_facts_against_user_text-> bina-grounded (banawati) facts/relations girata hai
#    _normalize_tool_calls         -> model ke tool_calls ko {name, arguments} me saaf
#    _normalize_structured_dict    -> upar walon se final {answer, facts, tool_calls}
#
# 4) Model ka JSON PARSE karna:
#    partial_answer_from_raw_json  -> aadhe-bane JSON se answer nikaal ke live token dikhata hai
#    _loads_lenient                -> json.loads + truncate hone par chhota repair
#    _parse_structured_raw         -> raw -> normalized dict
#
# 5) REQUEST + TOOLS + ek round:
#    _structured_request_kwargs    -> OpenAI request body (json_object)
#    _execute_tool_calls           -> model ne jo tools mange wo chalata hai
#    _stream_one_structured_round  -> ek LLM call, stream karke deltas + result deta hai
# ===========================================================================


"""
LLM client — OpenAI-compatible structured chat with a small agent loop.

Entry points:
  get_health_info
  run_chat_structured_stream   (tool_calls -> answer + extracted_facts, streamed)

Works against ANY OpenAI-compatible endpoint (vLLM, Gemini, OpenRouter, ...).
Each turn the model returns ONE JSON object (answer + extracted_facts, plus
tool_calls when tools are enabled). The strict system prompt + a lenient parser
enforce the shape — json_object is opt-in (JSON_OBJECT_MODE) since it disables
token streaming on providers that buffer to validate the JSON.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import AsyncIterator
from typing import Any

from langfuse import observe
from langfuse.openai import AsyncOpenAI

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

STRUCTURED_MIN_MAX_TOKENS = 1024
STRUCTURED_MAX_MAX_TOKENS = 2048
FACT_ARRAY_MAX_ITEMS = 8
MAX_TOOL_ROUNDS = 2

# response_format=json_object GUARANTEES valid JSON but many providers buffer the
# whole reply to validate it — so token streaming stops (answer arrives all at
# once). We rely on the strict system prompt + lenient parser instead, which
# keeps live token streaming. Flip to True only if a model needs the hard JSON
# guarantee and you don't mind losing streaming.
JSON_OBJECT_MODE = False


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_FACTS_EMPTY_RULE = (
    "extracted_facts ONLY from the latest user message. "
    "If nothing to extract, leave entities/facts/constraints/relations as empty [] — "
    "do not invent. Empty is correct for chit-chat. "
    "facts_about_user: store each personal fact as a short clear statement that "
    'INCLUDES the value — e.g. "name is Vipin", "works at Swiggy", "lives in Delhi" '
    "— so it can be found later by keyword. Always add a name fact when the user "
    "tells you their name. "
    'relations: link two named things as {"subject":"..","predicate":"UPPER_SNAKE","object":".."} '
    "using the exact words in this message."
)

STRUCTURED_SYSTEM_PROMPT = (
    "Reply in the user's language. ONE JSON object only. ASCII quotes only. "
    'Shape: {"answer":"<full reply>","extracted_facts":{"entities":[],'
    '"facts_about_user":[],"constraints":[],"relations":[]}}. '
    "Write the FULL answer, then fill extracted_facts. "
    f"{_FACTS_EMPTY_RULE}"
)

STRUCTURED_SYSTEM_PROMPT_WITH_TOOLS = (
    "Reply in the user's language. ONE JSON object only. ASCII quotes only.\n"
    'Shape: {"tool_calls":[],"answer":"<full reply>","extracted_facts":'
    '{"entities":[],"facts_about_user":[],"constraints":[],"relations":[]}}\n'
    "tool_calls if you need a tool (else []); answer may be \"\" while a tool runs.\n"
    "Tools: search_conversation (past chat / ES), search_context (graph facts).\n"
    "DECIDE per message:\n"
    "- Greeting / chit-chat / general knowledge (hello, hi, how are you, thanks, "
    "jokes, facts) -> tool_calls=[] and write a normal friendly reply. Do NOT "
    "search, do NOT mention names. A plain 'hello' just gets a plain greeting.\n"
    "- If the user asks about THEMSELVES or something they told you before "
    "(name/city/job, 'what do you know about me', 'what did I say about X') -> this "
    "is their own saved data, not a privacy issue: call search_context (stored "
    "facts) or search_conversation (exact past wording) with answer=\"\". After "
    "TOOL RESULTS arrive, answer from them; if nothing is found, say you don't have "
    "it stored yet. Never invent a name, never refuse.\n"
    f"{_FACTS_EMPTY_RULE}"
)

def get_health_info() -> dict:
    return {"status": "ok", "base_url": BASE_URL, "model": MODEL_NAME}


def _latest_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


# ---------------------------------------------------------------------------
# Chat message shaping
# ---------------------------------------------------------------------------
def _sanitize_chat_roles(messages: list[dict]) -> list[dict]:
    """Keep user/assistant only, start with user, no two same roles in a row."""
    rest = [
        {"role": m.get("role"), "content": m.get("content") or ""}
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]
    while rest and rest[0]["role"] != "user":
        rest.pop(0)
    cleaned: list[dict] = []
    for m in rest:
        if cleaned and cleaned[-1]["role"] == m["role"]:
            cleaned[-1] = m  # merge consecutive same-role (keep latest)
            continue
        cleaned.append(m)
    if cleaned and cleaned[-1]["role"] == "assistant":
        cleaned.pop()  # end on user (generation turn)
    return cleaned


def _with_system_and_memory(
    messages: list[dict], *, system_prompt: str = STRUCTURED_SYSTEM_PROMPT
) -> list[dict]:
    """Prepend the (already-composed) system prompt to the sanitized turns."""
    return [{"role": "system", "content": system_prompt}, *_sanitize_chat_roles(messages)]


def _build_system_prompt(
    *,
    tools_block: str | None = None,
    extra_memory_block: str | None = None,
    base: str = STRUCTURED_SYSTEM_PROMPT,
) -> str:
    parts = [base]
    if tools_block and tools_block.strip():
        parts.append(tools_block.strip())
    if extra_memory_block and extra_memory_block.strip():
        parts.append(extra_memory_block.strip())
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Fact normalization + grounding (drop hallucinated extracts)
# ---------------------------------------------------------------------------
def _dedupe_list(items: list, *, max_items: int = FACT_ARRAY_MAX_ITEMS) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items or []:
        s = str(raw).strip()
        if not s or s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
        if len(out) >= max_items:
            break
    return out


def _item_grounded_in_user(item: str, user_lower: str) -> bool:
    """Cheap anti-hallucination: fact must appear in / overlap user text."""
    sl = (item or "").strip().lower()
    if not sl:
        return False
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
        if not isinstance(r, dict):
            continue
        sub = str(r.get("subject") or "").strip()
        obj = str(r.get("object") or "").strip()
        pred = _normalize_predicate(str(r.get("predicate") or ""))
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
    """Keep only entities/facts/relations grounded in the user's own text."""
    user_lower = (user_text or "").lower()
    empty = {"entities": [], "facts_about_user": [], "constraints": [], "relations": []}
    if not user_lower.strip():
        return empty
    out: dict = {}
    for key in ("entities", "facts_about_user", "constraints"):
        kept = [
            str(item).strip()
            for item in facts.get(key) or []
            if _item_grounded_in_user(str(item), user_lower)
        ]
        out[key] = _dedupe_list(kept)
    rels_kept = [
        r
        for r in facts.get("relations") or []
        if isinstance(r, dict)
        and _item_grounded_in_user(str(r.get("subject") or ""), user_lower)
        and _item_grounded_in_user(str(r.get("object") or ""), user_lower)
    ]
    out["relations"] = _normalize_relations(rels_kept)
    return out


def _normalize_tool_calls(raw: list | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        args = item.get("arguments", "")
        args_s = (
            json.dumps(args, ensure_ascii=False, separators=(",", ":"))
            if isinstance(args, dict)
            else str(args or "")
        )
        out.append({"name": name, "arguments": args_s})
        if len(out) >= 2:
            break
    return out


def _normalize_structured_dict(data: dict, *, user_text: str | None = None) -> dict:
    answer = data.get("answer") or ""
    answer = (answer if isinstance(answer, str) else str(answer)).strip()

    facts_raw = data.get("extracted_facts")
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

    return {
        "answer": answer,
        "extracted_facts": facts,
        "tool_calls": _normalize_tool_calls(data.get("tool_calls")),
    }


# ---------------------------------------------------------------------------
# JSON parsing (response_format=json_object gives valid JSON; the lenient
# fallback only covers a mid-JSON cut-off when finish_reason == "length")
# ---------------------------------------------------------------------------
def partial_answer_from_raw_json(raw: str) -> str:
    """Live stream: the answer string decoded from an incomplete JSON buffer."""
    m = re.search(r'"answer"\s*:\s*"', raw)
    if not m:
        return ""
    i = m.end()
    out: list[str] = []
    escape = False
    while i < len(raw):
        c = raw[i]
        if escape:
            out.append({"n": "\n", "t": "\t", "r": "\r"}.get(c, c))
            escape = False
        elif c == "\\":
            escape = True
        elif c == '"':
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _loads_lenient(text: str) -> dict | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        pass
    # Minimal repair for a truncated object: close open string/brackets/braces.
    cand = text
    if cand.count('"') % 2:
        cand += '"'
    cand += "]" * max(0, cand.count("[") - cand.count("]"))
    cand += "}" * max(0, cand.count("{") - cand.count("}"))
    try:
        d = json.loads(cand)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_structured_raw(
    raw: str, finish_reason: str | None, *, user_text: str | None = None
) -> dict[str, Any]:
    data = _loads_lenient(raw)
    if data is None:
        # Not valid JSON (rare without json_object) — don't error: treat whatever
        # text the model produced as the answer so the user still gets a reply.
        logger.warning("structured output not JSON (finish_reason=%r); using raw as answer", finish_reason)
        data = {"answer": (raw or "").strip(), "extracted_facts": {}, "tool_calls": []}
    return _normalize_structured_dict(data, user_text=user_text)


# ---------------------------------------------------------------------------
# Request + agent loop
# ---------------------------------------------------------------------------
def _structured_request_kwargs(
    final_messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    stream: bool = False,
) -> dict[str, Any]:
    """OpenAI-compatible request. Shape is enforced by the system prompt + the
    lenient parser; json_object is opt-in (JSON_OBJECT_MODE) because it kills
    token streaming on providers that buffer to validate the JSON."""
    structured_max_tokens = min(
        STRUCTURED_MAX_MAX_TOKENS,
        max(STRUCTURED_MIN_MAX_TOKENS, int(max_tokens or STRUCTURED_MIN_MAX_TOKENS)),
    )
    kwargs: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": final_messages,
        "temperature": min(float(temperature), 0.3),
        "max_tokens": structured_max_tokens,
    }
    if JSON_OBJECT_MODE:
        kwargs["response_format"] = {"type": "json_object"}
    if stream:
        kwargs["stream"] = True
    return kwargs


async def _execute_tool_calls(
    tool_calls: list[dict[str, str]], *, user_id: str, session_id: str
) -> tuple[str, list[str]]:
    """Run registry tools; return (results text for the next prompt, names used)."""
    from agent_tools.registry import ToolContext, get_tool_registry

    registry = get_tool_registry()
    ctx = ToolContext(user_id=user_id, session_id=session_id)
    used: list[str] = []
    chunks: list[str] = []
    for tc in tool_calls:
        name = tc.get("name") or ""
        res = await registry.execute(name, tc.get("arguments"), ctx)
        used.append(name)
        chunks.append(
            f"[tool:{name}]\n{res.result}" if res.ok else f"[tool:{name} ERROR]\n{res.error}"
        )
    return "\n\n".join(chunks), used


async def _stream_one_structured_round(
    final_messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    user_text: str,
    round_i: int,
) -> AsyncIterator[dict[str, Any]]:
    """One LLM call, streamed. Yields answer_delta events then one _round_done."""
    kwargs = _structured_request_kwargs(
        final_messages, temperature=temperature, max_tokens=max_tokens, stream=True
    )
    logger.debug("round=%d create (%d messages)", round_i, len(final_messages))

    raw_parts: list[str] = []
    streamed = ""  # what we've actually sent to the user so far
    finish_reason: str | None = None

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
        buf = "".join(raw_parts)
        # Hybrid live view: if the model is emitting a JSON object, pull the answer
        # field out of it; if it's plain prose (no json_object enforcement), show
        # the text as-is. Either way the user sees tokens as they arrive.
        answer_so_far = (
            partial_answer_from_raw_json(buf) if buf.lstrip()[:1] in ("{", "`") else buf
        )
        if answer_so_far.startswith(streamed) and len(answer_so_far) > len(streamed):
            new_text = answer_so_far[len(streamed):]
            streamed = answer_so_far
            if new_text.strip():
                yield {"type": "answer_delta", "text": new_text}

    raw = "".join(raw_parts)
    logger.debug("round=%d raw=%s", round_i, raw[:500])
    norm = _parse_structured_raw(raw, finish_reason, user_text=user_text)

    # Catch-up: emit any part of the final (cleaned) answer not yet streamed.
    final_answer = norm.get("answer") or ""
    if final_answer and final_answer != streamed:
        if final_answer.startswith(streamed):
            yield {"type": "answer_delta", "text": final_answer[len(streamed):]}
        elif not streamed.strip():
            yield {"type": "answer_delta", "text": final_answer}

    yield {"type": "_round_done", "norm": norm, "finish_reason": finish_reason}


@observe()
async def run_chat_structured_stream(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    *,
    tools_enabled: bool = False,
    user_id: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    from agent_tools.registry import get_tool_registry, tools_prompt_block

    try:
        user_text = _latest_user_text(messages)

        # No hardcoded intent detection: the model decides via guided JSON when to
        # call the search tools (search_context / search_conversation) to pull
        # memory. Backend only executes what the model asks for.
        tools_block = ""
        base_prompt = STRUCTURED_SYSTEM_PROMPT
        if tools_enabled and user_id and session_id:
            tools_block = tools_prompt_block(get_tool_registry().list_tools())
            base_prompt = STRUCTURED_SYSTEM_PROMPT_WITH_TOOLS

        work_messages = [m for m in messages if m.get("role") != "system"]
        tools_used: list[str] = []
        tool_results_block = ""
        last_norm: dict[str, Any] = {
            "answer": "",
            "extracted_facts": {
                "entities": [],
                "facts_about_user": [],
                "constraints": [],
                "relations": [],
            },
            "tool_calls": [],
        }
        finish_reason: str | None = None

        for round_i in range(MAX_TOOL_ROUNDS + 1):
            is_last_round = round_i == MAX_TOOL_ROUNDS
            extra_bits = []
            if tool_results_block:
                extra_bits.append(
                    "TOOL RESULTS (from your previous tool_calls — use to answer now):\n"
                    + tool_results_block
                )
            # On the final round, drop tools entirely so the model MUST write a
            # reply — no endless tool-calling that leaves the answer empty.
            system_prompt = _build_system_prompt(
                tools_block="" if is_last_round else tools_block,
                extra_memory_block="\n\n".join(extra_bits) if extra_bits else None,
                base=STRUCTURED_SYSTEM_PROMPT if is_last_round else base_prompt,
            )
            if tool_results_block:
                system_prompt += (
                    "\n\nYou already called tools. Answer NOW from the results above; "
                    "tool_calls=[]. If the results are empty, say you don't have it stored."
                )

            final_messages = _with_system_and_memory(
                work_messages, system_prompt=system_prompt
            )

            round_done: dict[str, Any] | None = None
            async for ev in _stream_one_structured_round(
                final_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                user_text=user_text,
                round_i=round_i,
            ):
                if ev.get("type") == "_round_done":
                    round_done = ev
                    continue
                yield ev

            if not round_done:
                yield {"type": "error", "message": "stream round produced no result"}
                return

            last_norm = round_done["norm"]
            finish_reason = round_done.get("finish_reason")
            tcalls = last_norm.get("tool_calls") or []

            can_call_tools = (
                tools_enabled and user_id and session_id and round_i < MAX_TOOL_ROUNDS
            )
            if tcalls and can_call_tools:
                for tc in tcalls:
                    yield {
                        "type": "tool_call",
                        "name": tc.get("name") or "",
                        "arguments": tc.get("arguments") or "",
                    }
                results_text, used = await _execute_tool_calls(
                    tcalls, user_id=user_id, session_id=session_id
                )
                tools_used.extend(used)
                for name in used:
                    yield {"type": "tool_result", "name": name, "ok": True}
                tool_results_block = (
                    (tool_results_block + "\n\n" if tool_results_block else "") + results_text
                )
                logger.info("tools executed: %s (round=%s)", used, round_i)
                continue  # next round produces the user-facing answer

            break

        answer = (last_norm.get("answer") or "").strip()
        facts = last_norm.get("extracted_facts") or {
            "entities": [],
            "facts_about_user": [],
            "constraints": [],
            "relations": [],
        }
        # Safety net: the model must never leave the user with a blank reply.
        if not answer:
            answer = "Sorry, I didn't quite get a reply together — could you say that another way?"
            yield {"type": "answer_delta", "text": answer}

        yield {
            "type": "final",
            "answer": answer,
            "extracted_facts": facts,
            "finish_reason": finish_reason,
            "tools_used": tools_used,
        }
    except Exception as e:
        logger.exception("structured stream failed")
        yield {"type": "error", "message": str(e)}
