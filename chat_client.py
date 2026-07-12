"""
chat_client.py — live token print via /chat/structured/stream + server memory.
"""

from __future__ import annotations

import json
import uuid

import requests

BASE = "http://localhost:5000"
CHAT_STREAM_URL = f"{BASE}/chat/structured/stream"
MEMORY_HEALTH = f"{BASE}/memory/health"
MEMORY_RECALL = f"{BASE}/memory/recall"

USER_ID = "local-user"
SESSION_ID = str(uuid.uuid4())
HISTORY_WINDOW = 12

messages: list[dict] = []
facts: dict = {
    "entities": [],
    "facts_about_user": [],
    "constraints": [],
    "relations": [],
}


def merge_facts(new_facts: dict):
    if not isinstance(new_facts, dict):
        return
    for category in ("entities", "facts_about_user", "constraints"):
        if category not in facts:
            facts[category] = []
        new_items = new_facts.get(category) or []
        if not isinstance(new_items, list):
            continue
        for item in new_items:
            if not item:
                continue
            existing_lower = {x.lower() for x in facts[category]}
            if str(item).lower() not in existing_lower:
                facts[category].append(item)
    # relations: list of {subject, predicate, object}
    if "relations" not in facts:
        facts["relations"] = []
    for r in new_facts.get("relations") or []:
        if not isinstance(r, dict):
            continue
        sub = str(r.get("subject") or "").strip()
        pred = str(r.get("predicate") or "").strip()
        obj = str(r.get("object") or "").strip()
        if not sub or not obj:
            continue
        key = f"{sub.lower()}|{pred.lower()}|{obj.lower()}"
        existing = {
            f"{x.get('subject','').lower()}|{x.get('predicate','').lower()}|{x.get('object','').lower()}"
            for x in facts["relations"]
            if isinstance(x, dict)
        }
        if key not in existing:
            facts["relations"].append(
                {"subject": sub, "predicate": pred, "object": obj}
            )


def print_facts_summary():
    if not any(facts.get(k) for k in facts):
        print("\n(Local facts cache empty — server KG alag ho sakta hai)\n")
        return
    print("\n--- LOCAL FACTS CACHE ---")
    for category, items in facts.items():
        if items:
            print(f"  {category}: {items}")
    print("---------------------------\n")


def windowed_messages() -> list[dict]:
    if len(messages) <= HISTORY_WINDOW:
        return list(messages)
    return list(messages[-HISTORY_WINDOW:])


def send_query(user_input: str) -> tuple[str, dict, dict | None]:
    messages.append({"role": "user", "content": user_input})

    payload = {
        "messages": windowed_messages(),
        "memory": {
            "entities": list(facts["entities"]),
            "facts_about_user": list(facts["facts_about_user"]),
            "constraints": list(facts["constraints"]),
            "relations": list(facts.get("relations") or []),
        },
        "user_id": USER_ID,
        "session_id": SESSION_ID,
        "use_agent_memory": True,
        "max_tokens": 2048,
    }

    answer = ""
    extracted: dict = {}
    mem_status = None

    try:
        with requests.post(
            CHAT_STREAM_URL, json=payload, stream=True, timeout=300
        ) as response:
            if response.status_code != 200:
                try:
                    error_body = response.json()
                except Exception:
                    error_body = response.text
                print(f"\n[BACKEND ERROR {response.status_code}]: {error_body}\n")
                messages.pop()
                response.raise_for_status()

            print("\nBot: ", end="", flush=True)
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                else:
                    continue
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")
                if etype == "answer_delta":
                    text = event.get("text") or ""
                    print(text, end="", flush=True)
                    answer += text
                elif etype == "final":
                    answer = event.get("answer") or answer
                    extracted = event.get("extracted_facts") or {}
                    mem_status = event.get("memory_status")
                elif etype == "error":
                    print(f"\n[STREAM ERROR] {event.get('message')}\n")
                    messages.pop()
                    raise RuntimeError(event.get("message") or "stream error")
            print("\n")
    except Exception:
        if messages and messages[-1].get("role") == "user":
            messages.pop()
        raise

    merge_facts(extracted)
    messages.append({"role": "assistant", "content": answer})
    return answer, extracted, mem_status


def server_recall(query: str | None = None):
    payload = {
        "user_id": USER_ID,
        "session_id": SESSION_ID,
        "query": query,
        "recent_limit": 20,
        "search_limit": 10,
        "graph_limit": 40,
    }
    r = requests.post(MEMORY_RECALL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def server_clear_session():
    r = requests.delete(
        f"{BASE}/memory/session",
        params={"user_id": USER_ID, "session_id": SESSION_ID},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def server_clear_user():
    r = requests.delete(f"{BASE}/memory/user/{USER_ID}", timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    print("Chat + live stream + server agent memory")
    print(f"  user_id    = {USER_ID}")
    print(f"  session_id = {SESSION_ID}")
    print("  facts / health / recall / clear / wipe / exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        low = user_input.lower()
        if low in ("exit", "quit"):
            print("Bye!")
            break
        if low == "facts":
            print_facts_summary()
            continue
        if low == "health":
            try:
                h = requests.get(MEMORY_HEALTH, timeout=30).json()
                print(json.dumps(h, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"\n[ERROR] {e}\n")
            continue
        if low == "recall" or low.startswith("recall "):
            q = user_input[6:].strip() or None
            try:
                data = server_recall(q)
                print("\n--- SERVER RECALL ---")
                print(data.get("memory_block") or "(empty)")
                print("---------------------\n")
            except Exception as e:
                print(f"\n[ERROR] {e}\n")
            continue
        if low == "clear":
            messages.clear()
            for k in facts:
                facts[k] = []
            try:
                server_clear_session()
                print("\n(Local + session clear; KG kept)\n")
            except Exception as e:
                print(f"\n(Local clear; server failed: {e})\n")
            continue
        if low == "wipe":
            messages.clear()
            for k in facts:
                facts[k] = []
            try:
                server_clear_user()
                print("\n(Full user wipe)\n")
            except Exception as e:
                print(f"\n(Local clear; server failed: {e})\n")
            continue

        try:
            _answer, turn_facts, mem_status = send_query(user_input)
            print("--- this turn new facts ---")
            print(json.dumps(turn_facts, indent=2, ensure_ascii=False))
            if mem_status:
                print("--- memory_status ---")
                print(json.dumps(mem_status, indent=2, ensure_ascii=False))
            print("--- local facts cache ---")
            print(json.dumps(facts, indent=2, ensure_ascii=False))
            print("-------------------------\n")
        except requests.exceptions.RequestException as e:
            print(f"\n[ERROR] Backend: {e}\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()
