"""
chat_client.py — thin client for POST /chat/structured/stream.

Does NOT decide memory on/off. Backend always runs agent memory + tools;
the model decides whether to call tools.
"""

from __future__ import annotations

import getpass
import json

import requests

BASE = "http://localhost:5000"
CHAT_STREAM_URL = f"{BASE}/chat/structured/stream"
MEMORY_HEALTH = f"{BASE}/memory/health"
MEMORY_RECALL = f"{BASE}/memory/recall"
AUTH_REGISTER = f"{BASE}/auth/register"
AUTH_LOGIN = f"{BASE}/auth/login"

# Identity — set after login (name + password). Each user's memory is isolated
# under their username in Postgres / Elasticsearch / Neo4j.
USER_ID = ""
SESSION_ID = ""
HISTORY_WINDOW = 12

messages: list[dict] = []


def authenticate() -> None:
    """Prompt for register/login, then set USER_ID / SESSION_ID from the server."""
    global USER_ID, SESSION_ID
    print("=== Login ===  (new user? type 'register')")
    while True:
        mode = input("login / register: ").strip().lower() or "login"
        if mode not in ("login", "register"):
            print("  type 'login' or 'register'")
            continue
        username = input("username: ").strip()
        password = getpass.getpass("password: ")
        if not username or not password:
            print("  username and password required")
            continue
        url = AUTH_REGISTER if mode == "register" else AUTH_LOGIN
        try:
            r = requests.post(
                url, json={"username": username, "password": password}, timeout=30
            )
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] backend unreachable: {e}")
            continue
        if r.status_code == 200:
            data = r.json()
            USER_ID = data["user_id"]
            SESSION_ID = data["session_id"]
            who = "Registered" if data.get("new_user") else "Logged in"
            print(f"  {who} as {USER_ID}\n")
            return
        try:
            detail = r.json().get("detail")
        except Exception:
            detail = r.text
        print(f"  [{r.status_code}] {detail}")


def windowed_messages() -> list[dict]:
    """Last N turns; window must start with user (vLLM role alternate)."""
    if len(messages) <= HISTORY_WINDOW:
        out = list(messages)
    else:
        out = list(messages[-HISTORY_WINDOW:])
    while out and out[0].get("role") != "user":
        out.pop(0)
    while len(out) >= 2 and out[-1].get("role") == out[-2].get("role"):
        out.pop(-2)
    return out


def send_query(user_input: str) -> tuple[str, dict, dict | None]:
    messages.append({"role": "user", "content": user_input})

    # Minimal payload — no use_agent_memory, no client memory bag, no max_tokens
    payload = {
        "messages": windowed_messages(),
        "user_id": USER_ID,
        "session_id": SESSION_ID,
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
                elif etype == "tool_call":
                    name = event.get("name") or "?"
                    print(f"\n[tool → {name}]", end="", flush=True)
                elif etype == "tool_result":
                    name = event.get("name") or "?"
                    print(f" [ok:{name}]\nBot: ", end="", flush=True)
                    answer = ""
                elif etype == "final":
                    final_ans = event.get("answer") or ""
                    if final_ans and final_ans != answer:
                        if not answer.strip():
                            print(final_ans, end="", flush=True)
                        answer = final_ans
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

    messages.append({"role": "assistant", "content": answer})
    return answer, extracted, mem_status


def server_recall(query: str | None = None):
    r = requests.post(
        MEMORY_RECALL,
        json={
            "user_id": USER_ID,
            "session_id": SESSION_ID,
            "query": query,
            "recent_limit": 20,
            "search_limit": 10,
            "graph_limit": 40,
        },
        timeout=60,
    )
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
    authenticate()
    print("Chat stream (memory/tools = backend always on; model chooses tools)")
    print(f"  user_id    = {USER_ID}")
    print(f"  session_id = {SESSION_ID}")
    print("  health / recall / clear / wipe / exit\n")

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
        if low == "health":
            try:
                print(json.dumps(requests.get(MEMORY_HEALTH, timeout=30).json(), indent=2))
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
            try:
                server_clear_session()
                print("\n(Local + session clear; KG kept)\n")
            except Exception as e:
                print(f"\n(Local clear; server: {e})\n")
            continue
        if low == "wipe":
            messages.clear()
            try:
                server_clear_user()
                print("\n(Full user wipe)\n")
            except Exception as e:
                print(f"\n(Local clear; server: {e})\n")
            continue

        try:
            _answer, turn_facts, mem_status = send_query(user_input)
            print("--- extracted_facts (this turn) ---")
            print(json.dumps(turn_facts, indent=2, ensure_ascii=False))
            if mem_status:
                print("--- memory_status ---")
                print(json.dumps(mem_status, indent=2, ensure_ascii=False))
            print("----------------------------------\n")
        except requests.exceptions.RequestException as e:
            print(f"\n[ERROR] Backend: {e}\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()
