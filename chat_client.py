# """
# chat_client.py -- Terminal se backend ke /chat/stream endpoint pe query bhejta
# rahega jab tak tu 'exit', 'quit', ya Ctrl+C se rok na de.

# Ye backend se SSE (Server-Sent Events) format me streaming response leta hai
# aur usse live, word-by-word terminal me print karta hai.

# FIX: agar request fail ho jaye (jaise backend 400/502 de), to us user
# message ko history se WAPAS HATA dete hain -- warna agli baar do "user"
# messages laggatar history me reh jaate hain aur LLM "roles must alternate"
# error deta hai.
# """

# import requests

# BACKEND_URL = "http://localhost:5000/chat/stream"  # apne backend ka URL/port yaha set karo

# # Conversation history yaha store hoga taaki backend ko context mile
# messages = []


# def send_query_stream(user_input: str) -> str:

#     # print(user_input)
#     print(messages)
#     messages.append({"role": "user", "content": user_input})
     
#     payload = {
#         "messages": messages,
#     }
#     # print(payload)
#     full_response = ""

#     try:
#         with requests.post(BACKEND_URL, json=payload, stream=True, timeout=120) as response:
#             response.raise_for_status()

#             for line in response.iter_lines(decode_unicode=True):
#                 if not line:
#                     continue  # SSE me blank lines aati hain, skip karo

#                 # SSE format: "data: <content>"
#                 if line.startswith("data: "):
#                     chunk = line[len("data: "):]

#                     if chunk == "[DONE]":
#                         break

#                     if chunk.startswith("[ERROR]"):
#                         raise RuntimeError(chunk)

#                     print(chunk, end="", flush=True)
#                     full_response += chunk
#     except Exception:

#         messages.pop()
#         raise

#     print()  # stream khatam hone ke baad newline
#     messages.append({"role": "assistant", "content": full_response})
#     return full_response


# def main():
#     print("Chat shuru ho gaya. Rokne ke liye 'exit' ya 'quit' likho (ya Ctrl+C dabao).\n")

#     while True:
#         try:
#             user_input = input("You: ").strip()
#         except (KeyboardInterrupt, EOFError):
#             print("\nBand ho raha hai. Bye!")
#             break

#         if not user_input:
#             continue

#         if user_input.lower() in ("exit", "quit"):
#             print("Bye!")
#             break

#         try:
#             print("Bot: ", end="", flush=True)
#             send_query_stream(user_input)
#             print()
#         except requests.exceptions.RequestException as e:
#             print(f"\n[ERROR] Backend se connect nahi ho paya: {e}\n")
#         except RuntimeError as e:
#             print(f"\n[ERROR] {e}\n")


# if __name__ == "__main__":
#     main()








"""
chat_client.py -- Terminal se backend ke /chat/structured endpoint pe query
bhejta rahega jab tak tu 'exit', 'quit', ya Ctrl+C se rok na de.

Ye backend se STRUCTURED (non-streaming) response leta hai:
  {
    "answer": "...",              <- user ko dikhta hai
    "extracted_facts": {...}       <- jo bhi categories backend bheje,
                                       dynamically merge/persist hoti hain
  }

Categories hardcode NAHI ki gayi hain -- backend (schemas.py) jo bhi keys
extracted_facts me bheje, unhi ko is file me consume kiya jaata hai. Schema
backend me badle (category add/remove ho) to is file ko touch nahi karna
padta -- backend hi decide karta hai "kaisa data chahiye".

FIX: agar request fail ho jaye (jaise backend 400/502 de), to us user 
message ko history se WAPAS HATA dete hain -- warna agli baar do "user"
messages laggatar history me reh jaate hain aur LLM "roles must alternate"
error deta hai.
"""

import requests

BACKEND_URL = "http://localhost:5000/chat/structured"  # apne backend ka URL/port yaha set karo

# Conversation history yaha store hoga taaki backend ko context mile
messages = []

# Persistent facts-store -- backend se jo bhi category-keys aayengi,
# unke liye dynamically list ban jaayegi
facts = {}


def merge_facts(new_facts: dict):
    """Naye turn ke facts ko persistent facts-store me merge karta hai (dedupe ke saath).
    Categories backend se hi aati hain -- yaha koi fixed list nahi."""
    for category, new_items in new_facts.items():
        if category not in facts:
            facts[category] = []
        for item in new_items:
            if item and item not in facts[category]:
                facts[category].append(item)


def print_facts_summary():
    """Debug ke liye -- abhi tak accumulate hue saare facts dikhata hai."""
    non_empty = {k: v for k, v in facts.items() if v}
    if not non_empty:
        print("\n(Abhi tak koi facts extract nahi hue)\n")
        return
    print("\n--- Extracted facts (persistent) ---")
    for category, items in non_empty.items():
        print(f"  {category}: {items}")
    print("-------------------------------------\n")


def send_query_structured(user_input: str) -> str:
    """
    Backend ko structured request bhejta hai. 'answer' return karta hai
    (user ko dikhane ke liye), aur 'extracted_facts' ko persistent
    facts-store me merge kar deta hai.
    """
    messages.append({"role": "user", "content": user_input})

    payload = {
        "messages": messages,
    }

    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=120)
        if response.status_code != 200:
            # Backend ka actual error-detail dikhao, sirf generic HTTP status nahi
            try:
                error_body = response.json()
            except Exception:
                error_body = response.text
            print(f"\n[BACKEND ERROR {response.status_code}]: {error_body}\n")
        response.raise_for_status()
        data = response.json()
    except Exception:
        # Request fail hui -- abhi jo user message add kiya tha wapas hata do,
        # warna agli baar history me 2 consecutive "user" messages reh jaayenge
        # aur LLM "roles must alternate" error dega.
        messages.pop()
        raise

    answer = data["answer"]
    extracted = data.get("extracted_facts", {})

    merge_facts(extracted)
    messages.append({"role": "assistant", "content": answer})

    return answer


def main():
    print("Chat shuru ho gaya. Rokne ke liye 'exit' ya 'quit' likho (ya Ctrl+C dabao).")
    print("'facts' likhkar abhi tak extract hue saare facts dekh sakta hai.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBand ho raha hai. Bye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break

        if user_input.lower() == "facts":
            print_facts_summary()
            continue

        try:
            answer = send_query_structured(user_input)
            print(f"Bot: {answer}\n")
        except requests.exceptions.RequestException as e:
            print(f"\n[ERROR] Backend se connect nahi ho paya: {e}\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()