"""
chat_client.py -- Terminal se backend ke /chat/stream endpoint pe query bhejta
rahega jab tak tu 'exit', 'quit', ya Ctrl+C se rok na de.

Ye backend se SSE (Server-Sent Events) format me streaming response leta hai
aur usse live, word-by-word terminal me print karta hai.

FIX: agar request fail ho jaye (jaise backend 400/502 de), to us user
message ko history se WAPAS HATA dete hain -- warna agli baar do "user"
messages laggatar history me reh jaate hain aur LLM "roles must alternate"
error deta hai.
"""

import requests

BACKEND_URL = "http://localhost:5000/chat/stream"  # apne backend ka URL/port yaha set karo

# Conversation history yaha store hoga taaki backend ko context mile
messages = []


def send_query_stream(user_input: str) -> str:

    # print(user_input)
    print(messages)
    messages.append({"role": "user", "content": user_input})
     
    payload = {
        "messages": messages,
    }
    # print(payload)
    full_response = ""

    try:
        with requests.post(BACKEND_URL, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue  # SSE me blank lines aati hain, skip karo

                # SSE format: "data: <content>"
                if line.startswith("data: "):
                    chunk = line[len("data: "):]

                    if chunk == "[DONE]":
                        break

                    if chunk.startswith("[ERROR]"):
                        raise RuntimeError(chunk)

                    print(chunk, end="", flush=True)
                    full_response += chunk
    except Exception:

        messages.pop()
        raise

    print()  # stream khatam hone ke baad newline
    messages.append({"role": "assistant", "content": full_response})
    return full_response


def main():
    print("Chat shuru ho gaya. Rokne ke liye 'exit' ya 'quit' likho (ya Ctrl+C dabao).\n")

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

        try:
            print("Bot: ", end="", flush=True)
            send_query_stream(user_input)
            print()
        except requests.exceptions.RequestException as e:
            print(f"\n[ERROR] Backend se connect nahi ho paya: {e}\n")
        except RuntimeError as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()













