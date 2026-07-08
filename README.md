# Local RAG System — vLLM Backend + Observability

A local, Docker-based LLM serving stack built around vLLM, with a FastAPI backend proxy and Langfuse observability. This README documents the architecture and design decisions made so far.

---

## 1. Project Structure

```
rag/
├── config/
│   ├── .env                    # vLLM server config (model path, GPU settings, etc.)
│   └── docker-compose.yml      # Orchestrates vllm-server, backend, and Langfuse stack
│
├── backend/
│   ├── Dockerfile
│   ├── main.py                 # FastAPI app init + router registration only
│   ├── Dockerfile
│   ├── requirements.txt
│   └── vllm/
│       ├── .env                 # BASE_URL, MODEL_NAME, Langfuse keys (container-scoped)
│       ├── client.py            # LLM call logic (business logic layer)
│       ├── routes.py            # HTTP endpoints (/health, /chat, /chat/stream)
│       └── schemas.py           # Pydantic request/response models
│
├── chat_client.py               # Standalone terminal client for testing the backend
└── client_/                     # Separate standalone script (direct vLLM/Azure client)
    ├── client.py

```

### Design principle: feature-based (vertical slice) architecture

Instead of grouping files by technical layer (all routes together, all schemas together), each service owns a self-contained folder with its own `client.py`, `routes.py`, and `schemas.py`. This was a deliberate choice for a project that will eventually have multiple heterogeneous services (vLLM, PostgreSQL, Redis, vector DB, etc.) — when working on one service, everything related to it lives in one place instead of being scattered across `routers/`, `schemas/`, and `services/` folders.

`main.py` stays minimal — its only job is to create the FastAPI app and register each service's router:

```python
from vllm_service.routes import router as vllm_router
app.include_router(vllm_router)
```

Adding a new service (e.g. Postgres) later means creating `postgres_service/` with the same three files and adding two lines to `main.py` — nothing else changes.

---

## 2. Backend Service (`vllm`)

- **`client.py`** — owns the LLM connection and call logic (currently an OpenAI-compatible async client pointed at vLLM). and LANGFUSE SDK for Observability   
- **`routes.py`** — owns the HTTP layer only. Delegates all actual work to `client.py`.
- **`schemas.py`** — the single source of truth for request/response shapes and their default values (e.g. `temperature`, `max_tokens`).

### Async from the start

All endpoints and client functions are `async def`, and the LLM client uses `AsyncOpenAI` rather than the sync `OpenAI` client. This matters because FastAPI runs sync (`def`) endpoints in a limited thread pool — with many concurrent requests (e.g. thousands hitting the backend at once), sync blocking I/O calls would bottleneck at the thread pool size. Async endpoints combined with an async client let concurrent requests share the event loop without blocking each other during network wait time. This was verified with a controlled test: 20 concurrent 1-second calls completed in ~1 second total (not ~20 seconds), confirming true non-blocking concurrency.

---






## 3. Observability — Self-Hosted Langfuse

Langfuse was added to capture prompt/response pairs, token usage, latency, and cost for every LLM call, without adding manual logging code throughout the codebase.

### Why self-hosted, and why it's a 6-container stack

Langfuse v3's self-hosted deployment separates concerns across multiple specialized services:

| Service | Role |
|---|---|
| PostgreSQL | Metadata (projects, users, API keys, settings) |
| ClickHouse | Trace/log data store, optimized for high-volume analytical queries |
| Redis | Queue between the API layer and the background worker, so incoming traces don't block on being written to storage |
| MinIO | Object storage for large blobs (e.g. attached media) |
| `langfuse-web` | The UI and API surface |
| `langfuse-worker` | Background processor that drains the Redis queue into ClickHouse/Postgres |

### Why each service exists — Langfuse v3's internal design

| Service | Role | Why it's necessary (in Langfuse's design) |
|---|---|---|
| **PostgreSQL** | Metadata store — users, projects, API keys, settings | This is the "control plane" — any web app needs somewhere to store its config/auth data |
| **ClickHouse** | Stores the actual traces/logs — prompts, responses, tokens, latency | This is the "data plane" — Postgres becomes slow for traces once volume reaches millions of rows, so ClickHouse (an analytics-optimized database) is used instead |
| **Redis** | Queue + cache | When a trace comes in, it isn't written to Postgres/ClickHouse immediately — it's placed on a Redis queue first, then processed by a background worker, so the actual LLM request isn't slowed down |
| **MinIO** | Stores large blobs (e.g. if images/files are also traced) | S3-like object storage — used when multimodal (image) traces are sent |
| **`langfuse-web`** | UI + API that's accessed in the browser | Frontend/API layer |
| **`langfuse-worker`** | Background processing (drains the Redis queue and writes into ClickHouse/Postgres) | Async processing engine |

This is heavier than a small project strictly needs, but it's the officially supported self-hosted architecture, and was chosen deliberately over lighter alternatives (Langfuse Cloud, a Postgres-only legacy version, or a custom logging solution) to match production-realistic tooling.

All Langfuse services were added into the same `docker-compose.yml` used by `vllm-server` and the backend, so they share the same Docker network and can reach each other by service name — following the same networking principle established in section 3.

### Integration approach

Langfuse's Python SDK offers a drop-in OpenAI client replacement, which was used instead of manual instrumentation:

```python
# before
from openai import AsyncOpenAI

# after
from langfuse.openai import AsyncOpenAI
```

Combined with an `@observe()` decorator on `run_chat` and `run_chat_stream`, this automatically captures the full trace (inputs, outputs, token usage, latency) for every LLM call with no other code changes required in `routes.py` or `schemas.py`.

This was verified to fail gracefully: when Langfuse's endpoint isn't reachable, the SDK logs a retry/export warning in the background but does not break or delay the actual chat request — confirmed by testing with no Langfuse server running.

---


## 4. Standalone Terminal Client (`chat_client.py`)

A simple script for manually testing the backend from a terminal, independent of any frontend.

- Sends user input to the backend's `/chat/stream` endpoint and keeps the conversation going in a loop until the user types `exit`/`quit` or presses Ctrl+C.
- Maintains conversation history locally so multi-turn context is preserved across turns.
- Parses the backend's Server-Sent Events (SSE) response format (`data: <chunk>\n\n`) and prints tokens live as they arrive, rather than waiting for the full response — this required switching from `response.json()` (which only works for a single JSON payload) to `requests.post(..., stream=True)` with line-by-line SSE parsing.


---



