# Local LLM Stack + Agent Memory

Local Docker stack: **chat client → FastAPI backend → LLM** with **agent memory** (SQL + Elasticsearch + Neo4j), **GraphXR 3D**, and optional **Langfuse** tracing.

Inference is pluggable — the backend speaks the OpenAI API, so the LLM can be a **local vLLM (GPU)** or **any hosted OpenAI-compatible API** (Gemini, OpenRouter, …). See [§2 Inference backends](#2-inference-backends--two-interchangeable-methods).

---

## 1. Architecture (big picture)

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│  HOST                                                                              │
│                                                                                    │
│  chat_client.py                                                                    │
│       │  POST :5000/chat/structured/stream  (live answer + facts)                  │
│       ▼                                                                            │
│  backend (llm_serve :5000)                                                         │
│       │  1) recall memory                                                          │
│       │  2) 1× structured LLM call (answer + extracted_facts)                      │
│       │  3) write teeno stores                                                     │
│       ├──────────────────┬───────────────────┬──────────────────┐                  │
│       ▼                  ▼                   ▼                  ▼                  │
│  vLLM :8000      Postgres           Elasticsearch :9200    Neo4j :7687/:7474       │
│  guided JSON     agent_memory DB    BM25 + dense_vector    entities + RELATES_TO   │
│                  (transcript)       (search index)         (knowledge graph)       │
│                                                         │                          │
│                                                         ▼                          │
│                                              GraphXR :8080 (3D UI)                 │
│                                              (+ mongo only for GraphXR meta)       │
│       │                                                                            │
│       │ traces (async)                                                             │
│       ▼                                                                            │
│  Langfuse :3000 → Redis :6379 → worker :3030 → ClickHouse / Postgres / MinIO        │
└────────────────────────────────────────────────────────────────────────────────────┘
```

| Service | Host port | Job |
|---------|-----------|-----|
| `chat_client.py` | — | Live stream chat + local facts cache |
| `backend` (`llm_serve`) | **5000** | FastAPI: chat, structured stream, memory API |
| `vllm-server` | **8000** | Model + guided JSON (xgrammar) |
| `postgres` | **5433**→5432 | Langfuse meta + DB `agent_memory` |
| `elasticsearch` | **9200** | Full-text + dense_vector field |
| `neo4j` | **7474** / **7687** | Knowledge graph |
| `graphxr` | **8080** | 3D GraphXR UI (reads Neo4j) |
| `graphxr-mongo` | — | GraphXR app metadata only (not product graph) |
| `memory-migrate` | one-shot | SQL schema from `schema.sql` |
| `elasticsearch-migrate` | one-shot | ES index from `index.json` |
| `neo4j-migrate` | one-shot | Neo4j constraints from `schema.cypher` |
| `langfuse-web` | **3000** | Traces UI |
| `langfuse-worker` | **3030** | Redis → storage |
| Redis / ClickHouse / MinIO | **6379** / **8123** / **9090** | Langfuse stack |

---

## 2. Inference backends — two interchangeable methods

The backend talks to the model over the **OpenAI Chat Completions API**, so the
**same code** runs against a local model or a hosted API. Switch by editing only
`backend/vllm/.env` (`BASE_URL`, `API_KEY`, `MODEL_NAME`) — nothing else changes.

### Method A — Local vLLM (GPU) + guided JSON  *(original)*

Weights loaded locally by `vllm-server`; structured output enforced server-side by
**xgrammar guided JSON** → guaranteed shape.

```env
# backend/vllm/.env
BASE_URL=http://vllm-server:8000/v1
API_KEY=not-needed
MODEL_NAME=phi-4-mini            # = SERVED_MODEL_NAME in config/.env
```

- Needs a GPU + the `vllm-server` container.
- Best JSON reliability, fully offline, no per-token cost.

### Method B — Hosted OpenAI-compatible API  *(added)*

Point the same backend at any hosted provider — **no GPU, no `vllm-server`**.

```env
# backend/vllm/.env — pick ONE provider

# Google Gemini (OpenAI-compat endpoint)
BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
API_KEY=<your-gemini-key>
MODEL_NAME=gemini-1.5-flash

# OpenRouter (many models incl. free tiers)
BASE_URL=https://openrouter.ai/api/v1
API_KEY=<your-openrouter-key>
MODEL_NAME=nvidia/nemotron-3-ultra-550b-a55b:free
```

- No server-side grammar available, so shape is enforced by the **strict system
  prompt + a lenient parser** (`backend/vllm/client.py`).
- **Streaming vs `json_object`** — toggle `JSON_OBJECT_MODE` in `client.py`:
  - `False` *(default)* → live **token-by-token streaming**. Many providers
    **buffer the whole reply** to validate `response_format=json_object`, which
    kills streaming — so we skip it and rely on the prompt.
  - `True` → hard JSON guarantee, but the answer arrives **all at once** (no stream).
- A **hybrid streamer** shows tokens live whether the model emits a JSON object
  (answer field extracted live) or plain prose.

> **Switch method:** edit `backend/vllm/.env`, then recreate the backend:
> `cd config && docker compose up -d --force-recreate --no-deps client`

---

## 3. Environment files (secrets yahan — compose me hardcode nahi)

Docker Compose **`config/`** se chalta hai → variable substitution + `env_file` ke liye **`config/.env`**.

| File | Kya rakho |
|------|-----------|
| **`config/.env`** | Model path, GPU, Postgres, **Neo4j auth**, **GraphXR login**, Langfuse stack secrets |
| **`config/.env.example`** | Template (repo me; copy → `.env`) |
| **`backend/vllm/.env`** | `BASE_URL`, `MODEL_NAME`, Langfuse API keys (backend → vLLM / Langfuse) |
| **`backend/agent_memory/.env`** | Backend app → stores: `MEMORY_DATABASE_URL`, ES URL, **Neo4j URI + same user/pass** |

### Key vars (values **README me nahi** — apne `.env` me rakho)

| Where | Variables (names only) |
|-------|------------------------|
| `config/.env` | `MEMORY_NEO4J_USER`, `MEMORY_NEO4J_PASSWORD`, `GRAPHXR_ADMIN_EMAIL`, `GRAPHXR_ADMIN_PASSWORD`, Postgres, model path, Langfuse stack |
| `backend/agent_memory/.env` | `MEMORY_DATABASE_URL`, `MEMORY_ELASTICSEARCH_*`, `MEMORY_NEO4J_URI`, `MEMORY_NEO4J_USER`, `MEMORY_NEO4J_PASSWORD` |
| `backend/vllm/.env` | `BASE_URL`, `MODEL_NAME`, `LANGFUSE_*` |

Templates: `*.env.example` — wahan se copy karke **apni** values bharo.  
**Rule:** passwords / emails / secrets **kabhi README ya compose me mat commit karo** — sirf `.env` (gitignored ideally).

---

## 4. Run

1. **Env copy / fill**
   ```text
   config/.env.example          →  config/.env
   backend/vllm/.env.example    →  backend/vllm/.env
   backend/agent_memory/.env.example → backend/agent_memory/.env
   ```
   Model path + Neo4j/GraphXR/Postgres passwords set karo.  
   Neo4j user/pass **dono** `.env` files me match hone chahiye (`config` + `agent_memory`).

2. **Start**
   ```powershell
   cd config
   docker compose up -d --build
   ```
   Order: Postgres/ES/Neo4j healthy → **3 migrates** → backend (+ GraphXR). vLLM model load alag time lega.

3. **Migrates check**
   ```powershell
   docker compose logs memory-migrate elasticsearch-migrate neo4j-migrate
   ```

4. **Health**
   - http://localhost:5000/health  
   - http://localhost:5000/memory/health  

5. **Chat**
   ```powershell
   cd ..
   python chat_client.py
   ```

| Command | Action |
|---------|--------|
| (type message) | live answer stream + facts/relations |
| `facts` | local facts cache |
| `health` | memory stores |
| `recall` / `recall <q>` | server memory block |
| `clear` | local + session transcript (KG kept) |
| `wipe` | full user wipe (SQL+ES+KG) |
| `exit` | quit |

### URLs

| URL | Service |
|-----|---------|
| http://localhost:5000 | Backend |
| http://localhost:8000 | vLLM |
| http://localhost:3000 | Langfuse |
| http://localhost:9200 | Elasticsearch |
| http://localhost:7474 | Neo4j Browser (2D) |
| bolt://localhost:7687 | Neo4j Bolt |
| **http://localhost:8080** | **GraphXR 3D** |
| http://localhost:5000/memory/health | Memory health |

Schema re-apply (safe):
```powershell
cd config
docker compose run --rm memory-migrate
docker compose run --rm elasticsearch-migrate
docker compose run --rm neo4j-migrate
```

---

## 5. GraphXR 3D (Neo4j)

[GraphXR Lite](https://github.com/Kineviz/graphxr-lite) — self-hosted 3D UI over **same** agent-memory Neo4j.

```powershell
cd config
docker compose up -d neo4j graphxr-mongo graphxr
```

| Item | Value |
|------|--------|
| UI | http://localhost:8080 |
| Login | values from `config/.env` (`GRAPHXR_ADMIN_*`) — not listed here |
| Neo4j | service host `neo4j`; auth from `config/.env` (`MEMORY_NEO4J_*`) |

**Mongo (`graphxr-mongo`)** = sirf GraphXR app meta. **Product graph = Neo4j only.**

Flow:
1. Chat se entities/relations save  
2. GraphXR UI → login with your `.env` values  
3. Load graph → 3D explore  

2D: http://localhost:7474 (same Neo4j credentials from `.env`).

---

## 6. One chat turn (agent memory)

```
You type message
        │
        ▼
chat_client → POST /chat/structured/stream
        │
        ▼
1. RECALL  (best-effort)
   · Neo4j: entities, facts, constraints, relations
   · Elasticsearch: related past
   · → system MEMORY inject
        │
        ▼
2. SINGLE LLM CALL (vLLM guided JSON)
   · live answer_delta stream
   · extracted_facts: entities, facts_about_user, constraints, relations
   · facts/relations: latest user message only + code grounding filter
        │
        ▼
3. WRITE
   · SQL: messages + turn_facts (incl. relations JSON)
   · Elasticsearch: message doc
   · Neo4j: MERGE entities/facts + RELATES_TO
```

**1 LLM call** only — store I/O alag.

Debug (backend logs): final vLLM input/output print  
`backend/vllm/client.py` → variables `final_messages` / `result`  
```powershell
docker logs -f llm_serve
```

---

## 7. Agent memory design

| Store | What | Write |
|-------|------|--------|
| **SQL** `agent_memory` | Episodic transcript + turn fact snapshot | append |
| **Elasticsearch** | Searchable messages | append docs |
| **Neo4j** | Semantic graph + **relations** | MERGE |

```
(:User)-[:MENTIONED]->(:Entity)
(:User)-[:HAS_FACT]->(:UserFact)
(:User)-[:HAS_CONSTRAINT]->(:Constraint)
(:Entity)-[:RELATES_TO {predicate, user_id}]->(:Entity)
```

### Schema = migrate (not app boot)

| Store | File | Job |
|-------|------|-----|
| Postgres | `agent_memory/sql/schema.sql` | `memory-migrate` |
| Elasticsearch | `agent_memory/elasticsearch/index.json` | `elasticsearch-migrate` |
| Neo4j | `agent_memory/knowledge_graph/schema.cypher` | `neo4j-migrate` |

### Anti-hallucination (single-call)

- Extract only from **latest user message** (not model answer)  
- Code filter: ungrounded entity/fact/relation → drop  

### API

| Endpoint | Behavior |
|----------|----------|
| `POST /chat/structured` | Full JSON |
| `POST /chat/structured/stream` | SSE live answer + final facts |
| `POST /memory/write` · `/recall` · `/health` | Memory API |
| `DELETE /memory/session` · `/user/{id}` | Clear / wipe |

---

## 8. Backend layout

```
main.py
  ├── vllm/          /chat, /chat/structured, /chat/structured/stream
  └── agent_memory/  /memory/*
```

```
backend/
├── vllm/client.py          # final_messages → vLLM; stream; grounding
├── vllm/routes.py
├── vllm/.env
└── agent_memory/
    ├── sql/                  # Postgres only
    ├── elasticsearch/        # ES only
    ├── knowledge_graph/      # Neo4j only
    ├── service.py / bridge.py
    ├── routes.py / schemas.py
    └── .env                  # store connection strings
```

`.py` code change → volume mount + uvicorn reload (auto).  
Naye packages → `docker compose up -d --build client`.

---

## 9. Langfuse (side path)

```
LLM → Langfuse SDK → langfuse-web:3000 → Redis → worker:3030
                         → ClickHouse / Postgres / MinIO
```

Langfuse down → chat chal sakta hai.  
Langfuse DB ≠ product DB name: chat truth = Postgres DB **`agent_memory`**.

---

## 10. Project structure

```
rag/
├── config/
│   ├── .env / .env.example     # compose secrets: Neo4j, GraphXR, model, Langfuse stack
│   ├── docker-compose.yml      # no hardcoded passwords for Neo4j/GraphXR
│   └── Dockerfile
│
├── backend/
│   ├── vllm/.env               # vLLM + Langfuse keys
│   └── agent_memory/
│       ├── .env                # SQL/ES/Neo4j URLs for app
│       ├── sql/
│       ├── elasticsearch/
│       └── knowledge_graph/
│
├── chat_client.py
└── README.md
```

---

## 11. Cloud note

Local: compose + `config/.env` + migrate services.  
Cloud: same schema files; migrate = CI/K8s Job; secrets = vault/env (not compose hardcode); GraphXR optional separate deploy.
