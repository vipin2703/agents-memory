# Local LLM Stack + Agent Memory

Local Docker stack: **chat client → FastAPI backend → vLLM (GPU)** with **agent memory** (SQL + Elasticsearch + Neo4j), **GraphXR 3D**, and optional **Langfuse** tracing.

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

## 2. Environment files (secrets yahan — compose me hardcode nahi)

Docker Compose **`config/`** se chalta hai → variable substitution + `env_file` ke liye **`config/.env`**.

| File | Kya rakho |
|------|-----------|
| **`config/.env`** | Model path, GPU, Postgres, **Neo4j auth**, **GraphXR login**, Langfuse stack secrets |
| **`config/.env.example`** | Template (repo me; copy → `.env`) |
| **`backend/vllm/.env`** | `BASE_URL`, `MODEL_NAME`, Langfuse API keys (backend → vLLM / Langfuse) |
| **`backend/agent_memory/.env`** | Backend app → stores: `MEMORY_DATABASE_URL`, ES URL, **Neo4j URI + same user/pass** |

### `config/.env` — Neo4j + GraphXR (compose services)

```env
# Neo4j (service neo4j, neo4j-migrate, graphxr)
MEMORY_NEO4J_USER=neo4j
MEMORY_NEO4J_PASSWORD=agentmemory

# GraphXR Lite UI http://localhost:8080
GRAPHXR_ADMIN_EMAIL=graphxr@local.dev
GRAPHXR_ADMIN_PASSWORD=graphxr123456
```

### `backend/agent_memory/.env` — Python Neo4j client (same credentials)

```env
MEMORY_NEO4J_URI=bolt://neo4j:7687
MEMORY_NEO4J_USER=neo4j
MEMORY_NEO4J_PASSWORD=agentmemory
```

**Rule:** passwords / IDs **`docker-compose.yml` me mat likho** — sirf `.env` files.  
Compose sirf `${MEMORY_NEO4J_USER}`, `${GRAPHXR_ADMIN_PASSWORD}`, … expand karta hai.

---

## 3. Run

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

## 4. GraphXR 3D (Neo4j)

[GraphXR Lite](https://github.com/Kineviz/graphxr-lite) — self-hosted 3D UI over **same** agent-memory Neo4j.

```powershell
cd config
docker compose up -d neo4j graphxr-mongo graphxr
```

| Item | Value |
|------|--------|
| UI | http://localhost:8080 |
| Login | `config/.env` → `GRAPHXR_ADMIN_EMAIL` / `GRAPHXR_ADMIN_PASSWORD` |
| Neo4j link | compose env: host `neo4j`, user/pass from `MEMORY_NEO4J_*` |

**Mongo (`graphxr-mongo`)** = sirf GraphXR app meta (projects/UI). **Product graph data Neo4j me hai**, Mongo me nahi.

Flow:
1. Chat se entities/relations save (`python chat_client.py`)  
2. http://localhost:8080 → login (`.env` credentials)  
3. Graph load / Cypher → 3D explore  

2D: http://localhost:7474 (same `MEMORY_NEO4J_*`).

---

## 5. One chat turn (agent memory)

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

## 6. Agent memory design

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

## 7. Backend layout

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

## 8. Langfuse (side path)

```
LLM → Langfuse SDK → langfuse-web:3000 → Redis → worker:3030
                         → ClickHouse / Postgres / MinIO
```

Langfuse down → chat chal sakta hai.  
Langfuse DB ≠ product DB name: chat truth = Postgres DB **`agent_memory`**.

---

## 9. Project structure

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

## 10. Cloud note

Local: compose + `config/.env` + migrate services.  
Cloud: same schema files; migrate = CI/K8s Job; secrets = vault/env (not compose hardcode); GraphXR optional separate deploy.
