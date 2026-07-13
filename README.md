# Local LLM Stack + Agent Memory

Local Docker stack: **chat client → FastAPI backend → vLLM (GPU)** with **agent memory** (SQL + Elasticsearch + Neo4j) and optional **Langfuse** tracing.

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
| `postgres` | **5433**→5432 | Langfuse meta + DB `agent_memory` (chat truth) |
| `elasticsearch` | **9200** | Full-text + dense_vector field |
| `neo4j` | **7474** / **7687** | Knowledge graph |
| `graphxr` | **8080** | **3D GraphXR** UI (Neo4j pe) |
| `graphxr-mongo` | — | GraphXR metadata store |
| `memory-migrate` | one-shot | SQL schema |
| `elasticsearch-migrate` | one-shot | ES index |
| `neo4j-migrate` | one-shot | Neo4j constraints |
| `langfuse-web` | **3000** | Traces UI |
| `langfuse-worker` | **3030** | Redis → storage |
| Redis / ClickHouse / MinIO | **6379** / **8123** / **9090** | Langfuse stack |

---

## 2. Run

1. **Env**
   - `config/.env` (model path, GPU, Postgres, Neo4j auth, …)
   - `backend/vllm/.env` (BASE_URL, MODEL_NAME, Langfuse keys)
   - `backend/agent_memory/.env` (MEMORY_DATABASE_URL, ES, Neo4j)

2. **Start**
   ```powershell
   cd config
   docker compose up -d --build
   ```
   Order: Postgres/ES/Neo4j healthy → **3 migrates** → backend. vLLM model load alag se time lega.

3. **Check migrates**
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

| URL | Service |
|-----|---------|
| http://localhost:5000 | Backend |
| http://localhost:8000 | vLLM |
| http://localhost:3000 | Langfuse |
| http://localhost:9200 | Elasticsearch |
| http://localhost:7474 | Neo4j Browser |
| bolt://localhost:7687 | Neo4j Bolt |
| **http://localhost:8080** | **GraphXR 3D** (login below) |
| http://localhost:5000/memory/health | Memory health |

### GraphXR 3D (Neo4j)

Agent memory graph ko 3D me dekhne ke liye [GraphXR Lite](https://github.com/Kineviz/graphxr-lite) stack compose me hai.

```powershell
cd config
docker compose up -d neo4j graphxr-mongo graphxr
# pehli baar image pull bada ho sakta hai (linux/amd64)
```

| | |
|--|--|
| URL | http://localhost:8080 |
| Login | `config/.env` → `GRAPHXR_ADMIN_EMAIL` / `GRAPHXR_ADMIN_PASSWORD` |
| Neo4j (auto) | `config/.env` → `MEMORY_NEO4J_USER` / `MEMORY_NEO4J_PASSWORD` |

Passwords **compose.yml me hardcode nahi** — sirf `config/.env` (template: `config/.env.example`).  
Backend app Neo4j client: `backend/agent_memory/.env` (`MEMORY_NEO4J_URI` + same user/pass).

1. Chat se kuch entities/relations save karo (`python chat_client.py`)  
2. GraphXR kholo → project / Neo4j connection (defaults pehle se env se set)  
3. Query e.g. `MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 200` ya UI se load → **3D** explore  

Neo4j Browser 2D: http://localhost:7474 (same credentials).

Schema dubara apply (safe re-run):
```powershell
cd config
docker compose run --rm memory-migrate
docker compose run --rm elasticsearch-migrate
docker compose run --rm neo4j-migrate
```

---

## 3. One chat turn (agent memory)

```
You type message
        │
        ▼
chat_client → POST /chat/structured/stream
        │
        ▼
1. RECALL  (best-effort)
   · Neo4j: entities, facts, constraints, relations
   · Elasticsearch: related past (query = user text)
   · SQL: recent session (optional in block)
   · → system MEMORY inject
        │
        ▼
2. SINGLE LLM CALL (vLLM guided JSON)
   · stream answer tokens live (answer_delta)
   · then extracted_facts (final event)
   · shape:
     {
       "answer": "...",                 // complete reply first
       "extracted_facts": {
         "entities": [],
         "facts_about_user": [],
         "constraints": [],
         "relations": [                 // subject-predicate-object
           {"subject":"Rahul","predicate":"LIVES_IN","object":"Pune"}
         ]
       }
     }
   · facts/relations: latest user message only + code grounding filter
        │
        ▼
3. WRITE
   · SQL: messages append + turn_facts (entities, facts, constraints, relations)
   · Elasticsearch: message doc append
   · Neo4j: MERGE entities/facts + RELATES_TO edges
        │
        ▼
Client: live Bot print + merge local facts cache
```

**Still 1 LLM call** (cost) — recall/write stores are not extra LLM calls.

---

## 4. Agent memory design

| Store | What | Write style |
|-------|------|-------------|
| **SQL** (`agent_memory`) | Episodic transcript + per-turn fact snapshot | **append** |
| **Elasticsearch** | Searchable messages (BM25; dense_vector ready) | **append** docs |
| **Neo4j** | Semantic graph: entities, facts, **relations** | **MERGE / upsert** |

```
SQL (truth tape)     ←── every message
ES (find past)       ←── every message  
Neo4j (belief graph) ←── verified entities + relations
```

### Neo4j graph shape

```
(:User)-[:MENTIONED]->(:Entity)
(:User)-[:HAS_FACT]->(:UserFact)
(:User)-[:HAS_CONSTRAINT]->(:Constraint)
(:Entity)-[:RELATES_TO {predicate, user_id}]->(:Entity)
```

### Schema = migrate, not app boot

| Store | File | Compose job |
|-------|------|-------------|
| Postgres | `agent_memory/sql/schema.sql` | `memory-migrate` |
| Elasticsearch | `agent_memory/elasticsearch/index.json` | `elasticsearch-migrate` |
| Neo4j | `agent_memory/knowledge_graph/schema.cypher` | `neo4j-migrate` |

App only DML / index docs / MERGE — **no CREATE TABLE/index on request path**.

### Anti-hallucination (single-call, no 2× tokens)

- Prompt: extract **only from latest user message**, not from model answer  
- After LLM: **code filter** — entity/fact/relation subject+object must ground in user text; else drop  
- Empty arrays better than inventing  

### Streaming API

| Endpoint | Behavior |
|----------|----------|
| `POST /chat/structured` | Full JSON response |
| `POST /chat/structured/stream` | SSE: `answer_delta` live → `final` (facts + memory_status) |
| `POST /memory/write` · `/recall` · `/health` | Direct memory API |
| `DELETE /memory/session` · `/memory/user/{id}` | Clear session / wipe user |

---

## 5. Backend layout

```
main.py
  ├── vllm router
  │     /health
  │     /chat
  │     /chat/structured
  │     /chat/structured/stream   ★ live + facts
  │     /chat/stream
  └── agent_memory router
        /memory/*
```

```
backend/
├── vllm/                 # LLM client, guided JSON, stream parse
└── agent_memory/
    ├── sql/              # Postgres only
    ├── elasticsearch/    # ES only
    ├── knowledge_graph/  # Neo4j only (store.py)
    ├── service.py        # orchestrate 3 stores
    ├── bridge.py         # chat recall/write glue
    ├── routes.py
    └── schemas.py
```

---

## 6. Langfuse (side path)

```
LLM call → langfuse SDK → langfuse-web:3000
                              → Redis queue
                              → langfuse-worker:3030
                              → ClickHouse / Postgres / MinIO
```

Langfuse down → chat ab bhi chal sakta hai; traces miss ho sakte hain.  
Langfuse Postgres DB ≠ product chat DB name: product uses DB **`agent_memory`** (same Postgres server).

---

## 7. Project structure

```
rag/
├── config/
│   ├── .env / .env.example
│   ├── docker-compose.yml      # vLLM, backend, ES, Neo4j, GraphXR, Langfuse
│   └── Dockerfile
│
├── backend/
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   ├── vllm/
│   │   ├── client.py           # structured + stream + grounding filter
│   │   ├── routes.py
│   │   ├── schemas.py          # answer + entities/facts/constraints/relations
│   │   └── .env
│   └── agent_memory/
│       ├── .env / .env.example
│       ├── sql/schema.sql + init_db.sh
│       ├── elasticsearch/index.json + init_index.sh
│       ├── knowledge_graph/schema.cypher + store.py
│       ├── service.py / bridge.py
│       └── routes.py / schemas.py
│
├── chat_client.py              # live SSE client
└── README.md
```

---

## 8. Cloud note

Compose migrates = local shortcut. Cloud pe same idea:

1. Infra up (RDS / ES / Neo4j)  
2. **One-shot migrate job** (schema files from repo)  
3. Deploy app (no DDL in app)  

Files (`schema.sql`, `index.json`, `schema.cypher`) reuse; runner = CI/K8s Job, not necessarily compose.
