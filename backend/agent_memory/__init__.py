"""
Agent memory — 3 alag backends:

  agent_memory.sql             → Postgres (source of truth)
  agent_memory.elasticsearch   → Elasticsearch (BM25 + dense_vector)
  agent_memory.knowledge_graph → Neo4j (distilled facts)

Orchestration: service.py + bridge.py (chat glue).
"""
