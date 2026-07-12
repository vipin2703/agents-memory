"""
Knowledge graph — Neo4j ONLY.

Schema: knowledge_graph/schema.cypher via docker `neo4j-migrate` (not app).
"""

from .store import GraphStore

__all__ = ["GraphStore"]
