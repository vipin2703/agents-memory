"""
Elasticsearch store — BM25 + dense_vector (hybrid-ready).

Index: elasticsearch/index.json via docker `elasticsearch-migrate` (not app).
"""

from .store import SearchStore

__all__ = ["SearchStore"]
