#!/usr/bin/env bash
# One-shot: apply Neo4j constraints — NOT from app code.
set -euo pipefail

NEO4J_USER="${MEMORY_NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${MEMORY_NEO4J_PASSWORD:-agentmemory}"
NEO4J_HOST="${NEO4J_HOST:-neo4j}"
NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7687}"
SCHEMA_FILE="${SCHEMA_FILE:-/schema.cypher}"

echo "[neo4j-migrate] waiting for bolt://${NEO4J_HOST}:${NEO4J_BOLT_PORT}..."
until cypher-shell -a "bolt://${NEO4J_HOST}:${NEO4J_BOLT_PORT}" \
  -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" 'RETURN 1' >/dev/null 2>&1; do
  sleep 2
done
echo "[neo4j-migrate] neo4j is up"

echo "[neo4j-migrate] applying ${SCHEMA_FILE}"
cypher-shell -a "bolt://${NEO4J_HOST}:${NEO4J_BOLT_PORT}" \
  -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
  -f "$SCHEMA_FILE"

echo "[neo4j-migrate] done"
