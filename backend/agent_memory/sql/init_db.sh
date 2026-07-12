#!/usr/bin/env bash
# One-shot: create DB agent_memory (if missing) + apply schema.sql
# Used by docker-compose service `memory-migrate`.
set -euo pipefail

PGHOST="${PGHOST:-postgres}"
PGPORT="${PGPORT:-5432}"
PGUSER="${POSTGRES_USER:-postgresuser}"
PGPASSWORD="${POSTGRES_PASSWORD:-postgres1938}"
export PGPASSWORD

MEMORY_DB="${MEMORY_DB_NAME:-agent_memory}"
SCHEMA_FILE="${SCHEMA_FILE:-/schema.sql}"

echo "[memory-migrate] waiting for postgres at ${PGHOST}:${PGPORT}..."
until psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres -c '\q' 2>/dev/null; do
  sleep 1
done
echo "[memory-migrate] postgres is up"

exists="$(psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres -tAc \
  "SELECT 1 FROM pg_database WHERE datname = '${MEMORY_DB}'")"
if [[ "$exists" != "1" ]]; then
  echo "[memory-migrate] creating database ${MEMORY_DB}"
  psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d postgres \
    -v ON_ERROR_STOP=1 \
    -c "CREATE DATABASE \"${MEMORY_DB}\""
else
  echo "[memory-migrate] database ${MEMORY_DB} already exists"
fi

echo "[memory-migrate] applying ${SCHEMA_FILE}"
psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$MEMORY_DB" \
  -v ON_ERROR_STOP=1 \
  -f "$SCHEMA_FILE"

echo "[memory-migrate] done"
