#!/usr/bin/env bash
# One-shot: create Elasticsearch index (BM25 + dense_vector) — NOT from app code.
set -euo pipefail

ES_URL="${MEMORY_ELASTICSEARCH_URL:-http://elasticsearch:9200}"
INDEX="${MEMORY_ELASTICSEARCH_INDEX:-agent_memory_messages}"
SCHEMA_FILE="${SCHEMA_FILE:-/index.json}"

echo "[elasticsearch-migrate] waiting for ${ES_URL}..."
until curl -sf "${ES_URL}" >/dev/null 2>&1; do
  sleep 2
done
echo "[elasticsearch-migrate] elasticsearch is up"

code="$(curl -s -o /dev/null -w '%{http_code}' "${ES_URL}/${INDEX}")"
if [[ "$code" == "200" ]]; then
  echo "[elasticsearch-migrate] index ${INDEX} already exists"
  exit 0
fi

echo "[elasticsearch-migrate] creating index ${INDEX}"
resp="$(curl -s -w '\n%{http_code}' -X PUT "${ES_URL}/${INDEX}" \
  -H 'Content-Type: application/json' \
  --data-binary @"${SCHEMA_FILE}")"
body="$(echo "$resp" | sed '$d')"
http="$(echo "$resp" | tail -n1)"

if [[ "$http" == "200" || "$http" == "201" ]]; then
  echo "[elasticsearch-migrate] index created: ${INDEX}"
  exit 0
fi

echo "[elasticsearch-migrate] FAILED HTTP ${http}: ${body}"
exit 1
