-- Agent memory SQL schema (Postgres).
-- Apply ONCE via memory-migrate service / manual psql — NOT from app code.
-- Safe to re-run: IF NOT EXISTS throughout.

-- Registered users. user_id everywhere (sessions/messages/graph/ES) = username.
-- password_hash format: "pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>".
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session_ts
    ON messages (session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_messages_user_ts
    ON messages (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS turn_facts (
    id BIGSERIAL PRIMARY KEY,
    message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    facts_about_user JSONB NOT NULL DEFAULT '[]'::jsonb,
    constraints JSONB NOT NULL DEFAULT '[]'::jsonb,
    relations JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- existing DBs (re-run migrate safe)
ALTER TABLE turn_facts
    ADD COLUMN IF NOT EXISTS relations JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_turn_facts_user
    ON turn_facts (user_id, created_at DESC);
