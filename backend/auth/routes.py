"""
Auth HTTP API — name + password login so each user's memory is isolated.

  POST /auth/register  {username, password} -> {user_id, session_id}
  POST /auth/login     {username, password} -> {user_id, session_id}

user_id == username everywhere (sessions/messages/graph/ES). The session is
stable per user (main-<username>) so a user's conversation continues across
logins. Passwords are hashed with stdlib pbkdf2_sha256 (no extra dependency).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent_memory.service import get_memory_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

_PBKDF2_ITERATIONS = 200_000


def _hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters_s, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    try:
        iters = int(iters_s)
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
    # constant-time compare
    return hmac.compare_digest(dk.hex(), hash_hex)


def _normalize_username(name: str) -> str:
    return (name or "").strip().lower()


def _session_for(username: str) -> str:
    return f"main-{username}"


class AuthRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=256)


class AuthResponse(BaseModel):
    user_id: str
    session_id: str
    new_user: bool = False


@router.post("/register", response_model=AuthResponse)
async def register(req: AuthRequest) -> AuthResponse:
    username = _normalize_username(req.username)
    if not username:
        raise HTTPException(status_code=400, detail="username required")
    sql = get_memory_service().sql
    created = await sql.create_user(
        username=username, password_hash=_hash_password(req.password)
    )
    if not created:
        raise HTTPException(status_code=409, detail="username already taken")
    logger.info("user registered: %s", username)
    return AuthResponse(
        user_id=username, session_id=_session_for(username), new_user=True
    )


@router.post("/login", response_model=AuthResponse)
async def login(req: AuthRequest) -> AuthResponse:
    username = _normalize_username(req.username)
    sql = get_memory_service().sql
    user = await sql.get_user(username)
    if not user or not _verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid username or password")
    return AuthResponse(
        user_id=username, session_id=_session_for(username), new_user=False
    )
