"""Password hashing + session token generation -- pure functions, no SQL/HTTP.

Kept separate from db.py (pure SQL I/O) and api.py (HTTP routing) so the one place that
knows about bcrypt/token format is this file -- swapping either later is a one-file change.
"""

from __future__ import annotations

import secrets

import bcrypt


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def generate_session_token() -> str:
    return secrets.token_urlsafe(32)
