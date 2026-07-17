"""Postgres access (direct connection string) -- job storage + our own auth tables.

Two separate concerns, deliberately kept apart:
  * The job SQL helpers -- ONLY THE BACKEND ever holds `DATABASE_URL` or talks to Postgres
    directly. The frontend never gets a connection string, only a session token from our
    own `/auth/*` routes; every read/write of job data goes through this API instead.
    User-scoping is an explicit `user_id` filter in each query (no RLS -- see schema.sql
    for why).
  * The auth SQL helpers -- we run our own auth (no Supabase Auth): `users` holds
    email + bcrypt password hash (hashing itself lives in app/auth.py, kept out of this
    file since it's pure business logic, not SQL), `sessions` holds opaque tokens.
    `get_current_user` is "reading a table" -- look up the token, join to its user, check
    expiry -- not JWT verification.

Connections are opened per call, not cached/pooled across the process -- RQ's default
worker forks a subprocess per job, and a DB connection's live socket shared across a
fork() is unsafe. Per-call connections sidestep that; this system's request volume doesn't
need pooling.

Nothing here ever touches raw audio -- only job metadata, structured results, and auth data.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from fastapi import Header, HTTPException

from app.config import DATABASE_URL


@contextmanager
def _cursor():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not configured")
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        with conn:
            with conn.cursor() as cur:
                yield cur
    finally:
        conn.close()


# --- Jobs ----------------------------------------------------------------------------


def insert_job(batch_id: str, user_id: str, filename: str, status: str, error: str | None = None) -> str:
    with _cursor() as cur:
        cur.execute(
            """
            insert into jobs (batch_id, user_id, filename, status, error)
            values (%s, %s, %s, %s, %s)
            returning id
            """,
            (batch_id, user_id, filename, status, error),
        )
        return str(cur.fetchone()["id"])


def update_job_rq_id(job_row_id: str, rq_job_id: str) -> None:
    with _cursor() as cur:
        cur.execute("update jobs set rq_job_id = %s where id = %s", (rq_job_id, job_row_id))


def update_job_stage(job_row_id: str, stage: str) -> None:
    with _cursor() as cur:
        cur.execute(
            "update jobs set status = 'started', stage = %s where id = %s",
            (stage, job_row_id),
        )


def update_job_success(job_row_id: str, result: dict) -> None:
    with _cursor() as cur:
        cur.execute(
            "update jobs set status = 'finished', stage = 'done', result = %s where id = %s",
            (json.dumps(result), job_row_id),
        )


def update_job_failure(job_row_id: str, error: str) -> None:
    with _cursor() as cur:
        cur.execute(
            "update jobs set status = 'failed', error = %s where id = %s",
            (error, job_row_id),
        )


def list_batches(user_id: str) -> list[dict]:
    """One row per batch (for the dashboard's history view), most recent first."""
    with _cursor() as cur:
        cur.execute(
            """
            select
                batch_id,
                min(created_at) as created_at,
                count(*) as file_count,
                count(*) filter (where status = 'finished') as finished_count,
                count(*) filter (where status = 'failed') as failed_count,
                count(*) filter (where status in ('queued', 'started')) as pending_count
            from jobs
            where user_id = %s
            group by batch_id
            order by min(created_at) desc
            """,
            (user_id,),
        )
        return list(cur.fetchall())


def list_batch_jobs(batch_id: str, user_id: str) -> list[dict]:
    with _cursor() as cur:
        cur.execute(
            "select * from jobs where batch_id = %s and user_id = %s order by created_at",
            (batch_id, user_id),
        )
        return list(cur.fetchall())


def get_job(job_id: str, user_id: str) -> dict | None:
    with _cursor() as cur:
        cur.execute("select * from jobs where id = %s and user_id = %s", (job_id, user_id))
        return cur.fetchone()


# --- Auth (our own users/sessions tables) ---------------------------------------------


def create_user(email: str, password_hash: str) -> str:
    """Raises psycopg2.errors.UniqueViolation if the email is already registered --
    left to propagate; api.py turns it into a 409."""
    with _cursor() as cur:
        cur.execute(
            "insert into users (email, password_hash) values (%s, %s) returning id",
            (email, password_hash),
        )
        return str(cur.fetchone()["id"])


def get_user_by_email(email: str) -> dict | None:
    with _cursor() as cur:
        cur.execute("select * from users where email = %s", (email,))
        return cur.fetchone()


def create_session(user_id: str, token: str, expires_at) -> None:
    with _cursor() as cur:
        cur.execute(
            "insert into sessions (token, user_id, expires_at) values (%s, %s, %s)",
            (token, user_id, expires_at),
        )


def get_user_by_session_token(token: str) -> dict | None:
    with _cursor() as cur:
        cur.execute(
            """
            select users.id, users.email
            from sessions
            join users on users.id = sessions.user_id
            where sessions.token = %s and sessions.expires_at > now()
            """,
            (token,),
        )
        return cur.fetchone()


def delete_session(token: str) -> None:
    """Idempotent -- deleting an already-gone token is a no-op, not an error."""
    with _cursor() as cur:
        cur.execute("delete from sessions where token = %s", (token,))


def get_current_user(authorization: str = Header(...)) -> dict:
    """FastAPI dependency: look up the bearer token in `sessions`, return `{id, email}`.

    A plain table lookup, not JWT verification -- expired/missing tokens both come back
    as `None` from get_user_by_session_token and are treated identically (401).
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    user = get_user_by_session_token(authorization.removeprefix("Bearer "))
    if not user:
        raise HTTPException(401, "invalid or expired session")
    return {"id": str(user["id"]), "email": user["email"]}
