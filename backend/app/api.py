"""FastAPI app: the HTTP surface for the dashboard.

Signup/signin -> upload -> enqueue -> poll -> download. Every route requires a valid
session token (see app/auth.py, app/db.py) except /health and /auth/signup, /auth/signin.
Raw audio is saved to local disk (UPLOAD_DIR) and never touches Postgres; only job metadata
and structured results are.

Run with: uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import csv
import io
import json
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2.errors
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, Field

from app.auth import generate_session_token, hash_password, verify_password
from app.config import CORS_ORIGINS, MAX_ZIP_ENTRIES, SESSION_TTL_SEC, SUPPORTED_EXTENSIONS, UPLOAD_DIR
from app.db import (
    create_session,
    create_user,
    delete_session,
    get_current_user,
    get_job as db_get_job,
    get_user_by_email,
    insert_job,
    list_batch_jobs as db_list_batch_jobs,
    list_batches as db_list_batches,
    update_job_rq_id,
)
from app.queue import enqueue_audio_file

app = FastAPI(title="Voice Tone Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)


class SigninRequest(BaseModel):
    email: EmailStr
    password: str


def _new_session(user_id: str) -> str:
    token = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL_SEC)
    create_session(user_id, token, expires_at)
    return token


@app.post("/auth/signup")
def signup(body: SignupRequest) -> dict:
    # No email verification, no password complexity beyond min_length above -- acceptable
    # for a handful of known internal trial users, not a public signup surface.
    try:
        user_id = create_user(body.email, hash_password(body.password))
    except psycopg2.errors.UniqueViolation as err:
        raise HTTPException(409, "an account with that email already exists") from err
    # Auto-signin after signup, matching the UX the dashboard already had.
    token = _new_session(user_id)
    return {"token": token, "user": {"id": user_id, "email": body.email}}


@app.post("/auth/signin")
def signin(body: SigninRequest) -> dict:
    user = get_user_by_email(body.email)
    # Same error for "no such user" and "wrong password" -- don't leak which one was wrong.
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "invalid email or password")
    token = _new_session(str(user["id"]))
    return {"token": token, "user": {"id": str(user["id"]), "email": user["email"]}}


@app.post("/auth/signout")
def signout(authorization: str = Header(...)) -> dict:
    # Own header param rather than Depends(get_current_user) -- signing out with an
    # already-expired/invalid token should still succeed (idempotent), not 401.
    if authorization.startswith("Bearer "):
        delete_session(authorization.removeprefix("Bearer "))
    return {"ok": True}


def _register_audio_file(batch_dir: Path, batch_id: str, user_id: str, filename: str, content: bytes) -> str:
    """Write one audio file to disk, create its job row, enqueue it. Returns the job row id."""
    dest = batch_dir / filename
    with dest.open("wb") as out:
        out.write(content)
    job_row_id = insert_job(batch_id, user_id, filename, "queued")
    rq_job_id = enqueue_audio_file(str(dest), job_row_id)
    update_job_rq_id(job_row_id, rq_job_id)
    return job_row_id


def _extract_zip_entries(content: bytes) -> tuple[list[tuple[str, bytes]], bytes | None]:
    """Split a zip's contents into (audio entries, the manifest CSV's bytes if present).

    Non-audio/non-manifest entries (metadata, __MACOSX/.DS_Store junk that zip tools
    routinely add, directory entries, a stray readme) are skipped silently -- they aren't
    jobs, so there's nothing meaningful to report as failed.

    Entries are flattened to their basename (Path(name).name) rather than preserving the
    zip's internal folder structure -- this is also what makes path traversal ("../..") in
    a crafted entry name harmless: a basename can never escape batch_dir.
    """
    audio_entries: list[tuple[str, bytes]] = []
    manifest_bytes: bytes | None = None
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        entries = [
            info for info in zf.infolist()
            if not info.is_dir() and not Path(info.filename).name.startswith((".", "__"))
        ][:MAX_ZIP_ENTRIES]
        for info in entries:
            name = Path(info.filename).name
            ext = Path(name).suffix.lower()
            if ext == ".csv" and manifest_bytes is None:
                manifest_bytes = zf.read(info)
            elif ext in SUPPORTED_EXTENSIONS:
                audio_entries.append((name, zf.read(info)))
    return audio_entries, manifest_bytes


def _parse_manifest(content: bytes) -> set[str]:
    """Extract the set of filenames a CSV manifest (name, result_json columns) lists.

    Only `name` is read -- `result_json` is the reviewer's own reference for scoring, not
    something this endpoint validates or acts on.
    """
    text = content.decode("utf-8-sig", errors="replace")  # utf-8-sig: tolerate an Excel BOM
    reader = csv.DictReader(io.StringIO(text))
    return {row["name"].strip() for row in reader if row.get("name", "").strip()}


@app.post("/batches")
async def create_batch(
    files: list[UploadFile],
    user: dict = Depends(get_current_user),
) -> dict:
    """Accept a batch of audio files (or zip archives of them, e.g. the trial's test
    folder), optionally alongside one CSV manifest (name, result_json columns; see spec
    section 7): save audio locally, create job rows, enqueue each, and cross-check against
    the manifest if one was provided.

    A manifest is optional -- if none is uploaded, no cross-checking happens (unchanged,
    backwards-compatible behavior). At most one manifest is used, whether it arrives as a
    direct .csv upload or inside a zip; a `name` present in the manifest with no matching
    audio file becomes a failed job (so it's visible in the persistent job table and CSV
    export). An audio file with no matching manifest row is still processed normally --
    it's only surfaced in this response's `validation.unmatched`, not persisted, since it
    isn't a failure.
    """
    if not files:
        raise HTTPException(400, "no files provided")

    batch_id = str(uuid.uuid4())
    batch_dir = Path(UPLOAD_DIR) / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    job_ids: list[str] = []
    audio_entries: list[tuple[str, bytes]] = []
    manifest_names: set[str] | None = None

    for f in files:
        ext = Path(f.filename).suffix.lower()
        content = await f.read()

        if ext == ".zip":
            try:
                zip_audio, manifest_bytes = _extract_zip_entries(content)
            except zipfile.BadZipFile:
                job_ids.append(
                    insert_job(batch_id, user["id"], f.filename, "failed", error="corrupt zip file")
                )
                continue
            if not zip_audio and manifest_bytes is None:
                job_ids.append(
                    insert_job(
                        batch_id, user["id"], f.filename, "failed",
                        error="zip contained no supported audio files",
                    )
                )
                continue
            audio_entries.extend(zip_audio)
            if manifest_bytes is not None and manifest_names is None:
                manifest_names = _parse_manifest(manifest_bytes)
            continue

        if ext == ".csv":
            if manifest_names is None:
                manifest_names = _parse_manifest(content)
            continue  # the manifest itself isn't audio -- not a job

        if ext not in SUPPORTED_EXTENSIONS:
            # Recorded as an immediately-failed job rather than silently dropped, so the
            # dashboard's error column is where unsupported files surface -- not a 400 that
            # kills the whole batch over one bad file.
            job_ids.append(
                insert_job(
                    batch_id, user["id"], f.filename, "failed",
                    error=f"unsupported file extension: {ext}",
                )
            )
            continue

        audio_entries.append((f.filename, content))

    validation = {"missing": [], "unmatched": []}
    if manifest_names is not None:
        audio_names = {name for name, _ in audio_entries}
        validation["missing"] = sorted(manifest_names - audio_names)
        validation["unmatched"] = sorted(audio_names - manifest_names)
        for name in validation["missing"]:
            job_ids.append(
                insert_job(
                    batch_id, user["id"], name, "failed",
                    error="listed in manifest but file not found in upload",
                )
            )

    for name, content in audio_entries:
        job_ids.append(_register_audio_file(batch_dir, batch_id, user["id"], name, content))

    return {"batch_id": batch_id, "job_ids": job_ids, "validation": validation}


@app.get("/batches")
def list_batches(user: dict = Depends(get_current_user)) -> list[dict]:
    """One row per batch this user has ever uploaded, most recent first -- the dashboard's
    history view. A separate route from POST /batches (upload); FastAPI dispatches by
    method, so the same path is fine."""
    return db_list_batches(user["id"])


@app.get("/batches/{batch_id}/jobs")
def list_batch_jobs(batch_id: str, user: dict = Depends(get_current_user)) -> list[dict]:
    return db_list_batch_jobs(batch_id, user["id"])


@app.get("/jobs/{job_id}")
def get_job(job_id: str, user: dict = Depends(get_current_user)) -> dict:
    job = db_get_job(job_id, user["id"])
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/batches/{batch_id}/download")
def download_batch(batch_id: str, user: dict = Depends(get_current_user)) -> StreamingResponse:
    """Export the batch's results as CSV in the same shape as docs/labels.csv -- `name` and
    `result_json` columns, so the output is directly diffable against the ground truth.
    `result_json` is empty for jobs with no result yet (failed / still processing), matching
    the spec's own allowance for an empty result_json on unscored rows."""
    rows = db_list_batch_jobs(batch_id, user["id"])
    if not rows:
        raise HTTPException(404, "batch not found or has no jobs")

    # Exactly the 9 schema fields, in the same order as docs/labels.csv -- the stored
    # `result` also carries a `name` key (see app/queue.py's process_audio_file), which
    # isn't part of the schema and would make result_json not byte-for-byte comparable to
    # the ground truth's, so it's excluded here rather than dumped as-is.
    schema_fields = [
        "emotional_tone", "emotional_intensity", "background_noise_present",
        "background_noise_type", "background_noise_severity", "audio_quality",
        "speaker_overlap_present", "long_silence_present", "confidence",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["name", "result_json"])
    writer.writeheader()
    for row in rows:
        result = row.get("result")
        result_json = json.dumps({k: result[k] for k in schema_fields}) if result else ""
        writer.writerow({"name": row["filename"], "result_json": result_json})
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="predictions_{batch_id}.csv"'},
    )
