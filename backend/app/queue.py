"""Job queue: decouples "a file was uploaded" from "a file was processed".

Why this exists: diarization alone runs ~1x realtime, so processing a file inside an HTTP
request is a broken design -- a 10-minute call would hold a request open for 10+ minutes.
Instead, uploading enqueues a job and returns immediately; a separate worker process (started
with `rq worker -c app.queue`, or `python -m app.worker`) picks it up independently. This is
also what makes the pipeline survive the failure modes we're designing for: a dropped
connection, a closed browser tab, or a client that never checks back. The job lives in Redis,
not in any request's memory, so none of that affects whether it runs to completion.

Each job is a single audio file -> the final structured result (see result.py). Progress is
reported via `job.meta["stage"]`, updated as each pipeline stage starts, so a caller can poll
a job's status and show *which* stage it's in, not just queued/running/done.
"""

from __future__ import annotations

from typing import Any

import redis
from rq import Callback, Queue, Retry
from rq.job import Job

from app.config import (
    JOB_FAILURE_TTL_SEC,
    JOB_MAX_RETRIES,
    JOB_RESULT_TTL_SEC,
    JOB_RETRY_INTERVALS_SEC,
    JOB_TIMEOUT_SEC,
    QUEUE_NAME,
    REDIS_URL,
)


def get_redis() -> redis.Redis:
    return redis.from_url(REDIS_URL)


def get_queue() -> Queue:
    return Queue(QUEUE_NAME, connection=get_redis())


def _set_stage(path: str, job_row_id: str, stage: str) -> None:
    """Record which pipeline stage a running job is in.

    Written to two places: RQ's job.meta (fast, for direct-Redis polling / tests) and the
    Postgres `jobs` row (durable, what the dashboard actually reads via the FastAPI API).
    The Postgres write is best-effort -- a transient DB hiccup here must not fail the actual
    audio processing job; the final result/error (see the on_success/on_failure callbacks
    below) is the durable source of truth and is NOT swallowed the same way.
    """
    from rq import get_current_job

    job = get_current_job()
    if job is not None:
        job.meta["stage"] = stage
        job.save_meta()

    try:
        from app.db import update_job_stage

        update_job_stage(job_row_id, stage)
    except Exception:  # noqa: BLE001 - progress reporting is best-effort
        pass


def process_audio_file(path: str, job_row_id: str) -> dict[str, Any]:
    """The task a worker runs: preprocess one file -> the full structured result.

    `job_row_id` is the Postgres `jobs.id` this run corresponds to -- passed through so
    stage updates and the terminal on_success/on_failure callbacks know which row to write.
    This is what gets enqueued (by reference, as `app.queue.process_audio_file`) -- RQ
    imports and calls it by dotted path, so it must stay importable without side effects.
    """
    from app.analysis.result import build_result
    from app.audio.preprocessing import PreprocessError, preprocess

    _set_stage(path, job_row_id, "preprocessing")
    pre = preprocess(path)
    if isinstance(pre, PreprocessError):
        raise ValueError(f"{pre.name}: {pre.error}")

    _set_stage(path, job_row_id, "analyzing")  # diarization/noise/quality/silence/emotion
    result = build_result(pre)

    _set_stage(path, job_row_id, "done")
    return {"name": pre.name, **result.model_dump()}


def on_job_success(job, connection, result, *args, **kwargs) -> None:
    """RQ success callback (fires once, after retries are exhausted/unneeded): persist the
    final result to Postgres. Import-path-resolved by RQ -- must stay a top-level function."""
    from app.db import update_job_success

    update_job_success(job.args[1], result)


def on_job_failure(job, connection, type, value, traceback) -> None:
    """RQ failure callback (fires once, after all retries are exhausted): persist the error."""
    from app.db import update_job_failure

    update_job_failure(job.args[1], str(value))


def enqueue_audio_file(path: str, job_row_id: str) -> str:
    """Submit one file for processing; returns immediately with the RQ job id.

    `job_row_id` is the Postgres `jobs.id` row the caller has already created (status
    "queued") -- the worker updates that same row as it progresses and on completion.
    """
    job = get_queue().enqueue(
        process_audio_file,
        path,
        job_row_id,
        job_timeout=JOB_TIMEOUT_SEC,
        retry=Retry(max=JOB_MAX_RETRIES, interval=JOB_RETRY_INTERVALS_SEC),
        result_ttl=JOB_RESULT_TTL_SEC,
        failure_ttl=JOB_FAILURE_TTL_SEC,
        on_success=Callback(on_job_success),
        on_failure=Callback(on_job_failure),
    )
    return job.id


def get_job_status(job_id: str) -> dict[str, Any]:
    """Poll a job's current state -- status, stage (if running), result, or error."""
    job = Job.fetch(job_id, connection=get_redis())
    status = job.get_status(refresh=True)
    out: dict[str, Any] = {"job_id": job_id, "status": status}
    if status == "started":
        out["stage"] = job.meta.get("stage")
    elif status == "finished":
        out["result"] = job.result
    elif status == "failed":
        out["error"] = job.latest_result().exc_string if job.latest_result() else "unknown error"
    return out
