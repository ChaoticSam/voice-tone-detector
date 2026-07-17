# Voice Tone & Background Noise Detector

A self-hosted system for classifying **emotional tone** and detecting **background
noise / audio quality / speaker overlap / long silence** in production call audio, built
for AutoAce AI's technical trial (see `docs/voice_tone_background_noise_dashboard_trial.pdf`).

Two parts:

- **`backend/`** — the analysis pipeline (Python/FastAPI), a Redis/RQ job queue, and a
  Postgres-backed API with its own auth (no third-party auth provider).
- **`frontend/`** — a React dashboard: sign up/in, upload a batch (files or a zip, optionally
  with a CSV manifest), watch it process, download results.

## Architecture

```
Upload (dashboard) ──▶ FastAPI (/batches) ──▶ Redis/RQ queue ──▶ worker process
                              │                                        │
                              ▼                                        ▼
                         Postgres (jobs,                    per-file pipeline:
                         users, sessions)              preprocess → diarize (customer
                              ▲                          isolation + overlap) → dimensional
                              │                          model (arousal → intensity) +
                     dashboard polls/downloads           categorical SER (tone) → noise
                                                          (AST+DSP) → quality → silence
                                                                  │
                                                                  ▼
                                                          structured 9-field result
                                                          written back to Postgres
```

Why a queue instead of processing inline: diarization alone runs roughly 1x realtime, so a
10-minute call would hold an HTTP request open for 10+ minutes. Uploading enqueues jobs and
returns immediately; a separate worker processes them independently of any client connection
— a dropped connection, closed tab, or client that never checks back doesn't affect whether
a job finishes.

**No LLM in the final pipeline.** Earlier iterations tried acoustic-only dimensional SER and
an LLM lexical-fusion step; both underperformed a categorical SER model on the labeled calls
and the LLM added cost/latency for no accuracy gain (see `docs/technical_report.md` §5 for
the full comparison). The shipped system is 100% self-hosted models + deterministic DSP —
zero external API calls, zero per-minute cash cost.

## Setup

Requires **Python 3.12**, **ffmpeg**, **Redis**, **Node 18+**, and a Postgres database
(any Postgres works; this was built against Supabase's, used only as a plain Postgres host —
see "Auth" below for why).

### Backend

```bash
cd backend
brew install python@3.12 ffmpeg redis

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env`:

```
DATABASE_URL=postgresql://...        # your Postgres connection string
HF_TOKEN=...                         # HuggingFace token; pyannote diarization is gated,
                                      # accept its terms once at huggingface.co with this account
UPLOAD_DIR=uploads                   # optional, defaults to ./uploads
CORS_ORIGINS=http://localhost:5173   # optional, comma-separated
```

Apply the schema once:

```bash
psql "$DATABASE_URL" -f supabase/schema.sql
```

Run the pieces (three long-lived processes):

```bash
redis-server --daemonize yes

# worker (macOS needs the fork-safety workaround; harmless elsewhere)
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python -m app.worker

# API
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env   # set VITE_API_BASE_URL to the backend above
npm run dev
```

Sign up for an account from the login page (our own auth — see below), then upload audio
files or a zip.

### Tests

```bash
cd backend
pytest tests/ --ignore=tests/test_diarization.py   # fast suite
pytest tests/test_diarization.py                    # slow, model-gated, ~4 min
```

## Auth

The dashboard has its own `users`/`sessions` tables (`backend/supabase/schema.sql`) —
passwords are bcrypt-hashed and verified by the backend (`app/auth.py`), and a session is an
opaque random token looked up in the `sessions` table (`app/db.py`), not a JWT. No
third-party auth provider is involved. Deliberate scope cuts for this trial's scale (a
handful of known users, not a public product) are documented inline at each call site in
`app/api.py`: no rate limiting on signin/signup, no email verification, no password-reset
flow.

## Data handling

Raw audio never leaves local disk (`UPLOAD_DIR`) and is never sent to Postgres or any
third-party service — only derived JSON results and auth data touch the database. No
external paid API is used anywhere in the final pipeline (see `docs/technical_report.md`
§7 for the cost model this enables).

## Documentation

- [`docs/technical_report.md`](docs/technical_report.md) — approach comparison, validation
  results + confusion matrix, cost analysis, latency analysis, failure modes & next steps.
- [`docs/predictions.csv`](docs/predictions.csv) / [`predictions.json`](docs/predictions.json)
  — final predictions for the 3 provided calls.
- [`backend/README.md`](backend/README.md) — backend module-by-module reference.

## Note on data

Confidential trial audio and documents are intentionally excluded from version control (see
`.gitignore`) — this repository contains code and (non-confidential) reports only.
