try:
    from dotenv import load_dotenv as _load_dotenv
    from pathlib import Path as _Path

    _load_dotenv(_Path(__file__).resolve().parents[1] / ".env")
except Exception:  # noqa: BLE001 - dotenv optional; env vars may be set another way
    pass


SUPPORTED_EXTENSIONS = frozenset({".wav", ".mp3", ".ogg", ".flac", ".m4a"})
MIN_DURATION_SEC = 0.1

# A zip upload (e.g. the trial's test folder) is extracted server-side; cap the number of
# entries read to bound worst-case resource use from an oversized/malicious archive -- not
# full zip-bomb hardening (no compressed-ratio check), just a sane ceiling for this trial's
# scale.
MAX_ZIP_ENTRIES = 500


# --- Silence detection (long_silence_present) --------------------------------
SILENCE_FRAME_MS = 25         
SILENCE_DROP_DB = 35.0
SILENCE_ACTIVE_PERCENTILE = 95 
LONG_SILENCE_SEC = 10.0


# --- Audio quality (clear | slightly_impaired | severely_impaired) -----------
CLIP_SLIGHT_RATIO = 0.001      
CLIP_SEVERE_RATIO = 0.01       
SNR_SLIGHT_DB = 18.0
SNR_SEVERE_DB = 10.0
VOL_SLIGHT_DBFS = -32.0
VOL_SEVERE_DBFS = -45.0
SNR_NOISE_PERCENTILE = 10
SNR_ACTIVE_PERCENTILE = 90


# --- Background noise detection (AST / AudioSet) -----------------------------
# Audio Spectrogram Transformer fine-tuned on AudioSet (BSD-3, native transformers).
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
NOISE_WINDOW_SEC = 10.0
NOISE_PRESENT_THRESHOLD = 0.10
NOISE_SEVERITY_MEDIUM_PROB = 0.13
NOISE_SEVERITY_HIGH_PROB = 0.45


# --- DSP static / broadband-noise detector (hybrid with AST) -----------------
STATIC_FRAME_MS = 50
STATIC_ACTIVE_PERCENTILE = 95
STATIC_FLOOR_LO_FRAC = 0.005
STATIC_FLOOR_HI_FRAC = 0.05
STATIC_MIN_FLOOR_FRAMES = 10
STATIC_SFM_BAND_HZ = 8000 
STATIC_SFM_THRESHOLD = 0.04
STATIC_SEVERITY_MEDIUM_SFM = 0.045
STATIC_SEVERITY_HIGH_SFM = 0.10
STATIC_STRONG_EVENT_PROB = 0.30


# --- Emotion: acoustic channel (dimensional SER) -----------------------------
EMOTION_MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
EMOTION_WINDOW_SEC = 10.0
EMOTION_WINDOW_OVERLAP = 0.0
INTENSITY_HIGH_AROUSAL = 0.63
INTENSITY_MED_AROUSAL = 0.45
VALENCE_SATISFIED = 0.60      
VALENCE_NEUTRAL_LO = 0.40     
AROUSAL_UPSET = 0.60
AROUSAL_DISTRESSED = 0.75


# --- Emotion: categorical SER (tone, second acoustic vote) -------------------
CATEGORICAL_SER_MODEL_NAME = "superb/wav2vec2-base-superb-er"

# --- Transcription (internal to WhisperX diarization only) -------------------
WHISPER_MODEL_SIZE = "small"
WHISPER_COMPUTE_TYPE = "int8"

# --- Speaker diarization + overlap (stage 6, WhisperX + pyannote) -------------
DIARIZATION_MAX_SPEAKERS = 2
DIARIZATION_OVERLAP_TOLERANCE_SEC = 0.2
DIARIZATION_MIN_OVERLAP_SEC = 0.5
DIARIZATION_MIN_TURN_SEC = 0.3

# --- Job queue (Redis + RQ) ---------------------------------------------------
import os as _os

REDIS_URL = _os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = "audio-analysis"
# Generous per-file timeout: longest labeled call so far is ~3 min audio at ~1.2x realtime
# (~4 min processing); real calls could run longer, so this is a safety ceiling, not a
# calibrated bound -- a stuck job should die rather than block a worker slot forever.
JOB_TIMEOUT_SEC = 30 * 60
# Transient failures (e.g. a flaky model download, disk hiccup) get a couple of retries
# with backoff; a genuine bug in the audio (corrupt file, wrong format) will fail all three
# and surface as an error the dashboard can show -- not silently retried forever.
JOB_MAX_RETRIES = 2
JOB_RETRY_INTERVALS_SEC = [10, 60]
# How long a finished/failed job's result stays fetchable from Redis before eviction.
JOB_RESULT_TTL_SEC = 7 * 24 * 60 * 60
JOB_FAILURE_TTL_SEC = 7 * 24 * 60 * 60

# --- Postgres (durable job/results storage + our own auth tables) ------------
# Direct Postgres connection (Supabase's connection string, `postgres`/pooler role) is the
# durable store for job status/results -- Redis's result TTL (above) is a cache, not
# permanent storage. ONLY THE BACKEND holds DATABASE_URL; the frontend never gets a DB
# connection, only a session token from our own /auth/* routes -- everything else goes
# through this API. AUDIO FILES NEVER TOUCH THIS DATABASE -- only derived JSON results and
# auth data. Raw audio stays on local disk (UPLOAD_DIR). Placeholder here; fill in the real
# value in backend/.env.
DATABASE_URL = _os.getenv("DATABASE_URL", "")

# --- Auth (our own users/sessions tables, no Supabase Auth) -------------------
# Session tokens are opaque random strings looked up in the `sessions` table (see db.py) --
# not JWTs. 7 days matches JOB_RESULT_TTL_SEC above for consistency, not a specific security
# requirement; fixed from creation, no sliding-window renewal.
SESSION_TTL_SEC = 7 * 24 * 60 * 60

# --- Local upload storage (audio never leaves this disk) ---------------------
UPLOAD_DIR = _os.getenv("UPLOAD_DIR", "uploads")

# --- CORS (frontend origins allowed to call this API) -------------------------
# Comma-separated list; defaults to the Vite dev server. Set to the deployed dashboard's
# real origin(s) in production -- e.g. CORS_ORIGINS=https://dashboard.autoace.example.com
CORS_ORIGINS = [
    o.strip() for o in _os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()
]

