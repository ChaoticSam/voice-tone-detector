"""Shared configuration for the audio pipeline.

Thresholds and constants live here so every stage (validation, preprocessing,
analysis) reads from one source of truth.
"""

# Load secrets (HF_TOKEN, OPENAI_API_KEY) from backend/.env if present. .env is gitignored;
# never commit it. No-op if python-dotenv or the file is absent.
try:
    from dotenv import load_dotenv as _load_dotenv
    from pathlib import Path as _Path

    _load_dotenv(_Path(__file__).resolve().parents[1] / ".env")
except Exception:  # noqa: BLE001 - dotenv optional; env vars may be set another way
    pass

# Container formats we accept. The provided samples are .ogg; the hidden test set
# may add mp3/m4a. soundfile handles wav/flac/ogg natively; ffmpeg covers the rest.
SUPPORTED_EXTENSIONS = frozenset({".wav", ".mp3", ".ogg", ".flac", ".m4a"})

# Clips shorter than this carry no analyzable signal and are rejected as too_short.
MIN_DURATION_SEC = 0.1


# --- Silence detection (long_silence_present) --------------------------------
# Energy-based dead-air detection. Thresholds are relative to each file's own active
# level, so quiet calls are handled without an absolute-volume assumption.
SILENCE_FRAME_MS = 25          # analysis frame length
# A frame is "silent" when its RMS falls this many dB below the file's active level
# (the high-percentile frame energy). Dead air / digital silence sits far below this.
SILENCE_DROP_DB = 35.0
SILENCE_ACTIVE_PERCENTILE = 95  # percentile of frame RMS taken as the "active" level
# Longest silent run at/above this length flips long_silence_present to true.
#
# Calibrated against docs/labels.csv (all 3 real calls, all long_silence_present=false):
#   call_001 longest run 2.90s, call_002 3.18s (near-start "connecting" gap, not a
#   mid-call problem), call_003 7.33s (a quiet-but-not-silent stretch -- the "sharp
#   static" background noise persists at low level, so a human wouldn't call it dead
#   air even though relative energy drops ~50-70dB below the active level).
# At the old value (3.0s) both call_002 and call_003 were false positives. Any value
# in (7.33, ...] reproduces all 3 labels; 10.0 gives headroom above the boundary
# rather than snapping to it. CAVEAT: all 3 labeled examples are negative for this
# field, so this only bounds the threshold from below -- we have no positive example
# to validate sensitivity, and this should be revisited once positive examples exist.
LONG_SILENCE_SEC = 10.0


# --- Audio quality (clear | slightly_impaired | severely_impaired) -----------
# NOTE: uncalibrated principled defaults — no ground-truth labels available yet.
# Tune against labels.csv once provided. Final quality = worst-case across signals.
# Clipping: fraction of samples at/above full scale. A few samples touching 1.0 (ogg
# encode overshoot) is not audible distortion, so slight starts well above that.
CLIP_SLIGHT_RATIO = 0.001      # 0.1% of samples clipped
CLIP_SEVERE_RATIO = 0.01       # 1% of samples clipped
# SNR (rough percentile estimate): low SNR = static / hiss / noisy line.
SNR_SLIGHT_DB = 18.0
SNR_SEVERE_DB = 10.0
# Overall loudness. Very low level = hard-to-analyze quiet recording.
VOL_SLIGHT_DBFS = -32.0
VOL_SEVERE_DBFS = -45.0
# Percentiles for the rough SNR estimate (active vs noise-floor frame energy).
SNR_NOISE_PERCENTILE = 10
SNR_ACTIVE_PERCENTILE = 90


# --- Background noise detection (AST / AudioSet) -----------------------------
# Audio Spectrogram Transformer fine-tuned on AudioSet (BSD-3, native transformers).
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
# AST is a fixed-length-input model trained on ~10s clips; long calls are chunked to
# this window and class probabilities are max-aggregated across windows (a noise event
# in even one window should register rather than being averaged away).
NOISE_WINDOW_SEC = 10.0
# Top non-speech (and non-telephony) class probability at/above which noise is present.
# Calibrated against docs/labels.csv after excluding telephony artifacts:
#   call_001 (no noise) top ~0.072 | call_002 (TV) 0.149 | call_003 (static) 0.203.
# 0.10 separates the negative from both positives with margin on each side.
NOISE_PRESENT_THRESHOLD = 0.10
# Severity bands on the top noise-class probability. CAVEAT: only 2 positive examples,
# both labelled "medium" (TV=0.149, static=0.203), so this is fit to 2 same-class points
# and is the least reliable field -- severity really tracks noise loudness, which the
# classifier probability only loosely proxies. Revisit with more labeled data.
NOISE_SEVERITY_MEDIUM_PROB = 0.13
NOISE_SEVERITY_HIGH_PROB = 0.45


# --- DSP static / broadband-noise detector (hybrid with AST) -----------------
# AST is an event classifier and misses additive broadband noise (hiss/static/crackle),
# which is a signal-level phenomenon best caught with DSP -- analogous to clipping/silence.
# Static shows up as a flat (broadband) noise floor + impulsive crackle bursts.
# Thresholds separate the real files: SFM(0-8k) of the noise floor was
#   call_001 (clean) 0.020 | call_002 (TV) 0.031 | call_003 (static) 0.055.
# Validated on ONE positive example (call_003) -- principled features, but revisit with
# more static-labeled data before trusting the exact thresholds.
STATIC_FRAME_MS = 50
STATIC_ACTIVE_PERCENTILE = 95
STATIC_FLOOR_LO_FRAC = 0.005     # noise-floor frames: quiet but not digital silence
STATIC_FLOOR_HI_FRAC = 0.05
STATIC_MIN_FLOOR_FRAMES = 10     # need enough floor frames to judge
STATIC_SFM_BAND_HZ = 8000        # measure flatness in the meaningful band, not empty HF
STATIC_SFM_THRESHOLD = 0.04      # noise-floor flatness above this => broadband static
STATIC_SEVERITY_MEDIUM_SFM = 0.045
STATIC_SEVERITY_HIGH_SFM = 0.10
# An AST discrete event this strong overrides a static finding for the reported type.
STATIC_STRONG_EVENT_PROB = 0.30


# --- Emotion: acoustic channel (stage 5a, dimensional SER) -------------------
# audeering wav2vec2 dimensional model -> arousal/dominance/valence (~0..1).
EMOTION_MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
# Chunk config chosen by the sweep (scratchpad experiment): 10s / no overlap / mean.
# 10s is closest to the model's ~8-11s (MSP-Podcast) training regime among the stable
# options; mean aggregation is stable (peak-salience grabbed outlier chunks); overlap
# barely changed the mean, so 0.0 is kept for lower cost.
EMOTION_WINDOW_SEC = 10.0
EMOTION_WINDOW_OVERLAP = 0.0

# Intensity from arousal. Calibrated to the 3 labels (10s/mean arousal):
#   call_001 0.68 (high) | call_002 0.59, call_003 0.58 (medium).
INTENSITY_HIGH_AROUSAL = 0.63
INTENSITY_MED_AROUSAL = 0.45

# Tone from valence. IMPORTANT: on the 3 real files acoustic valence is compressed
# (~0.50-0.57) and does NOT separate satisfied/neutral/upset -- the ordering is even
# ANTI-correlated (satisfied call_003 has the LOWEST valence). No threshold can fit the
# labels, so these are PRINCIPLED values, deliberately NOT tuned to force a pass. Acoustic
# tone is therefore unreliable by itself; the lexical channel (stage 5b) owns tone. This
# empirically motivates 5b -- see the memo.
VALENCE_SATISFIED = 0.60      # >= -> positive/satisfied
VALENCE_NEUTRAL_LO = 0.40     # [NEUTRAL_LO, SATISFIED) -> neutral; < -> negative
# Within negative valence, arousal separates the negatives. UNVALIDATED -- we have no
# frustrated/distressed labeled examples.
AROUSAL_UPSET = 0.60
AROUSAL_DISTRESSED = 0.75


# --- Transcription (shared: WhisperX diarization + faster-whisper lexical) ----
# Self-hosted; audio never leaves local infra. `small` int8 is the cost/quality sweet
# spot (we settled on it over large-v3 for cost/latency; the transcript is a feature for
# fusion, not a scored output). HF_TOKEN (.env) gates WhisperX's pyannote diarization.
WHISPER_MODEL_SIZE = "small"
WHISPER_COMPUTE_TYPE = "int8"

# --- Speaker diarization + overlap (stage 6, WhisperX + pyannote) -------------
# AI-receptionist calls are 2-party (bot + one caller); hint the diarizer to avoid
# over-segmenting the bot into multiple speakers. Set None for fully automatic.
DIARIZATION_MAX_SPEAKERS = 2
# A single cross-speaker overlap must exceed this (s) to count toward the total.
DIARIZATION_OVERLAP_TOLERANCE_SEC = 0.2
# speaker_overlap_present requires total overlap >= this. The spec says overlap must be
# "enough to affect understanding" -- brief boundary blips (~0.3s) don't qualify.
# Calibrated to the 3 labels: call_001 0.35s (False), call_002 0.98s, call_003 2.35s (True).
DIARIZATION_MIN_OVERLAP_SEC = 0.5
# Turns shorter than this (s) are ignored for role assignment / overlap (diarizer noise).
DIARIZATION_MIN_TURN_SEC = 0.3

# --- Emotion: lexical channel + fusion (stage 5b) ----------------------------
# Lexical-fusion LLM. Provider abstraction (see llm_client.py) — one-line swap.
# gpt-4.1-mini: ~$0.40/$1.60 per 1M. The transcript (derived text) leaves our infra to
# the provider -> DISCLOSE per trial §11.
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4.1-mini"
# Confidence when the LLM call fails / no API key: fall back to the acoustic verdict.
ACOUSTIC_FALLBACK_CONFIDENCE = 0.45
