# Voice Tone Detector — Backend

The analysis pipeline, job queue, and API for AutoAce's voice-tone + background-noise
trial. See the [root README](../README.md) for the full-system setup (frontend, Postgres,
Redis) and [`docs/technical_report.md`](../docs/technical_report.md) for the approach
comparison, validation results, cost/latency analysis, and known limitations.

## Setup

Requires **Python 3.12** (the ML stack lacks 3.14 wheels) and **ffmpeg**.

```bash
brew install python@3.12 ffmpeg

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.env` (see the root README for the full variable list): `DATABASE_URL`, `HF_TOKEN`
(pyannote diarization is gated — accept its terms once on Hugging Face with this account).

## Pipeline stages

### Validation

Confirms a file is a supported, decodable format and extracts metadata (duration, sample
rate, channels) without resampling/normalizing. Failures never raise — they return a
structured `invalid` result so a batch can isolate per-file problems. Supported: wav, mp3,
ogg, flac, m4a.

```bash
PYTHONPATH=. python -m app.audio.validation ../audio_files/*.ogg
```

### Preprocessing

Turns a validated file into the one canonical waveform the models consume (mono, 16 kHz)
and measures cheap acoustic features on the **original** signal — *measure, don't mutate*;
models self-normalize, and several "cleanups" (loudness normalization, silence trimming,
denoising) would destroy evidence a required output depends on (e.g. clipping for
`audio_quality`).

```bash
PYTHONPATH=. python -m app.audio.preprocessing ../audio_files/*.ogg
```

### DSP analysis: audio quality + long silence

Model-free, deterministic services:

- **`audio_quality`** (`clear | slightly_impaired | severely_impaired`) — worst-case over
  clipping, a rough percentile SNR estimate, and loudness.
- **`long_silence_present`** — energy-based dead-air detection with a threshold relative to
  each file's own active level (so quiet calls work without an absolute-volume assumption).

Calibrated against `docs/labels.csv`: both fields 3/3. `long_silence_present`'s threshold
(`LONG_SILENCE_SEC` in `app/config.py`) was raised from an initial 3.0s after two false
positives (a call-connecting gap; a quiet-but-not-silent stretch) — all 3 labeled calls are
negative for this field, so calibration only bounds the threshold from below; there's no
positive example to validate sensitivity against.

### Background noise

`MIT/ast-finetuned-audioset-10-10-0.4593` (Audio Spectrogram Transformer, BSD-3), chunked to
~10s windows with max-aggregated per-class probabilities. Speech/telephony artifacts (dial
tone, sidetone, busy signal) are excluded before ranking — they're intrinsic to phone audio,
not background noise.

**Hybrid with DSP.** AST is an event classifier — good at discrete sources (TV, music,
typing), blind to additive broadband noise (static/hiss). A DSP static detector
(`static_noise.py`, spectral flatness of noise-floor frames) runs alongside it; when static
is detected it drives the reported type unless AST found a strong discrete event.

Calibrated 3/3 on `background_noise_present`, `_severity`, and `_type` (semantically —
call_003's "static" vs. the label's "sharp static").

### Emotion (tone + intensity)

The approach that changed the most during development — see
[`docs/technical_report.md`](../docs/technical_report.md) §5 for the full comparison
(acoustic-dimensional alone → LLM lexical fusion → diarization-based customer isolation →
categorical SER, which won and is what ships). Summary of the final design:

- **`emotional_intensity`** — a dimensional SER model
  (`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`) run on the **whole call**;
  arousal thresholded into low/medium/high. Whole-call (not customer-only) arousal is what's
  used here — customer-only arousal collapsed two of the three labeled calls to
  near-identical values that whole-call arousal cleanly separates.
- **`emotional_tone`** — a categorical SER model (`superb/wav2vec2-base-superb-er`, 4
  classes: neutral/happy/angry/sad, trained on IEMOCAP) run on the **customer-only** audio
  (isolated via diarization — see below). `angry`/`sad` are disambiguated into
  frustrated/upset/distressed using the same arousal bands as the intensity model.
- **`confidence`** — the categorical model's top-class probability (the one genuinely
  probabilistic signal in the pipeline; everything else is deterministic thresholds).

No LLM, no transcript handling downstream of diarization — both were tried and dropped
(see the report).

```python
from app.analysis.emotion_pipeline import analyze_customer_emotion
from app.audio.preprocessing import preprocess

result = analyze_customer_emotion(preprocess("call_001.ogg"))
result.emotional_tone       # categorical model, customer-only
result.emotional_intensity  # dimensional model, whole-call
```

### Speaker diarization + overlap

These are AI-receptionist calls — a TTS agent ("Erica") plus the human customer. WhisperX
(faster-whisper + word alignment + pyannote `speaker-diarization-community-1`) separates
them. Role assignment: the agent is whoever speaks the scripted greeting ("how can I help",
"I'm Erica") — more robust than "who spoke first" (a stray caller utterance can precede the
greeting). `max_speakers=2` is hinted to stop the diarizer over-segmenting the bot.

Produces `speaker_overlap_present` (required output field; overlap must total ≥0.5s to
count — brief boundary blips don't) and the customer-only audio the emotion stage consumes.
Calibrated 3/3 on `speaker_overlap_present`.

```bash
PYTHONPATH=. python -m app.analysis.diarize_cli ../audio_files/*.ogg
```

### Result aggregation

`app/analysis/result.py`'s `build_result(pre)` runs every stage above and assembles the
final 9-field schema (`app/analysis/result.py::StructuredResult`) matching
`docs/labels.csv`'s exact shape.

## Job queue

`app/queue.py` (Redis + RQ) decouples "a file was uploaded" from "a file was processed" —
diarization alone runs ~1x realtime, so processing inline in an HTTP request would hold it
open for as long as the call is. `enqueue_audio_file` submits a job; a long-lived worker
process (`python -m app.worker`) picks it up independently, survives client
disconnects/dropped tabs, retries transient failures (2 retries, 10s/60s backoff), and
persists progress/results into Postgres via `app/db.py` as it runs.

## API + Auth

`app/api.py` (FastAPI): signup/signin/signout, batch upload (files, zips, and an optional
CSV manifest — see the root README and `docs/technical_report.md` §9 for the manifest
validation semantics), job status polling, CSV download.

Auth is our own — no third-party provider. `app/auth.py` hashes passwords (bcrypt) and
generates opaque session tokens; `app/db.py`'s `get_current_user` looks a token up directly
in the `sessions` table (a plain table read, not JWT verification) and joins to `users`.
Deliberate scope cuts for this trial's scale (documented inline at each call site in
`app/api.py`): no rate limiting, no email verification, no password-reset flow.

## Tests

```bash
PYTHONPATH=. pytest tests/ --ignore=tests/test_diarization.py -v   # fast suite
PYTHONPATH=. pytest tests/test_diarization.py -v                   # slow, model-gated, ~4 min
```
