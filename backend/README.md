# Voice Tone Detector — Backend

Stepwise implementation of AutoAce's voice-tone + background-noise analysis pipeline.

**Current status:** Stage 6 — speaker diarization + overlap.

## Setup

Requires **Python 3.12** (the ML stack we add later lacks 3.14 wheels) and **ffmpeg**.

```bash
brew install python@3.12 ffmpeg

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Validation (stage 1)

Validates that an audio file is a supported, decodable format and extracts its
metadata (duration, sample rate, channels). It does **not** resample or normalize —
that is the preprocessing stage. Failures never raise; they return a structured
`invalid` result so a batch can isolate per-file problems.

Supported: wav, mp3, ogg, flac, m4a. Decoder is soundfile first (fast, metadata-only
reads), with an ffprobe fallback for formats libsndfile can't handle (e.g. m4a).

```bash
# CLI (run from backend/, with the venv active)
PYTHONPATH=. python -m app.audio.validation ../audio_files/*.ogg

# Tests
PYTHONPATH=. pytest tests/test_validation.py -v
```

Programmatic use:

```python
from app.audio.validation import validate_file, validate_batch

r = validate_file("call_001.ogg")
# r.status -> "valid" | "invalid"
# r.error  -> machine code when invalid (empty_file, unsupported_extension,
#             decode_failed, no_audio_stream, too_short)
```

## Preprocessing (stage 2)

Turns a validated file into the one canonical waveform the models consume (mono, 16 kHz)
and measures cheap acoustic features on the **original** signal. Principle: *measure, don't
mutate* — the models self-normalize, so we skip loudness normalization, spectrogram
precompute, silence trimming, and denoising (all either redundant or destroy evidence a
required output depends on, e.g. clipping for `audio_quality`).

Output package (`Preprocessed`): `waveform_16k` (model input), `signal_original` (for
measurement), and `features` (`SignalFeatures`: peak, clipping_ratio, rms, rms_db,
zero_crossing_rate, dc_offset, crest_factor).

```bash
# CLI — prints a features table
PYTHONPATH=. python -m app.audio.preprocessing ../audio_files/*.ogg

# Tests
PYTHONPATH=. pytest tests/ -v
```

```python
from app.audio.preprocessing import preprocess, chunk_waveform

pre = preprocess("call_001.ogg")
pre.waveform_16k          # np.float32 mono @ 16 kHz -> feed to models
pre.features.clipping_ratio
# chunk_waveform is a reusable inference-windowing helper for the model services:
windows = chunk_waveform(pre.waveform_16k, pre.sample_rate, window_sec=5)
```

## DSP analysis (stage 3)

Model-free, deterministic services that consume the preprocessed package:

- **audio_quality** (`clear | slightly_impaired | severely_impaired`) — worst-case over
  clipping, a rough percentile SNR estimate, and loudness. Band-limiting (normal for phone
  audio) is deliberately *not* treated as impairment. Thresholds are principled but
  **uncalibrated** pending ground-truth labels.
- **long_silence_present** — energy-based dead-air detection with an adaptive threshold
  (relative to each file's own active level, so quiet calls work). A VAD is intentionally
  not used here; Silero is introduced later for true speech regions (overlap/speaking-rate).

### Calibration against `docs/labels.csv`

Both fields were checked against the 3 real labeled calls (`tests/test_calibration.py`,
skipped if `docs/labels.csv` is absent — it's gitignored as confidential). `audio_quality`
matched 3/3 out of the box. `long_silence_present` initially had 2/3 false positives at a
3.0s threshold: one was a near-start "call connecting" gap, the other a quiet-but-not-silent
stretch where low-level background noise persists. Raising `LONG_SILENCE_SEC` to 10.0s (see
`app/config.py` for the full rationale) fixes both without any other threshold change.
**Caveat:** all 3 labeled calls are negative for this field, so calibration only bounds the
threshold from below — there's no positive example to validate sensitivity against.

```bash
# CLI — runs both services (preprocess -> quality + silence)
PYTHONPATH=. python -m app.analysis ../audio_files/*.ogg
```

```python
from app.analysis.audio_quality import assess_quality
from app.analysis.silence import detect_silence
from app.audio.preprocessing import preprocess

pre = preprocess("call_001.ogg")
q = assess_quality(pre)                                   # -> audio_quality, snr_db, reasons
s = detect_silence(pre.signal_original, pre.original_sample_rate)  # -> long_silence_present
```

## Background noise detection (stage 4)

First ML model: the Audio Spectrogram Transformer fine-tuned on AudioSet
(`MIT/ast-finetuned-audioset-10-10-0.4593`, BSD-3, native `transformers`). Acoustic-only —
no transcript. Produces `background_noise_present / _type / _severity`.

- Long calls are chunked to ~10s windows (AST is fixed-length-input); per-class
  probabilities are **max-aggregated** across windows (multi-label → sigmoid).
- **Speech and telephony artifacts are excluded** before ranking. Telephony classes
  (Sidetone, Dial tone, Busy signal, Telephone) dominate real call audio and are intrinsic
  to the medium, not background noise — excluding them is what lets the real noise surface.
- Weights download once from the HF Hub (~330MB, cached in `~/.cache/huggingface`). Only
  public model artifacts are fetched; confidential audio never leaves local infra.

**Hybrid detection.** AST is an *event* classifier — great at discrete sources (TV, music,
typing) but blind to *additive broadband noise* (static/hiss/crackle), which is a
signal-level phenomenon. So a DSP **static detector** (`static_noise.py`) runs alongside
AST: it measures the spectral flatness of the noise-floor frames (broadband ⇒ static) plus
impulsive crackle activity. When static is detected, it takes the reported `type` unless AST
found a strong discrete event — so a weak AST guess (e.g. a spurious "crunch") can't
override real static. This is the deterministic-features + learned-model hybrid the spec
rewards.

Calibrated against `docs/labels.csv`: `background_noise_present` 3/3, `severity` 3/3, and
`type` now semantically 3/3 — call_001 "" ✓, call_002 "TV" ✓, call_003 "static" ✓ (AST
alone had mislabelled this "crunch"). The static detector is validated on one positive
example; severity bands remain lightly-calibrated (see `app/config.py`).

**Latency:** ~34× realtime on CPU (warm model), ≈1.7s compute per audio-minute.

```bash
# Runs quality + silence + noise together
PYTHONPATH=. python -m app.analysis ../audio_files/*.ogg
```

```python
from app.analysis.noise import detect_noise
n = detect_noise(pre)   # -> background_noise_present / _type / _severity, top_classes
```

## Emotion — acoustic channel (stage 5a)

Emotion has two channels: **prosodic** ("how it was said") and **lexical** ("what was
said"). This stage builds the prosodic channel with a dimensional SER model
(`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` → arousal/valence/dominance).
Arousal → `emotional_intensity`; valence → `emotional_tone`. It returns a **provisional**
verdict plus the raw dimensions, so stage 5b (Whisper transcript + LLM) can *fuse* rather
than override. Neutral is kept independent of arousal, so a loud-but-neutral speaker isn't
miscalled upset (respects the spec's loudness caveat). License CC-BY-NC-SA (non-commercial,
disclosed; one-file swap). ~1.2GB weights, cached locally; audio stays local.

**Chunking** (your context-loss concern, answered with data): a sweep over
{20,15,10,8,5,3}s × {0, 50%} overlap × {mean, peak} picked **10s / no overlap / mean** —
closest to the model's ~8-11s training regime, stable (peak-salience grabbed outlier
chunks), cheapest.

**Key empirical finding (calibration vs `docs/labels.csv`):**
- `emotional_intensity` — **3/3** (call_001 arousal 0.68 → high; 002/003 ~0.58 → medium).
- `emotional_tone` — **acoustically unreliable: valence is compressed (~0.50-0.57) and
  does NOT separate satisfied/neutral/upset** — the ordering is even anti-correlated
  (satisfied call_003 has the *lowest* valence). No threshold can fit the labels, so tone
  thresholds are principled (not overfit) and all three currently read "neutral". This is
  the concrete, measured proof that tone needs the **lexical channel** — the frustration in
  "I've called five times already" is in the *words*, not the prosody. Owned by stage 5b.

**Latency:** ~36× realtime on CPU (warm model), ≈1.7s compute per audio-minute.

```python
from app.analysis.emotion import detect_emotion
e = detect_emotion(pre)   # -> emotional_tone (provisional), emotional_intensity,
                          #    arousal, valence, dominance, per_chunk
```

## Speaker diarization + overlap (stage 6)

**Root-cause fix.** These are AI-receptionist calls — a TTS agent ("Erica") plus the human
customer. Analyzing the whole call mixes both speakers, which broke emotion (the bot's flat
voice dilutes the acoustics; its lines mislead the transcript). This stage separates them.

`diarization.py` uses **WhisperX** (faster-whisper + word alignment + pyannote
`speaker-diarization-community-1`), self-hosted — audio never leaves local infra. pyannote is
gated: set `HF_TOKEN` in `.env` and accept the model terms once on Hugging Face.

Produces:
- **`speaker_overlap_present`** — a required output field. Overlap must be *meaningful*
  ("enough to affect understanding" per the spec): total cross-speaker overlap ≥ 0.5s, so
  brief boundary blips don't count.
- **agent vs customer roles** — agent = speaker of the scripted greeting ("how can I help",
  "I'm Erica", "cómo puedo ayudar"), which is far more robust than "who spoke first" (a
  stray caller utterance can precede the greeting). Customer = the most-talking other speaker.
- **customer-only transcript + time-ranges** — consumed by the *later* emotion stage so it
  analyzes the customer only.

2-party is hinted (`max_speakers=2`) to stop the diarizer over-segmenting the bot.

**Calibration vs `docs/labels.csv`: `speaker_overlap_present` 3/3** (call_001 0.35s→False,
call_002 0.98s→True, call_003 2.35s→True). Roles verified: e.g. call_003's customer
transcript is *"I want an appointment… I need a checkup"* — the bot's "we're closed" lines
are correctly excluded, which is exactly the isolation the emotion stage needs.

**Latency:** ~1× realtime on CPU (diarization is the heaviest stage; call_003's 172s took
~165s). Much faster on GPU.

```bash
PYTHONPATH=. python -m app.analysis.diarize_cli ../audio_files/*.ogg
```

```python
from app.analysis.diarization import diarize
from app.audio.preprocessing import preprocess
d = diarize(preprocess("call_003.ogg"))
d.speaker_overlap_present   # required field
d.customer_transcript       # human-only text, for the emotion stage
d.customer_ranges           # customer turn [start,end] time-ranges
```

**Next:** wire `customer_transcript` + `customer_ranges` into the emotion stage (run acoustic
+ lexical on the customer only) — the hypothesis this stage sets up.
