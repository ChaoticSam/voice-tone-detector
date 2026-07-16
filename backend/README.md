# Voice Tone Detector — Backend

Stepwise implementation of AutoAce's voice-tone + background-noise analysis pipeline.

**Current status:** Stage 3 — DSP analysis (audio quality + silence).

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
