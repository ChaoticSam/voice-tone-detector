"""Audio preprocessing layer — stage 2 of the pipeline.

Turns a validated file into the single canonical waveform the models consume, plus a
cheap measurement pass on the *original* signal.

Design principle: **measure, don't mutate.** The models we chose all normalize their own
input (wav2vec2/emotion2vec processor, Whisper log-mel, AST mean/std), so we do NOT apply
loudness normalization, spectrogram pre-computation, silence trimming, or denoising here —
those would either duplicate the model's own preprocessing or destroy evidence that a
required output depends on (e.g. clipping for ``audio_quality``, silence for
``long_silence_present``).

What we DO:
  * decode once, convert to mono, resample to 16 kHz  -> the waveform models consume
  * keep the original-sample-rate mono signal          -> for quality/feature measurement
  * compute cheap, waveform-only acoustic features on the original signal

Heavier, consumer-specific work (model inference, VAD/silence decision, pitch, SNR,
spectral features, per-model chunking at inference) belongs to later stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import soxr
from pydantic import BaseModel

from app.audio.validation import ValidationResult

MODEL_SAMPLE_RATE = 16000

# A sample counts as "clipped" when it sits within this margin of digital full scale.
_CLIP_THRESHOLD = 0.99
# Floor for dB / ratio math so silent signals don't blow up to -inf / divide-by-zero.
_EPS = 1e-12


class SignalFeatures(BaseModel):
    """Cheap, pure-numpy acoustic stats measured on the original-rate signal.

    These feed downstream audio-quality and emotion-intensity logic. We compute only what
    is essentially free here; model- and transcript-dependent features come later.
    """

    duration_sec: float
    peak: float                 # max |sample|; can exceed 1.0 on clipped/overencoded audio
    clipping_ratio: float       # fraction of samples at/above full scale -> audio_quality
    rms: float                  # loudness (linear)
    rms_db: float               # loudness (dBFS)
    zero_crossing_rate: float   # noisiness / fricative proxy
    dc_offset: float            # mean sample value; measured, not corrected
    crest_factor: float         # peak / rms; dynamic-range proxy


@dataclass
class Preprocessed:
    """In-memory audio package handed to the analysis services.

    Holds numpy arrays, so it is a dataclass rather than a pydantic model; the serializable
    parts (metadata + ``features``) can be dumped for the feature store / API.
    """

    name: str
    waveform_16k: np.ndarray        # float32 mono @ 16 kHz — the model input
    signal_original: np.ndarray     # float32 mono @ original SR — for measurement
    original_sample_rate: int
    features: SignalFeatures
    sample_rate: int = MODEL_SAMPLE_RATE


@dataclass
class PreprocessError:
    """Per-file failure marker so a batch can isolate one bad file from the rest."""

    name: str
    error: str


def _to_mono(x: np.ndarray) -> np.ndarray:
    """Collapse to a 1-D float32 mono signal by averaging channels.

    soundfile returns shape (n,) for mono or (n, channels) for multi-channel. Averaging is
    lossless when channels are identical (as in the provided fake-stereo files) and a sane
    default otherwise.
    """
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)
    return x.mean(axis=1).astype(np.float32)


def _resample_16k(mono: np.ndarray, sr: int) -> np.ndarray:
    """Resample a mono signal to 16 kHz with soxr. No-op when already at 16 kHz."""
    if sr == MODEL_SAMPLE_RATE:
        return mono.astype(np.float32, copy=False)
    out = soxr.resample(mono, sr, MODEL_SAMPLE_RATE)
    return out.astype(np.float32, copy=False)


def compute_signal_features(mono: np.ndarray, sr: int) -> SignalFeatures:
    """Compute cheap acoustic features on a mono signal (numpy only)."""
    n = mono.size
    if n == 0:
        return SignalFeatures(
            duration_sec=0.0, peak=0.0, clipping_ratio=0.0, rms=0.0,
            rms_db=-np.inf, zero_crossing_rate=0.0, dc_offset=0.0, crest_factor=0.0,
        )

    abs_x = np.abs(mono)
    peak = float(abs_x.max())
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    clipping_ratio = float(np.count_nonzero(abs_x >= _CLIP_THRESHOLD) / n)
    # Zero-crossing rate: fraction of adjacent samples where the sign flips.
    zcr = float(np.count_nonzero(np.diff(np.signbit(mono))) / n)
    dc_offset = float(mono.mean(dtype=np.float64))
    rms_db = float(20.0 * np.log10(max(rms, _EPS)))
    crest_factor = float(peak / rms) if rms > _EPS else 0.0

    return SignalFeatures(
        duration_sec=round(n / sr, 3),
        peak=round(peak, 6),
        clipping_ratio=round(clipping_ratio, 6),
        rms=round(rms, 6),
        rms_db=round(rms_db, 3),
        zero_crossing_rate=round(zcr, 6),
        dc_offset=round(dc_offset, 8),
        crest_factor=round(crest_factor, 4),
    )


def chunk_waveform(
    x: np.ndarray, sr: int, window_sec: float, hop_sec: Optional[float] = None
) -> list[np.ndarray]:
    """Split a signal into (optionally overlapping) windows for model inference.

    Reusable utility for the emotion / AST services, which have different input-length
    limits and window preferences. It is intentionally NOT applied to the ``Preprocessed``
    package — each model chooses its own window size at inference time.

    - ``hop_sec`` defaults to ``window_sec`` (non-overlapping).
    - The final partial window is kept as-is (not padded); callers pad if a model requires
      a fixed length.
    - A signal shorter than one window returns a single chunk containing the whole signal.
    """
    if window_sec <= 0:
        raise ValueError("window_sec must be positive")
    hop_sec = window_sec if hop_sec is None else hop_sec
    if hop_sec <= 0:
        raise ValueError("hop_sec must be positive")

    win = int(round(window_sec * sr))
    hop = int(round(hop_sec * sr))
    if x.size <= win:
        return [x]

    chunks = [x[start:start + win] for start in range(0, x.size - win + 1, hop)]
    # Capture a trailing remainder that the stride skipped past.
    last_start = (x.size - win) // hop * hop
    if last_start + win < x.size:
        chunks.append(x[last_start + hop:])
    return chunks


def preprocess(path: str | Path, validation: Optional[ValidationResult] = None) -> Preprocessed:
    """Decode and prepare one validated file. Raises on decode failure.

    ``validation`` is accepted for pipeline symmetry (callers pass the stage-1 result); the
    decode itself is the source of truth here.
    """
    path = Path(path)
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)

    mono = _to_mono(data)
    features = compute_signal_features(mono, sr)
    waveform_16k = _resample_16k(mono, sr)

    return Preprocessed(
        name=path.name,
        waveform_16k=waveform_16k,
        signal_original=mono,
        original_sample_rate=sr,
        features=features,
    )


def preprocess_batch(paths: list[str | Path]) -> list[Preprocessed | PreprocessError]:
    """Preprocess many files with per-file isolation; never raises."""
    out: list[Preprocessed | PreprocessError] = []
    for p in paths:
        try:
            out.append(preprocess(p))
        except Exception as err:  # noqa: BLE001 - isolate one bad file from the batch
            out.append(PreprocessError(name=Path(p).name, error=str(err)))
    return out


def _main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m app.audio.preprocessing <file> [file ...]")
        return 2

    results = preprocess_batch(list(argv))
    name_w = max((len(r.name) for r in results), default=4)
    header = (
        f"{'FILE':<{name_w}}  {'SR->16k':>9}  {'DUR(s)':>7}  {'PEAK':>6}  "
        f"{'CLIP%':>7}  {'RMS':>7}  {'RMSdB':>7}  {'ZCR':>6}  {'CREST':>6}"
    )
    print(header)
    print("-" * len(header))
    ok = True
    for r in results:
        if isinstance(r, PreprocessError):
            ok = False
            print(f"{r.name:<{name_w}}  ERROR: {r.error}")
            continue
        f = r.features
        print(
            f"{r.name:<{name_w}}  {r.original_sample_rate:>5}->16k  {f.duration_sec:>7.2f}  "
            f"{f.peak:>6.3f}  {f.clipping_ratio * 100:>6.3f}%  {f.rms:>7.4f}  "
            f"{f.rms_db:>7.2f}  {f.zero_crossing_rate:>6.3f}  {f.crest_factor:>6.2f}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
