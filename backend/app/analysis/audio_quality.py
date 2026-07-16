"""Audio quality assessment — produces ``audio_quality``.

Classifies overall technical quality into ``clear | slightly_impaired |
severely_impaired`` from deterministic acoustic signals. The spec lists many possible
defects (clipping, static, low volume, muffled, echo, robotic, packet loss); we detect the
ones that are reliably measurable with cheap DSP and deliberately do NOT guess at the rest:

  * clipping    — clipping_ratio from preprocessing (fraction of samples at full scale)
  * static/noise — a rough percentile-based SNR estimate (active vs noise-floor energy)
  * low volume  — rms_db (overall loudness)

Final quality = the worst-case across these signals. Each contributing signal is recorded
in ``reasons`` for debuggability / the feature store.

Deliberately NOT used as an impairment trigger: high-frequency roll-off. Production call
audio is routinely band-limited (~3.4 kHz telephony), so a low high-frequency ratio is
normal, not a defect — penalizing it would mislabel clean phone calls. We still report
``spectral_centroid_hz`` as a diagnostic.

NOTE: thresholds are principled but UNCALIBRATED — no ground-truth labels available yet.
Measured on the ORIGINAL-rate signal so band-limiting / high-freq artifacts remain visible.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from app.analysis.framing import frame_rms
from app.audio.preprocessing import Preprocessed
from app.config import (
    CLIP_SEVERE_RATIO,
    CLIP_SLIGHT_RATIO,
    SNR_ACTIVE_PERCENTILE,
    SNR_NOISE_PERCENTILE,
    SNR_SEVERE_DB,
    SNR_SLIGHT_DB,
    VOL_SEVERE_DBFS,
    VOL_SLIGHT_DBFS,
)

QualityLabel = Literal["clear", "slightly_impaired", "severely_impaired"]

_EPS = 1e-12
# Severity ordering for worst-case combination.
_RANK: dict[QualityLabel, int] = {"clear": 0, "slightly_impaired": 1, "severely_impaired": 2}
_LABELS: tuple[QualityLabel, ...] = ("clear", "slightly_impaired", "severely_impaired")


class AudioQualityResult(BaseModel):
    audio_quality: QualityLabel
    snr_db: float
    spectral_centroid_hz: float     # diagnostic only (not an impairment trigger)
    reasons: list[str]              # which signals drove the verdict


def estimate_snr_db(signal: np.ndarray, sr: int) -> float:
    """Rough SNR: ratio of high-percentile (active) to low-percentile (noise-floor) frame RMS."""
    rms = frame_rms(signal, sr, frame_ms=25)
    if rms.size == 0:
        return 0.0
    noise = float(np.percentile(rms, SNR_NOISE_PERCENTILE))
    active = float(np.percentile(rms, SNR_ACTIVE_PERCENTILE))
    return float(20.0 * np.log10((active + _EPS) / (noise + _EPS)))


def spectral_centroid_hz(signal: np.ndarray, sr: int) -> float:
    """Energy-weighted mean frequency of the magnitude spectrum (whole-signal)."""
    if signal.size == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(signal.astype(np.float64)))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sr)
    total = spec.sum()
    if total <= 0:
        return 0.0
    return float((freqs * spec).sum() / total)


def _worst(*labels: QualityLabel) -> QualityLabel:
    return _LABELS[max(_RANK[l] for l in labels)]


def assess_quality(pre: Preprocessed) -> AudioQualityResult:
    """Assess technical audio quality from a preprocessed package."""
    signal = pre.signal_original
    sr = pre.original_sample_rate
    clip = pre.features.clipping_ratio
    rms_db = pre.features.rms_db

    snr = estimate_snr_db(signal, sr)
    centroid = spectral_centroid_hz(signal, sr)

    verdict: QualityLabel = "clear"
    reasons: list[str] = []

    # Clipping.
    if clip >= CLIP_SEVERE_RATIO:
        verdict = _worst(verdict, "severely_impaired")
        reasons.append(f"clipping {clip:.3%} >= {CLIP_SEVERE_RATIO:.1%}")
    elif clip >= CLIP_SLIGHT_RATIO:
        verdict = _worst(verdict, "slightly_impaired")
        reasons.append(f"clipping {clip:.3%} >= {CLIP_SLIGHT_RATIO:.1%}")

    # SNR / static.
    if snr < SNR_SEVERE_DB:
        verdict = _worst(verdict, "severely_impaired")
        reasons.append(f"low SNR {snr:.1f}dB < {SNR_SEVERE_DB}dB")
    elif snr < SNR_SLIGHT_DB:
        verdict = _worst(verdict, "slightly_impaired")
        reasons.append(f"low SNR {snr:.1f}dB < {SNR_SLIGHT_DB}dB")

    # Loudness.
    if rms_db < VOL_SEVERE_DBFS:
        verdict = _worst(verdict, "severely_impaired")
        reasons.append(f"very low volume {rms_db:.1f}dBFS < {VOL_SEVERE_DBFS}dBFS")
    elif rms_db < VOL_SLIGHT_DBFS:
        verdict = _worst(verdict, "slightly_impaired")
        reasons.append(f"low volume {rms_db:.1f}dBFS < {VOL_SLIGHT_DBFS}dBFS")

    if not reasons:
        reasons.append("no impairment signals above threshold")

    return AudioQualityResult(
        audio_quality=verdict,
        snr_db=round(snr, 2),
        spectral_centroid_hz=round(centroid, 1),
        reasons=reasons,
    )
