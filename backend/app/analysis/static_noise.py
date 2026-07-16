"""DSP detector for additive broadband noise (static / hiss / crackle).

Complements the AST event classifier. Static is *additive* contamination (a clean signal
plus extra broadband noise) — distinct from *distortion* (the signal itself breaking, e.g.
clipping/"crunch", which is an audio_quality concern). AST, being an event classifier,
has no reliable class for textural broadband noise and tends to misfire (e.g. "Crunch") on
static. This is a signal-level phenomenon, so we detect it with DSP — as we do for clipping
and silence.

Signature of static, measured on the noise-floor frames (quiet, non-speech, non-silence):
  * high spectral flatness in the meaningful band (broadband ~ white noise), and
  * elevated impulsive activity (crackle bursts).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from app.config import (
    STATIC_ACTIVE_PERCENTILE,
    STATIC_FLOOR_HI_FRAC,
    STATIC_FLOOR_LO_FRAC,
    STATIC_FRAME_MS,
    STATIC_MIN_FLOOR_FRAMES,
    STATIC_SEVERITY_HIGH_SFM,
    STATIC_SEVERITY_MEDIUM_SFM,
    STATIC_SFM_BAND_HZ,
    STATIC_SFM_THRESHOLD,
)

Severity = Literal["none", "low", "medium", "high"]
_TRANSIENT_CREST = 6.0  # peak/rms above this marks an impulsive (crackle) frame


class StaticResult(BaseModel):
    present: bool
    sfm_floor: float          # spectral flatness of the noise floor (0=tonal, 1=white)
    transient_rate: float     # fraction of frames that are impulsive (crackle)
    severity: Severity


def _band_sfm(frame: np.ndarray, sr: int, fmax: float) -> float:
    """Spectral flatness (geometric/arithmetic mean of power) up to ``fmax`` Hz."""
    ps = np.abs(np.fft.rfft(frame.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(frame.size, d=1.0 / sr)
    ps = ps[freqs <= fmax] + 1e-12
    return float(np.exp(np.mean(np.log(ps))) / np.mean(ps))


def _severity(sfm: float) -> Severity:
    if sfm >= STATIC_SEVERITY_HIGH_SFM:
        return "high"
    if sfm >= STATIC_SEVERITY_MEDIUM_SFM:
        return "medium"
    return "low"


def detect_static(signal: np.ndarray, sr: int) -> StaticResult:
    """Detect additive broadband static/hiss/crackle in a mono signal."""
    none = StaticResult(present=False, sfm_floor=0.0, transient_rate=0.0, severity="none")
    frame = max(1, int(STATIC_FRAME_MS * sr / 1000.0))
    if signal.size < frame * STATIC_MIN_FLOOR_FRAMES:
        return none

    frames = [signal[i:i + frame] for i in range(0, signal.size - frame + 1, frame)]
    rms = np.array([float(np.sqrt(np.mean(f.astype(np.float64) ** 2))) for f in frames])
    active = float(np.percentile(rms, STATIC_ACTIVE_PERCENTILE))
    if active <= 0:
        return none

    # Impulsiveness across all frames (crackle bursts).
    crest = np.array([np.max(np.abs(f)) / (r + 1e-9) for f, r in zip(frames, rms)])
    transient_rate = float(np.mean(crest > _TRANSIENT_CREST))

    # Noise-floor frames: quiet but not digital silence.
    lo, hi = STATIC_FLOOR_LO_FRAC * active, STATIC_FLOOR_HI_FRAC * active
    floor = [f for f, r in zip(frames, rms) if lo < r < hi]
    if len(floor) < STATIC_MIN_FLOOR_FRAMES:
        return StaticResult(present=False, sfm_floor=0.0, transient_rate=round(transient_rate, 4), severity="none")

    sfm = float(np.mean([_band_sfm(f, sr, STATIC_SFM_BAND_HZ) for f in floor]))
    present = sfm > STATIC_SFM_THRESHOLD
    return StaticResult(
        present=present,
        sfm_floor=round(sfm, 4),
        transient_rate=round(transient_rate, 4),
        severity=_severity(sfm) if present else "none",
    )
