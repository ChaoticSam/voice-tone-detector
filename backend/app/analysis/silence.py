"""Silence / dead-air detection — produces ``long_silence_present``.

Design (discussed): ``long_silence_present`` is defined by the spec as *"an unusually long
period of silence or dead air that may indicate a call-flow or audio problem."* That is a
signal-energy phenomenon, not a speech-vs-non-speech problem, so we use energy-based
dead-air detection rather than a VAD:

  * frame the signal and take per-frame RMS
  * derive a threshold **relative to the file's own active level** (so quiet calls are
    handled without an absolute-volume assumption)
  * a frame below the threshold is "silent"; the longest consecutive silent run is the
    longest dead-air stretch

A call with constant background noise (office chatter) never drops to dead air, so it is
correctly *not* flagged — which is what we want for a call-flow/audio-problem signal.

Measured on the ORIGINAL-rate signal (evidence preserved; no trimming upstream).
A proper VAD (Silero) is introduced later where true speech regions are needed
(speaker overlap, speaking rate) — deliberately not here.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from app.analysis.framing import frame_rms
from app.config import (
    LONG_SILENCE_SEC,
    SILENCE_ACTIVE_PERCENTILE,
    SILENCE_DROP_DB,
    SILENCE_FRAME_MS,
)


class SilenceResult(BaseModel):
    long_silence_present: bool
    longest_silence_sec: float
    silence_ratio: float          # fraction of frames that are silent


def _longest_run(mask: np.ndarray) -> int:
    """Length (in frames) of the longest consecutive True run in a boolean array."""
    if mask.size == 0 or not mask.any():
        return 0
    # Reset a running counter to 0 at every False; the max counter is the longest run.
    best = run = 0
    for silent in mask:
        run = run + 1 if silent else 0
        if run > best:
            best = run
    return best


def detect_silence(
    signal: np.ndarray,
    sr: int,
    *,
    frame_ms: float = SILENCE_FRAME_MS,
    drop_db: float = SILENCE_DROP_DB,
    long_silence_sec: float = LONG_SILENCE_SEC,
) -> SilenceResult:
    """Detect long dead-air in a mono signal."""
    rms = frame_rms(signal, sr, frame_ms)
    if rms.size == 0:
        return SilenceResult(long_silence_present=False, longest_silence_sec=0.0, silence_ratio=0.0)

    # Active level = high-percentile frame energy (robust to brief spikes).
    active = float(np.percentile(rms, SILENCE_ACTIVE_PERCENTILE))
    if active <= 0:
        # Entire signal is digital silence.
        total_sec = rms.size * frame_ms / 1000.0
        return SilenceResult(
            long_silence_present=total_sec >= long_silence_sec,
            longest_silence_sec=round(total_sec, 3),
            silence_ratio=1.0,
        )

    threshold = active * (10.0 ** (-drop_db / 20.0))
    silent = rms < threshold

    frame_sec = frame_ms / 1000.0
    longest_sec = _longest_run(silent) * frame_sec
    return SilenceResult(
        long_silence_present=longest_sec >= long_silence_sec,
        longest_silence_sec=round(longest_sec, 3),
        silence_ratio=round(float(silent.mean()), 4),
    )
