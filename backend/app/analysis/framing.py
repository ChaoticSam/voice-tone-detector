"""Small shared framing helpers for the DSP analysis services."""

from __future__ import annotations

import numpy as np


def frame_rms(signal: np.ndarray, sr: int, frame_ms: float, hop_ms: float | None = None) -> np.ndarray:
    """Per-frame RMS energy of a mono signal.

    Non-overlapping by default (``hop_ms`` == ``frame_ms``). Returns an empty array for a
    signal shorter than one frame. Computed in float64 for numerical stability.
    """
    if signal.size == 0:
        return np.empty(0, dtype=np.float64)

    frame = max(1, int(round(frame_ms * sr / 1000.0)))
    hop = frame if hop_ms is None else max(1, int(round(hop_ms * sr / 1000.0)))
    if signal.size < frame:
        return np.empty(0, dtype=np.float64)

    x = signal.astype(np.float64, copy=False)
    starts = range(0, x.size - frame + 1, hop)
    return np.array([np.sqrt(np.mean(np.square(x[s:s + frame]))) for s in starts])
