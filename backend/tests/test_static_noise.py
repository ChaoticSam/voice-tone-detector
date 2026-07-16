"""Tests for the DSP static / broadband-noise detector."""

from __future__ import annotations

import numpy as np

from app.analysis.static_noise import _band_sfm, detect_static

SR = 48000
RNG = np.random.default_rng(0)


# --- core discriminator: spectral flatness ------------------------------------

def test_band_sfm_white_is_flat():
    white = RNG.standard_normal(2400).astype(np.float32)
    assert _band_sfm(white, SR, 8000) > 0.3


def test_band_sfm_tone_is_peaky():
    t = np.arange(2400) / SR
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    assert _band_sfm(tone, SR, 8000) < 0.05


# --- detector on constructed signals ------------------------------------------

def _signal_with_floor(floor_fn) -> np.ndarray:
    """Alternating loud (speech-like) and quiet (floor) 0.5s segments."""
    segs = []
    for i in range(10):
        if i % 2 == 0:
            segs.append((RNG.standard_normal(int(0.5 * SR)) * 0.3).astype(np.float32))
        else:
            segs.append(floor_fn(int(0.5 * SR)))
    return np.concatenate(segs)


def test_broadband_floor_is_static():
    # Quiet gaps filled with low-level white noise -> additive static.
    sig = _signal_with_floor(lambda n: (RNG.standard_normal(n) * 0.006).astype(np.float32))
    r = detect_static(sig, SR)
    assert r.present is True
    assert r.severity in ("low", "medium", "high")


def test_tonal_floor_is_not_static():
    def tonal(n):
        t = np.arange(n) / SR
        return (0.006 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    sig = _signal_with_floor(tonal)
    assert detect_static(sig, SR).present is False


def test_too_short_returns_absent():
    assert detect_static(np.zeros(100, dtype=np.float32), SR).present is False
