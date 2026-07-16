"""Tests for energy-based dead-air / silence detection."""

from __future__ import annotations

import numpy as np
import pytest

from app.analysis.framing import frame_rms
from app.analysis.silence import _longest_run, detect_silence

SR = 16000
RNG = np.random.default_rng(0)


def _noise(seconds: float, amp: float) -> np.ndarray:
    return (RNG.standard_normal(int(seconds * SR)) * amp).astype(np.float32)


# --- framing helper -----------------------------------------------------------

def test_frame_rms_basic():
    x = np.ones(1000, dtype=np.float32)
    rms = frame_rms(x, sr=1000, frame_ms=100)  # frame=100 samples, 10 frames
    assert rms.size == 10
    assert np.allclose(rms, 1.0)


def test_frame_rms_short_signal_empty():
    assert frame_rms(np.ones(5, dtype=np.float32), sr=1000, frame_ms=100).size == 0


def test_longest_run():
    assert _longest_run(np.array([False, True, True, False, True])) == 2
    assert _longest_run(np.array([False, False])) == 0
    assert _longest_run(np.array([], dtype=bool)) == 0


# --- silence detection --------------------------------------------------------

def test_detects_long_gap():
    # Gap exceeds LONG_SILENCE_SEC (10s, calibrated against docs/labels.csv).
    sig = np.concatenate([_noise(1, 0.3), _noise(12, 0.0001), _noise(1, 0.3)])
    r = detect_silence(sig, SR)
    assert r.long_silence_present is True
    assert r.longest_silence_sec == pytest.approx(12.0, abs=0.15)


def test_short_gap_not_flagged():
    # A multi-second gap that stays below the calibrated 10s threshold.
    sig = np.concatenate([_noise(1, 0.3), _noise(5, 0.0001), _noise(1, 0.3)])
    r = detect_silence(sig, SR)
    assert r.long_silence_present is False
    assert r.longest_silence_sec == pytest.approx(5.0, abs=0.15)


def test_constant_noise_not_flagged():
    # Uniform-energy noise has no dead air, even though it is "noisy".
    sig = _noise(6, 0.05)
    r = detect_silence(sig, SR)
    assert r.long_silence_present is False
    assert r.longest_silence_sec < 0.5


def test_all_silence():
    sig = np.zeros(int(12 * SR), dtype=np.float32)
    r = detect_silence(sig, SR)
    assert r.long_silence_present is True
    assert r.silence_ratio == 1.0


def test_quiet_call_uses_adaptive_threshold():
    # A quiet recording (low absolute level) with a gap should still be detected,
    # because the threshold is relative to the file's own active level.
    sig = np.concatenate([_noise(1, 0.01), _noise(12, 0.00001), _noise(1, 0.01)])
    r = detect_silence(sig, SR)
    assert r.long_silence_present is True
