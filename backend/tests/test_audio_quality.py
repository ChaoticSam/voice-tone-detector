"""Tests for rule-based audio-quality assessment."""

from __future__ import annotations

import numpy as np
import pytest

from app.analysis.audio_quality import assess_quality, estimate_snr_db
from app.audio.preprocessing import (
    MODEL_SAMPLE_RATE,
    Preprocessed,
    compute_signal_features,
)

SR = 16000
RNG = np.random.default_rng(1)


def _make_pre(signal: np.ndarray, sr: int = SR) -> Preprocessed:
    """Build a Preprocessed directly from a signal (no file I/O)."""
    signal = signal.astype(np.float32)
    feats = compute_signal_features(signal, sr)
    return Preprocessed(
        name="synthetic",
        waveform_16k=signal if sr == MODEL_SAMPLE_RATE else signal,
        signal_original=signal,
        original_sample_rate=sr,
        features=feats,
    )


def _dynamic(seconds: float, loud_amp: float, quiet_amp: float) -> np.ndarray:
    """Alternating loud/quiet segments -> gives dynamic range (realistic SNR)."""
    seg = []
    t = 0.0
    loud = True
    while t < seconds:
        amp = loud_amp if loud else quiet_amp
        seg.append(RNG.standard_normal(int(0.5 * SR)) * amp)
        loud = not loud
        t += 0.5
    return np.concatenate(seg).astype(np.float32)


# --- SNR estimate -------------------------------------------------------------

def test_snr_high_for_dynamic_signal():
    sig = _dynamic(4, loud_amp=0.3, quiet_amp=0.0005)
    assert estimate_snr_db(sig, SR) > 30


# --- quality verdicts ---------------------------------------------------------

def test_clean_signal_is_clear():
    sig = _dynamic(4, loud_amp=0.3, quiet_amp=0.002)
    r = assess_quality(_make_pre(sig))
    assert r.audio_quality == "clear"


def test_clipping_drives_severe():
    # >1% of samples pinned at full scale.
    sig = _dynamic(3, loud_amp=0.3, quiet_amp=0.01)
    sig[: int(0.05 * sig.size)] = 1.0  # 5% clipped
    r = assess_quality(_make_pre(sig))
    assert r.audio_quality == "severely_impaired"
    assert any("clipping" in reason for reason in r.reasons)


def test_low_volume_flagged():
    sig = _dynamic(4, loud_amp=0.004, quiet_amp=0.00002)  # very quiet overall
    r = assess_quality(_make_pre(sig))
    assert r.audio_quality in ("slightly_impaired", "severely_impaired")
    assert any("volume" in reason for reason in r.reasons)


def test_reasons_present_when_clear():
    sig = _dynamic(4, loud_amp=0.3, quiet_amp=0.002)
    r = assess_quality(_make_pre(sig))
    assert r.reasons  # never empty


# --- real files ---------------------------------------------------------------

def test_real_files_produce_valid_verdicts():
    from pathlib import Path

    from app.audio.preprocessing import preprocess

    audio_dir = Path(__file__).resolve().parents[2] / "audio_files"
    files = sorted(audio_dir.glob("*.ogg"))
    if not files:
        pytest.skip("sample audio not present")
    for p in files:
        r = assess_quality(preprocess(p))
        assert r.audio_quality in ("clear", "slightly_impaired", "severely_impaired")
        assert r.snr_db > 0
