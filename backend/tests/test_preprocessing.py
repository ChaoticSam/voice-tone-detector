"""Tests for the audio preprocessing layer.

Covers the canonical-waveform transform, the cheap feature pass (grounded in the real
files' measured properties), the chunking utility, and per-file batch isolation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.audio.preprocessing import (
    MODEL_SAMPLE_RATE,
    PreprocessError,
    Preprocessed,
    _resample_16k,
    _to_mono,
    chunk_waveform,
    compute_signal_features,
    preprocess,
    preprocess_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = REPO_ROOT / "audio_files"
REAL_FILES = sorted(AUDIO_DIR.glob("*.ogg"))


# --- real files: reproduce the raw analysis -----------------------------------

@pytest.mark.skipif(not REAL_FILES, reason="sample audio files not present")
def test_real_files_transform_and_features():
    by_name = {}
    for p in REAL_FILES:
        r = preprocess(p)
        assert isinstance(r, Preprocessed)
        assert r.sample_rate == MODEL_SAMPLE_RATE
        assert r.waveform_16k.ndim == 1  # mono
        assert r.waveform_16k.dtype == np.float32
        # length ratio matches 16000 / original_sr
        expected = r.signal_original.size * MODEL_SAMPLE_RATE / r.original_sample_rate
        assert abs(r.waveform_16k.size - expected) < 2
        by_name[r.name] = r

    # Clipping present in call_001 & call_003, absent in call_002 (matches raw analysis).
    assert by_name["call_001.ogg"].features.clipping_ratio > 0
    assert by_name["call_003.ogg"].features.clipping_ratio > 0
    assert by_name["call_002.ogg"].features.clipping_ratio == 0.0

    # call_002 is the quiet file.
    assert by_name["call_002.ogg"].features.rms < by_name["call_001.ogg"].features.rms
    assert by_name["call_002.ogg"].features.rms < by_name["call_003.ogg"].features.rms


# --- mono conversion ----------------------------------------------------------

def test_to_mono_duplicated_stereo_is_lossless():
    ch = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    stereo = np.stack([ch, ch], axis=1)  # identical channels (fake stereo)
    mono = _to_mono(stereo)
    assert mono.shape == (1000,)
    assert np.allclose(mono, ch)


def test_to_mono_passthrough_1d():
    x = np.zeros(10, dtype=np.float32)
    assert _to_mono(x).shape == (10,)


# --- resampling ---------------------------------------------------------------

def test_resample_16k_noop_when_already_16k():
    x = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    out = _resample_16k(x, 16000)
    assert np.array_equal(out, x)


def test_resample_preserves_duration():
    sr = 48000
    x = np.zeros(sr * 2, dtype=np.float32)  # 2 seconds
    out = _resample_16k(x, sr)
    assert abs(out.size - MODEL_SAMPLE_RATE * 2) < 2


# --- features -----------------------------------------------------------------

def test_full_scale_square_wave_is_clipped():
    x = np.ones(1000, dtype=np.float32)
    x[::2] = -1.0
    f = compute_signal_features(x, 16000)
    assert f.clipping_ratio == 1.0
    assert f.peak == pytest.approx(1.0)


def test_half_amplitude_sine_not_clipped():
    t = np.arange(16000, dtype=np.float32) / 16000
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    f = compute_signal_features(x, 16000)
    assert f.clipping_ratio == 0.0
    assert f.peak < 0.51


def test_empty_signal_features_safe():
    f = compute_signal_features(np.array([], dtype=np.float32), 16000)
    assert f.duration_sec == 0.0
    assert f.rms == 0.0


# --- chunking -----------------------------------------------------------------

def test_chunk_non_overlapping_exact():
    x = np.arange(30, dtype=np.float32)
    chunks = chunk_waveform(x, sr=1, window_sec=10)  # win=10, hop=10
    assert len(chunks) == 3
    assert all(c.size == 10 for c in chunks)


def test_chunk_keeps_partial_tail():
    x = np.arange(25, dtype=np.float32)
    chunks = chunk_waveform(x, sr=1, window_sec=10)  # 10,10,5
    assert len(chunks) == 3
    assert chunks[-1].size == 5


def test_chunk_shorter_than_window_returns_whole():
    x = np.arange(5, dtype=np.float32)
    chunks = chunk_waveform(x, sr=1, window_sec=10)
    assert len(chunks) == 1
    assert chunks[0].size == 5


def test_chunk_overlapping():
    x = np.arange(20, dtype=np.float32)
    chunks = chunk_waveform(x, sr=1, window_sec=10, hop_sec=5)  # starts 0,5,10
    assert len(chunks) == 3
    assert np.array_equal(chunks[0], x[0:10])
    assert np.array_equal(chunks[1], x[5:15])


def test_chunk_rejects_bad_args():
    x = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError):
        chunk_waveform(x, sr=1, window_sec=0)
    with pytest.raises(ValueError):
        chunk_waveform(x, sr=1, window_sec=5, hop_sec=0)


# --- batch isolation ----------------------------------------------------------

@pytest.mark.skipif(not REAL_FILES, reason="sample audio files not present")
def test_batch_isolates_failures(tmp_path: Path):
    bad = tmp_path / "bad.wav"
    bad.write_text("not audio")
    results = preprocess_batch([REAL_FILES[0], bad])
    assert isinstance(results[0], Preprocessed)
    assert isinstance(results[1], PreprocessError)
    assert results[1].name == "bad.wav"
