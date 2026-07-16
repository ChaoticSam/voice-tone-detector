"""Tests for the audio validation layer.

Covers the three real sample files plus synthetic edge cases (empty, wrong
extension, corrupt bytes, truncated stream) and the per-file isolation contract of
``validate_batch``.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from app.audio.validation import validate_batch, validate_file
from app.config import MIN_DURATION_SEC

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = REPO_ROOT / "audio_files"
REAL_FILES = sorted(AUDIO_DIR.glob("*.ogg"))


def _write_wav(path: Path, seconds: float, sample_rate: int = 16000) -> Path:
    """Write a tiny mono 16-bit PCM WAV of the given duration."""
    n = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))
    return path


# --- real files ---------------------------------------------------------------

@pytest.mark.skipif(not REAL_FILES, reason="sample audio files not present")
@pytest.mark.parametrize("path", REAL_FILES, ids=lambda p: p.name)
def test_real_files_are_valid(path: Path):
    r = validate_file(path)
    assert r.status == "valid"
    assert r.error is None
    assert r.duration_sec and r.duration_sec > 0
    assert r.sample_rate and r.sample_rate > 0
    assert r.channels and r.channels >= 1
    assert r.decoder in ("soundfile", "ffmpeg")


# --- edge cases ---------------------------------------------------------------

def test_missing_file(tmp_path: Path):
    r = validate_file(tmp_path / "nope.wav")
    assert r.status == "invalid"
    assert r.error == "empty_file"


def test_empty_file(tmp_path: Path):
    p = tmp_path / "empty.wav"
    p.touch()
    r = validate_file(p)
    assert r.status == "invalid"
    assert r.error == "empty_file"


def test_unsupported_extension(tmp_path: Path):
    p = tmp_path / "note.xyz"
    p.write_bytes(b"whatever")
    r = validate_file(p)
    assert r.status == "invalid"
    assert r.error == "unsupported_extension"


def test_text_renamed_as_wav_is_decode_failed(tmp_path: Path):
    p = tmp_path / "bad.wav"
    p.write_text("this is not audio, just text pretending to be a wav" * 20)
    r = validate_file(p)
    assert r.status == "invalid"
    assert r.error in ("decode_failed", "no_audio_stream")


def test_truncated_ogg_is_decode_failed(tmp_path: Path):
    if not REAL_FILES:
        pytest.skip("no real ogg to truncate")
    data = REAL_FILES[0].read_bytes()
    p = tmp_path / "truncated.ogg"
    p.write_bytes(data[:2048])  # header-ish only, stream cut off
    r = validate_file(p)
    assert r.status == "invalid"
    assert r.error in ("decode_failed", "no_audio_stream")


def test_valid_generated_wav(tmp_path: Path):
    p = _write_wav(tmp_path / "tone.wav", seconds=1.0)
    r = validate_file(p)
    assert r.status == "valid"
    assert r.sample_rate == 16000
    assert r.channels == 1
    assert abs(r.duration_sec - 1.0) < 0.01


def test_too_short(tmp_path: Path):
    p = _write_wav(tmp_path / "blip.wav", seconds=MIN_DURATION_SEC / 2)
    r = validate_file(p)
    assert r.status == "invalid"
    assert r.error == "too_short"


# --- batch isolation ----------------------------------------------------------

def test_batch_isolates_failures(tmp_path: Path):
    good = _write_wav(tmp_path / "good.wav", seconds=0.5)
    bad = tmp_path / "bad.wav"
    bad.write_text("garbage")
    missing = tmp_path / "gone.wav"

    results = validate_batch([good, bad, missing])
    assert len(results) == 3
    assert results[0].status == "valid"
    assert results[1].status == "invalid"
    assert results[2].status == "invalid"
