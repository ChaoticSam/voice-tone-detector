"""Tests for speaker diarization + overlap.

Pure-logic tests (overlap math, role/customer extraction) run with no model. A model-gated
end-to-end test runs real WhisperX+pyannote when HF_TOKEN + audio + labels are present and
the pyannote terms are accepted (skips otherwise).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from app.analysis.diarization import DiarizationResult, Turn, _compute_overlap

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = REPO_ROOT / "docs" / "labels.csv"
AUDIO_DIR = REPO_ROOT / "audio_files"


# --- overlap math (no model) --------------------------------------------------

def test_overlap_detected_between_speakers():
    rows = [(0.0, 5.0, "A"), (4.0, 8.0, "B")]  # 1s overlap
    present, secs = _compute_overlap(rows)
    assert present is True
    assert secs == pytest.approx(1.0, abs=0.01)


def test_no_overlap_same_speaker_ignored():
    rows = [(0.0, 5.0, "A"), (4.0, 8.0, "A")]  # same speaker -> not overlap
    present, secs = _compute_overlap(rows)
    assert present is False
    assert secs == 0.0


def test_no_overlap_when_sequential():
    rows = [(0.0, 3.0, "A"), (3.1, 6.0, "B")]
    assert _compute_overlap(rows) == (False, 0.0)


def test_tiny_overlap_below_tolerance_ignored():
    rows = [(0.0, 5.0, "A"), (4.95, 8.0, "B")]  # 0.05s < 0.2s tolerance
    assert _compute_overlap(rows)[0] is False


# --- result shape -------------------------------------------------------------

def test_result_model_defaults():
    r = DiarizationResult(turns=[Turn(start=0, end=1, speaker="S", text="hi")], speakers=["S"])
    assert r.speaker_overlap_present is False
    assert r.customer_ranges == []


# --- model-gated end-to-end ---------------------------------------------------

def _diar_available():
    if not os.getenv("HF_TOKEN"):
        return False
    try:
        import whisperx  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _diar_available(), reason="HF_TOKEN / whisperx not available")
@pytest.mark.skipif(not LABELS_PATH.exists(), reason="labels not present")
@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_overlap_matches_labels(name):
    if not (AUDIO_DIR / name).exists():
        pytest.skip(f"{name} not present")
    from app.analysis.diarization import DiarizationUnavailable, diarize
    from app.audio.preprocessing import preprocess

    try:
        d = diarize(preprocess(AUDIO_DIR / name))
    except DiarizationUnavailable as err:
        pytest.skip(f"diarization unavailable (accept pyannote terms?): {err}")

    with open(LABELS_PATH) as f:
        gt = {r["name"]: json.loads(r["result_json"]) for r in csv.DictReader(f)}
    assert d.speaker_overlap_present == gt[name]["speaker_overlap_present"]
