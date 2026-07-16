"""Tests for acoustic emotion (stage 5a).

Fast, model-free unit tests for the pure map_dimensions logic, plus a model-gated
calibration test against docs/labels.csv (skips if model/audio/labels unavailable).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.analysis.emotion import map_dimensions
from app.config import (
    AROUSAL_DISTRESSED,
    AROUSAL_UPSET,
    INTENSITY_HIGH_AROUSAL,
    INTENSITY_MED_AROUSAL,
    VALENCE_NEUTRAL_LO,
    VALENCE_SATISFIED,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = REPO_ROOT / "docs" / "labels.csv"
AUDIO_DIR = REPO_ROOT / "audio_files"


# --- pure mapping logic -------------------------------------------------------

def test_high_valence_is_satisfied():
    tone, _ = map_dimensions(valence=VALENCE_SATISFIED + 0.1, arousal=0.5)
    assert tone == "satisfied"


def test_mid_valence_is_neutral():
    mid = (VALENCE_NEUTRAL_LO + VALENCE_SATISFIED) / 2
    tone, _ = map_dimensions(valence=mid, arousal=0.5)
    assert tone == "neutral"


def test_neutral_independent_of_loudness():
    # Mid valence + very high arousal must stay neutral (not upset) — loudness caveat.
    mid = (VALENCE_NEUTRAL_LO + VALENCE_SATISFIED) / 2
    tone, intensity = map_dimensions(valence=mid, arousal=0.95)
    assert tone == "neutral"
    assert intensity == "high"


def test_negative_low_arousal_is_frustrated():
    tone, _ = map_dimensions(valence=VALENCE_NEUTRAL_LO - 0.1, arousal=AROUSAL_UPSET - 0.05)
    assert tone == "frustrated"


def test_negative_high_arousal_is_upset():
    tone, _ = map_dimensions(valence=VALENCE_NEUTRAL_LO - 0.1, arousal=AROUSAL_UPSET + 0.02)
    assert tone == "upset"


def test_negative_very_high_arousal_is_distressed():
    tone, _ = map_dimensions(valence=VALENCE_NEUTRAL_LO - 0.1, arousal=AROUSAL_DISTRESSED + 0.02)
    assert tone == "distressed"


def test_intensity_bands():
    _, i_lo = map_dimensions(0.5, INTENSITY_MED_AROUSAL - 0.05)
    _, i_md = map_dimensions(0.5, INTENSITY_MED_AROUSAL + 0.01)
    _, i_hi = map_dimensions(0.5, INTENSITY_HIGH_AROUSAL + 0.01)
    assert (i_lo, i_md, i_hi) == ("low", "medium", "high")


# --- model-gated calibration --------------------------------------------------

def _load_model_or_skip():
    try:
        from app.analysis.emotion_model import load_dimensional_emotion
        from app.config import EMOTION_MODEL_NAME

        load_dimensional_emotion(EMOTION_MODEL_NAME)
    except Exception as err:  # noqa: BLE001 - offline / missing weights -> skip
        pytest.skip(f"emotion model unavailable: {err}")


@pytest.mark.skipif(not LABELS_PATH.exists(), reason="docs/labels.csv not present")
@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_acoustic_emotion_calibration(name: str):
    """What the ACOUSTIC channel can honestly deliver on the 3 labels.

    Assertions are set after the calibration run (see below). Cases the acoustic channel
    provably cannot reach without the lexical channel (5b) are documented, not force-passed.
    """
    if not (AUDIO_DIR / name).exists():
        pytest.skip(f"{name} not present")
    _load_model_or_skip()

    from app.analysis.emotion import detect_emotion
    from app.audio.preprocessing import preprocess

    with open(LABELS_PATH) as f:
        gt = {r["name"]: json.loads(r["result_json"]) for r in csv.DictReader(f)}

    result = detect_emotion(preprocess(AUDIO_DIR / name))
    # Provisional: assert intensity (the field the acoustic channel owns). Tone assertions
    # are added/relaxed per the calibration findings below.
    assert result.emotional_intensity == gt[name]["emotional_intensity"]
