"""Tests for background-noise detection.

Split into fast, model-free taxonomy/logic tests and a model-gated calibration test that
runs the real AST model against docs/labels.csv (skips cleanly if the model can't be
loaded, e.g. offline CI, or if the confidential audio/labels are absent).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.analysis.audioset_taxonomy import is_excluded, noise_type_for
from app.analysis.noise import _severity_for

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = REPO_ROOT / "docs" / "labels.csv"
AUDIO_DIR = REPO_ROOT / "audio_files"


# --- taxonomy (fast, no model) ------------------------------------------------

@pytest.mark.parametrize("label", [
    "Speech", "Male speech, man speaking", "Conversation",   # speech
    "Silence", "Inside, small room",                          # non-noise
    "Sidetone", "Dial tone", "Busy signal", "Telephone",      # telephony artifacts
])
def test_excluded_labels(label: str):
    assert is_excluded(label) is True


@pytest.mark.parametrize("label", ["Television", "Music", "Crunch", "Typing", "Wind"])
def test_noise_labels_not_excluded(label: str):
    assert is_excluded(label) is False


def test_noise_type_mapping_and_fallback():
    assert noise_type_for("Television") == "television"
    assert noise_type_for("Computer keyboard") == "keyboard typing"
    assert noise_type_for("Hubbub, speech noise, speech babble") == "office chatter"
    # Uncurated label falls back to a lowercased string, not a crash / drop.
    assert noise_type_for("Didgeridoo") == "didgeridoo"


def test_severity_bands():
    assert _severity_for(0.05) == "low"
    assert _severity_for(0.20) == "medium"
    assert _severity_for(0.80) == "high"


# --- model-gated calibration --------------------------------------------------

def _load_ast_or_skip():
    try:
        from app.analysis.model_cache import load_ast
        from app.config import AST_MODEL_NAME

        load_ast(AST_MODEL_NAME)
    except Exception as err:  # noqa: BLE001 - offline / missing weights -> skip, not fail
        pytest.skip(f"AST model unavailable: {err}")


@pytest.mark.skipif(not LABELS_PATH.exists(), reason="docs/labels.csv not present")
@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_noise_present_matches_labels(name: str):
    if not (AUDIO_DIR / name).exists():
        pytest.skip(f"{name} not present")
    _load_ast_or_skip()

    from app.analysis.noise import detect_noise
    from app.audio.preprocessing import preprocess

    with open(LABELS_PATH) as f:
        gt = {r["name"]: json.loads(r["result_json"]) for r in csv.DictReader(f)}

    result = detect_noise(preprocess(AUDIO_DIR / name))
    # Presence is the reliable, calibrated field.
    assert result.background_noise_present == gt[name]["background_noise_present"]
