"""Tests for emotion fusion (stage 5b).

The fallback path and output-validation logic are tested without any network/key by
monkeypatching the LLM client. A model-gated calibration test runs the real gpt-4o-mini
fusion against docs/labels.csv when OPENAI_API_KEY (and audio/labels) are present.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from app.analysis import emotion_fusion
from app.analysis.emotion import AcousticEmotionResult
from app.analysis.emotion_fusion import _TONES, fuse_emotion
from app.analysis.llm_client import LLMUnavailable
from app.analysis.transcription import Transcript

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = REPO_ROOT / "docs" / "labels.csv"
AUDIO_DIR = REPO_ROOT / "audio_files"


def _acoustic(tone="neutral", intensity="medium", arousal=0.6, valence=0.5):
    return AcousticEmotionResult(
        emotional_tone=tone, emotional_intensity=intensity,
        arousal=arousal, valence=valence, dominance=0.5, per_chunk=[(valence, arousal)],
    )


def _transcript(text="I've called five times already and nobody has helped."):
    return Transcript(
        text=text, language="en", language_probability=0.99,
        duration_sec=5.0, speaking_rate_wpm=120.0, num_words=len(text.split()),
    )


# --- fusion logic (no network) ------------------------------------------------

def test_fused_uses_llm_verdict(monkeypatch):
    def fake(system, user, **kw):
        return {"emotional_tone": "frustrated", "emotional_intensity": "medium",
                "confidence": 0.8, "rationale": "words show repeated failed contact"}
    monkeypatch.setattr(emotion_fusion, "complete_json", fake)

    r = fuse_emotion(_acoustic(tone="neutral"), _transcript())
    assert r.source == "fused"
    assert r.emotional_tone == "frustrated"   # lexical overrides acoustic 'neutral'
    assert r.confidence == 0.8


def test_fallback_when_llm_unavailable(monkeypatch):
    def boom(system, user, **kw):
        raise LLMUnavailable("no key")
    monkeypatch.setattr(emotion_fusion, "complete_json", boom)

    r = fuse_emotion(_acoustic(tone="upset", intensity="high"), _transcript())
    assert r.source == "acoustic_fallback"
    assert r.emotional_tone == "upset"        # falls back to acoustic verdict
    assert r.emotional_intensity == "high"
    assert r.confidence < 0.6


def test_malformed_output_falls_back(monkeypatch):
    monkeypatch.setattr(emotion_fusion, "complete_json",
                        lambda s, u, **kw: {"emotional_tone": "banana"})
    r = fuse_emotion(_acoustic(tone="satisfied"), _transcript())
    assert r.source == "acoustic_fallback"
    assert r.emotional_tone == "satisfied"


def test_confidence_clamped(monkeypatch):
    monkeypatch.setattr(emotion_fusion, "complete_json",
                        lambda s, u, **kw: {"emotional_tone": "neutral",
                                            "emotional_intensity": "low",
                                            "confidence": 5.0, "rationale": "x"})
    r = fuse_emotion(_acoustic(), _transcript())
    assert r.confidence == 1.0


# --- model-gated calibration --------------------------------------------------

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.skipif(not LABELS_PATH.exists(), reason="labels not present")
@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_fused_runs_and_intensity_matches(name):
    """End-to-end fusion on the real calls.

    Asserts only what is defensible on this 3-example dev set: the pipeline runs and
    produces a valid fused verdict, and `emotional_intensity` (acoustic-driven) matches
    the labels 3/3. `emotional_tone` is deliberately NOT asserted against the labels —
    on these 3 calls the tone labels are ambiguous/counterintuitive (e.g. call_003 is
    labelled "satisfied" despite a repeatedly-declined request), and tuning to make 3
    points pass would be overfitting/leakage. See the memo for the honest tone analysis.
    """
    if not (AUDIO_DIR / name).exists():
        pytest.skip(f"{name} not present")

    from app.analysis.emotion import detect_emotion
    from app.analysis.transcription import transcribe
    from app.audio.preprocessing import preprocess

    with open(LABELS_PATH) as f:
        gt = {r["name"]: json.loads(r["result_json"]) for r in csv.DictReader(f)}

    pre = preprocess(AUDIO_DIR / name)
    fused = fuse_emotion(detect_emotion(pre), transcribe(pre))
    assert fused.source == "fused"
    assert fused.emotional_tone in _TONES
    assert fused.emotional_intensity == gt[name]["emotional_intensity"]
