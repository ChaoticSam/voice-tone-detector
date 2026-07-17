"""Emotion — dimensional acoustic channel: the "how it was said" signal.

Uses a dimensional SER model (arousal / valence / dominance). We tried fusing this with an
LLM reading the transcript (lexical channel); the LLM lost to a categorical SER model on
tone (see emotion_categorical.py) and was dropped -- see emotion_pipeline.py. This module's
role in the final pipeline: `arousal` (on the WHOLE call) is the sole source of
`emotional_intensity`; `map_dimensions`/`emotional_tone` here are provisional/reference only
(valence was weak/anti-correlated on the real calls -- tone is now decided by the
categorical model).

Mapping rationale (for the provisional tone/intensity this module still computes):
  * intensity  = arousal bands (this is the only source of emotional_intensity).
  * tone       = valence decides positive/neutral/negative; within negative, arousal
                 separates frustrated (lower) < upset (higher) < distressed (highest).
                 Neutral is kept independent of arousal so a loud-but-neutral speaker is
                 not miscalled upset -- respecting the spec's "do not infer frustration/
                 distress solely from loudness."
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from app.audio.preprocessing import Preprocessed, chunk_waveform
from app.config import (
    AROUSAL_DISTRESSED,
    AROUSAL_UPSET,
    EMOTION_MODEL_NAME,
    EMOTION_WINDOW_OVERLAP,
    EMOTION_WINDOW_SEC,
    INTENSITY_HIGH_AROUSAL,
    INTENSITY_MED_AROUSAL,
    VALENCE_NEUTRAL_LO,
    VALENCE_SATISFIED,
)
from app.analysis.emotion_model import load_dimensional_emotion

Tone = Literal["neutral", "satisfied", "frustrated", "upset", "distressed"]
Intensity = Literal["low", "medium", "high"]
_MIN_CHUNK_RMS = 1e-3  # skip near-silent windows when aggregating


class AcousticEmotionResult(BaseModel):
    # Provisional acoustic-only verdict (final verdict comes after 5b lexical fusion).
    emotional_tone: Tone
    emotional_intensity: Intensity
    # Full evidence, exposed so 5b can fuse rather than override blindly.
    arousal: float
    valence: float
    dominance: float
    per_chunk: list[tuple[float, float]]  # (valence, arousal) per analysed window


def _intensity(arousal: float) -> Intensity:
    if arousal >= INTENSITY_HIGH_AROUSAL:
        return "high"
    if arousal >= INTENSITY_MED_AROUSAL:
        return "medium"
    return "low"


def map_dimensions(valence: float, arousal: float) -> tuple[Tone, Intensity]:
    """Map dimensional (valence, arousal) to a 5-class tone + intensity. Pure/testable."""
    intensity = _intensity(arousal)

    if valence >= VALENCE_SATISFIED:
        return "satisfied", intensity
    if valence >= VALENCE_NEUTRAL_LO:
        return "neutral", intensity
    # Negative valence: arousal separates the negative emotions.
    if arousal >= AROUSAL_DISTRESSED:
        return "distressed", intensity
    if arousal >= AROUSAL_UPSET:
        return "upset", intensity
    return "frustrated", intensity


def _infer(model, processor, chunk: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Return (arousal, valence, dominance) for one chunk."""
    inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
    _, logits = model(inputs.input_values)
    arousal, dominance, valence = logits[0].tolist()
    return arousal, valence, dominance


def build_customer_waveform(pre: Preprocessed, ranges: list[tuple[float, float]]) -> np.ndarray:
    """Concatenate the customer's turn time-ranges into a single 16 kHz waveform.

    Used to run the acoustic emotion model on the CUSTOMER only (the AI agent's flat TTS
    voice otherwise dilutes the read). Falls back to the whole clip if ranges are empty.
    """
    if not ranges:
        return pre.waveform_16k
    sr = pre.sample_rate
    segs = [pre.waveform_16k[int(s * sr):int(e * sr)] for s, e in ranges]
    segs = [x for x in segs if x.size]
    return np.concatenate(segs) if segs else pre.waveform_16k


def detect_emotion(
    pre: Preprocessed, waveform: np.ndarray | None = None
) -> AcousticEmotionResult:
    """Acoustic emotion for a clip (provisional; fused with lexical later).

    Pass ``waveform`` to analyse a specific signal (e.g. the customer-only waveform from
    ``build_customer_waveform``) instead of the whole preprocessed clip.
    """
    model, processor = load_dimensional_emotion(EMOTION_MODEL_NAME)
    wave = pre.waveform_16k if waveform is None else waveform

    hop = EMOTION_WINDOW_SEC * (1.0 - EMOTION_WINDOW_OVERLAP)
    chunks = chunk_waveform(wave, pre.sample_rate, EMOTION_WINDOW_SEC, hop_sec=hop)

    rows: list[tuple[float, float, float, float]] = []  # (arousal, valence, dominance, weight)
    for chunk in chunks:
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if rms < _MIN_CHUNK_RMS:
            continue
        a, v, d = _infer(model, processor, chunk, pre.sample_rate)
        rows.append((a, v, d, rms))

    if not rows:  # entire signal near-silent — analyse it whole
        a, v, d = _infer(model, processor, wave, pre.sample_rate)
        rows = [(a, v, d, 1.0)]

    arr = np.array(rows)
    weights = arr[:, 3]
    arousal = float(np.average(arr[:, 0], weights=weights))
    valence = float(np.average(arr[:, 1], weights=weights))
    dominance = float(np.average(arr[:, 2], weights=weights))

    tone, intensity = map_dimensions(valence, arousal)
    return AcousticEmotionResult(
        emotional_tone=tone,
        emotional_intensity=intensity,
        arousal=round(arousal, 4),
        valence=round(valence, 4),
        dominance=round(dominance, 4),
        per_chunk=[(round(v, 3), round(a, 3)) for a, v, d, _ in rows],
    )
