"""Emotion — categorical SER channel: a second acoustic vote for TONE.

The dimensional model (emotion.py) is reliable for intensity (arousal) but its valence is
weak/compressed on real calls -- see emotion.py's module docstring. A categorical SER model
(discrete classes: neutral/happy/angry/sad, trained on IEMOCAP) gives a second, independent
acoustic read on tone.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from app.config import AROUSAL_DISTRESSED, AROUSAL_UPSET, CATEGORICAL_SER_MODEL_NAME

CategoricalLabel = Literal["neu", "hap", "ang", "sad"]


class CategoricalEmotionResult(BaseModel):
    label: CategoricalLabel          # argmax class
    probs: dict[str, float]          # all class probabilities
    tone: str                        # mapped onto our 5-tone taxonomy (uses arousal for ang/sad)


@lru_cache(maxsize=1)
def load_categorical_emotion() -> tuple[Any, Any]:
    """Load the categorical SER model + feature extractor (cached across a batch)."""
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    model = AutoModelForAudioClassification.from_pretrained(CATEGORICAL_SER_MODEL_NAME)
    fe = AutoFeatureExtractor.from_pretrained(CATEGORICAL_SER_MODEL_NAME)
    model.eval()
    return model, fe


def map_categorical(label: CategoricalLabel, arousal: float) -> str:
    """Map a categorical class + arousal onto the 5-tone taxonomy.

    'ang'/'sad' are ambiguous about intensity of negativity (frustrated vs upset vs
    distressed); arousal disambiguates, reusing the existing bands rather than new ones.
    """
    if label == "neu":
        return "neutral"
    if label == "hap":
        return "satisfied"
    # ang / sad: negative, arousal decides how severe.
    if arousal >= AROUSAL_DISTRESSED:
        return "distressed"
    if arousal >= AROUSAL_UPSET:
        return "upset"
    return "frustrated"


def detect_categorical_emotion(waveform: np.ndarray, sample_rate: int, arousal: float) -> CategoricalEmotionResult:
    """Run the categorical SER model on `waveform`, mapping to our tone taxonomy via `arousal`."""
    import torch

    model, fe = load_categorical_emotion()
    inputs = fe(waveform, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    id2label = model.config.id2label
    prob_map = {id2label[i]: round(p, 4) for i, p in enumerate(probs)}
    label = max(prob_map, key=prob_map.get)
    return CategoricalEmotionResult(
        label=label,
        probs=prob_map,
        tone=map_categorical(label, arousal),
    )
