"""Background noise detection via the Audio Spectrogram Transformer (AudioSet).

Produces ``background_noise_present``, ``background_noise_type``,
``background_noise_severity``. Acoustic-only (no transcript), as the spec requires.

Approach:
  1. Chunk the 16 kHz waveform into ~10s windows (AST is a fixed-length-input model).
  2. Run AST per window -> per-class probabilities. AudioSet is MULTI-label, so a sigmoid
     (not softmax) is applied to the logits.
  3. Max-aggregate probabilities across windows per class (a noise event in even one
     window should register rather than being averaged away).
  4. Exclude speech / non-noise classes so the primary speaker is never counted as noise.
  5. Rank the remainder; the top class drives present / type / severity.

The spec caveat "do not infer background noise solely from poor audio quality" is honoured
structurally: this decision comes from acoustic event classification, independent of the
audio_quality DSP path.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel

from app.analysis.audioset_taxonomy import is_excluded, noise_type_for
from app.analysis.model_cache import load_ast
from app.audio.preprocessing import Preprocessed, chunk_waveform
from app.config import (
    AST_MODEL_NAME,
    NOISE_PRESENT_THRESHOLD,
    NOISE_SEVERITY_HIGH_PROB,
    NOISE_SEVERITY_MEDIUM_PROB,
    NOISE_WINDOW_SEC,
)

Severity = Literal["none", "low", "medium", "high"]


class NoiseResult(BaseModel):
    background_noise_present: bool
    background_noise_type: str                    # "" when not present
    background_noise_severity: Severity
    top_classes: list[tuple[str, float]]          # top non-excluded (label, prob) for debug


def _max_class_probs(pre: Preprocessed) -> dict[str, float]:
    """Max sigmoid probability per AudioSet class across all ~10s windows."""
    import torch

    model, feature_extractor = load_ast(AST_MODEL_NAME)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    labels = [id2label[i] for i in range(len(id2label))]

    chunks = chunk_waveform(pre.waveform_16k, pre.sample_rate, window_sec=NOISE_WINDOW_SEC)
    max_probs = np.zeros(len(labels), dtype=np.float64)
    for chunk in chunks:
        inputs = feature_extractor(
            chunk, sampling_rate=pre.sample_rate, return_tensors="pt"
        )
        logits = model(**inputs).logits[0]
        probs = torch.sigmoid(logits).cpu().numpy()
        max_probs = np.maximum(max_probs, probs)

    return {labels[i]: float(max_probs[i]) for i in range(len(labels))}


def _severity_for(prob: float) -> Severity:
    if prob >= NOISE_SEVERITY_HIGH_PROB:
        return "high"
    if prob >= NOISE_SEVERITY_MEDIUM_PROB:
        return "medium"
    return "low"


def detect_noise(pre: Preprocessed) -> NoiseResult:
    """Detect background noise in a preprocessed clip."""
    probs = _max_class_probs(pre)
    ranked = sorted(
        ((label, p) for label, p in probs.items() if not is_excluded(label)),
        key=lambda lp: lp[1],
        reverse=True,
    )
    top = ranked[:5]

    if not ranked or ranked[0][1] < NOISE_PRESENT_THRESHOLD:
        return NoiseResult(
            background_noise_present=False,
            background_noise_type="",
            background_noise_severity="none",
            top_classes=top,
        )

    top_label, top_prob = ranked[0]
    return NoiseResult(
        background_noise_present=True,
        background_noise_type=noise_type_for(top_label),
        background_noise_severity=_severity_for(top_prob),
        top_classes=top,
    )
