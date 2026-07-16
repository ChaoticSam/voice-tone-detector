"""Cached loaders for Hugging Face audio models.

Weights are downloaded once from the HF Hub (public model artifacts only — no customer
audio is ever uploaded) and cached on disk under ``~/.cache/huggingface``. These loaders
add an in-process ``lru_cache`` on top so a batch of many files reuses one loaded model
instead of re-instantiating it per clip.

Kept generic on purpose: the emotion stage (next layer) reuses the same pattern.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=4)
def load_ast(model_name: str) -> tuple[Any, Any]:
    """Load an Audio Spectrogram Transformer classifier + its feature extractor.

    Returns ``(model, feature_extractor)``. The model is put in eval mode. Import of
    ``torch``/``transformers`` is deferred to call time so lightweight, model-free code
    paths (validation, DSP) don't pay the heavy import cost.
    """
    import torch
    from transformers import ASTForAudioClassification, AutoFeatureExtractor

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ASTForAudioClassification.from_pretrained(model_name)
    model.eval()
    torch.set_grad_enabled(False)
    return model, feature_extractor
