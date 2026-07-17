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
