"""Dimensional speech-emotion model (audEERING wav2vec2, arousal/valence/dominance).

The model requires a small amount of custom modeling code (from its HF model card): a
regression head on top of pooled wav2vec2 hidden states. It is NOT loadable via the stock
`AutoModelForAudioClassification` path, but it also does NOT require `trust_remote_code` —
we define the classes here explicitly.

Output logits are ordered [arousal, dominance, valence], each ~0..1.

License note: the checkpoint is CC-BY-NC-SA-4.0 (non-commercial). Disclosed as a
production caveat; the modular loader makes it a one-file swap.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


def _build_classes():
    """Define the custom head + model classes lazily (deferred torch/transformers import)."""
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        """Regression head mapping pooled hidden states -> 3 emotion dimensions."""

        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features):
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        """wav2vec2 with a regression head predicting arousal/dominance/valence."""

        # transformers 5.x consults these during construction; the checkpoint is dense
        # (no tied weights) and we load a full state dict, so no weight init is needed.
        _tied_weights_keys: list[str] = []

        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            # deliberately no init_weights(): we load pretrained weights immediately after

        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            hidden = outputs[0].mean(dim=1)          # mean-pool over time
            logits = self.classifier(hidden)
            return hidden, logits

    return EmotionModel


@lru_cache(maxsize=2)
def load_dimensional_emotion(model_name: str) -> tuple[Any, Any]:
    """Load the dimensional emotion model + its processor (cached across a batch).

    Weights are loaded manually (build-from-config + ``load_state_dict``) rather than via
    ``from_pretrained``: the card's custom class predates transformers 5.x's meta-device /
    tied-weights loading path, which errors on it. Manual loading is version-robust.

    Returns ``(model, processor)``. Model is in eval mode with grad disabled.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2Config, Wav2Vec2Processor

    EmotionModel = _build_classes()
    config = Wav2Vec2Config.from_pretrained(model_name)
    config.num_labels = 3  # arousal, dominance, valence
    model = EmotionModel(config)

    try:
        from safetensors.torch import load_file

        state = load_file(hf_hub_download(model_name, "model.safetensors"))
    except Exception:
        state = torch.load(
            hf_hub_download(model_name, "pytorch_model.bin"), map_location="cpu"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The regression head + wav2vec2 backbone must all be present; only buffers may differ.
    critical = [k for k in missing if k.startswith(("classifier.", "wav2vec2.encoder", "wav2vec2.feature"))]
    if critical:
        raise RuntimeError(f"missing critical weights: {critical[:5]}")

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model.eval()
    torch.set_grad_enabled(False)
    return model, processor
