"""Assemble the final structured result: the exact JSON schema the trial requires.

One call into `build_result(pre)` runs every analysis stage and returns the combined
9-field object matching docs/labels.csv's schema (emotional_tone, emotional_intensity,
background_noise_present/type/severity, audio_quality, speaker_overlap_present,
long_silence_present, confidence).
"""

from __future__ import annotations

from pydantic import BaseModel

from app.analysis.audio_quality import QualityLabel, assess_quality
from app.analysis.emotion import Intensity, Tone
from app.analysis.emotion_pipeline import analyze_customer_emotion
from app.analysis.noise import Severity, detect_noise
from app.analysis.silence import detect_silence
from app.audio.preprocessing import Preprocessed


class StructuredResult(BaseModel):
    emotional_tone: Tone
    emotional_intensity: Intensity
    background_noise_present: bool
    background_noise_type: str
    background_noise_severity: Severity
    audio_quality: QualityLabel
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float


def build_result(pre: Preprocessed) -> StructuredResult:
    """Run every analysis stage on a preprocessed clip and assemble the final schema."""
    emo = analyze_customer_emotion(pre)
    noise = detect_noise(pre)
    quality = assess_quality(pre)
    silence = detect_silence(pre.signal_original, pre.original_sample_rate)

    # Confidence: top-class probability from the categorical SER model (the tone signal) --
    # already computed, no extra cost, and it is the one genuinely probabilistic output in
    # the pipeline (the acoustic-dimensional and DSP stages are deterministic thresholds).
    confidence = emo.categorical.probs[emo.categorical.label]

    return StructuredResult(
        emotional_tone=emo.emotional_tone,
        emotional_intensity=emo.emotional_intensity,
        background_noise_present=noise.background_noise_present,
        background_noise_type=noise.background_noise_type,
        background_noise_severity=noise.background_noise_severity,
        audio_quality=quality.audio_quality,
        speaker_overlap_present=emo.speaker_overlap_present,
        long_silence_present=silence.long_silence_present,
        confidence=round(confidence, 3),
    )
