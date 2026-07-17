"""Customer-only emotion — final pipeline (acoustic only, no LLM).

Two acoustic signals, both self-hosted, no LLM in the loop:
  * TONE: a categorical SER model (emotion_categorical.py) on the CUSTOMER-only audio.
    Its argmax alone matched tone 3/3 on the labeled calls, ahead of both the dimensional
    acoustic model (weak/anti-correlated valence on real calls) and an LLM lexical-fusion
    channel we tried and dropped (1/3, and dependent on transcript quality that was
    unreliable on a non-English call). ang/sad are disambiguated into
    frustrated/upset/distressed using the customer-only arousal band (see
    emotion_categorical.map_categorical).
  * INTENSITY: whole-call arousal (dimensional model, emotion.py) — matched labels 3/3.
    Customer-only arousal collapsed two labeled calls to near-identical values; whole-call
    arousal (its calibration domain) separates them.

Diarization is kept ONLY for: isolating the customer's audio (for the tone model above) and `speaker_overlap_present`.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.analysis.diarization import DiarizationUnavailable, diarize
from app.analysis.emotion import AcousticEmotionResult, Intensity, Tone, build_customer_waveform, detect_emotion
from app.analysis.emotion_categorical import CategoricalEmotionResult, detect_categorical_emotion
from app.audio.preprocessing import Preprocessed


class CustomerEmotionResult(BaseModel):
    emotional_tone: Tone                       # from categorical SER (customer-only audio)
    emotional_intensity: Intensity             # from dimensional model (whole-call arousal)
    categorical: CategoricalEmotionResult
    acoustic_customer: AcousticEmotionResult   # dimensional pass, customer-only (arousal used for tone disambiguation)
    acoustic_whole: AcousticEmotionResult      # dimensional pass, whole-call (source of intensity)
    speaker_overlap_present: bool
    customer_isolated: bool                    # False if we fell back to whole-clip


def analyze_customer_emotion(pre: Preprocessed) -> CustomerEmotionResult:
    """Diarize -> isolate the customer -> categorical tone + whole-call intensity."""
    try:
        d = diarize(pre)
        customer_wave = build_customer_waveform(pre, d.customer_ranges)
        acoustic_customer = detect_emotion(pre, waveform=customer_wave)
        acoustic_whole = detect_emotion(pre)
        categorical = detect_categorical_emotion(
            customer_wave, pre.sample_rate, arousal=acoustic_customer.arousal
        )
        return CustomerEmotionResult(
            emotional_tone=categorical.tone,
            emotional_intensity=acoustic_whole.emotional_intensity,
            categorical=categorical,
            acoustic_customer=acoustic_customer,
            acoustic_whole=acoustic_whole,
            speaker_overlap_present=d.speaker_overlap_present,
            customer_isolated=bool(d.customer_ranges),
        )
    except DiarizationUnavailable:
        # Fall back to whole-clip acoustic analysis (no customer isolation possible).
        acoustic_whole = detect_emotion(pre)
        categorical = detect_categorical_emotion(
            pre.waveform_16k, pre.sample_rate, arousal=acoustic_whole.arousal
        )
        return CustomerEmotionResult(
            emotional_tone=categorical.tone,
            emotional_intensity=acoustic_whole.emotional_intensity,
            categorical=categorical,
            acoustic_customer=acoustic_whole,
            acoustic_whole=acoustic_whole,
            speaker_overlap_present=False,
            customer_isolated=False,
        )
