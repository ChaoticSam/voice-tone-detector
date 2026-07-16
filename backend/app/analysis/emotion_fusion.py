"""Emotion fusion (stage 5b): combine the acoustic and lexical channels.

Emotion has two channels — prosodic ("how it was said", stage 5a) and lexical ("what was
said", the transcript). 5a showed the acoustic channel is reliable for intensity but not
for tone (valence was flat/anti-correlated on the real calls). This module runs LLM-based
fusion: the model sees BOTH the transcript and the acoustic summary and returns the final
tone/intensity/confidence — naturally handling "I've called five times," said flatly =
frustrated (calm voice, frustrated words).

If the LLM is unavailable (no key, network, error), we fall back to 5a's acoustic verdict
with reduced confidence rather than crash the batch.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.analysis.emotion import AcousticEmotionResult, Intensity, Tone
from app.analysis.llm_client import LLMUnavailable, complete_json
from app.analysis.transcription import Transcript
from app.config import ACOUSTIC_FALLBACK_CONFIDENCE

_TONES = ("neutral", "satisfied", "frustrated", "upset", "distressed")
_INTENSITIES = ("low", "medium", "high")

# Tone/intensity definitions, verbatim intent from the trial spec, kept short + stable
# (this is the cacheable instruction prefix).
_SYSTEM = """You classify a customer's emotion in a support call by fusing two channels:
- HOW it was said (acoustic: arousal/valence, 0..1) — reliable for intensity, weak for tone.
- WHAT was said (the transcript) — reliable for tone/meaning.

Weigh both. A calm voice can carry frustrated words ("I've called five times") — trust the
words for tone. Loudness alone is NOT frustration or distress.

emotional_tone (pick one):
- neutral: no clear positive or negative emotion.
- satisfied: pleased, relieved, appreciative, clearly positive.
- frustrated: annoyed, impatient, dissatisfied, without strong anger or distress.
- upset: clearly angry, agitated, or strongly dissatisfied.
- distressed: highly emotional, overwhelmed, panicked, crying, escalated.

Judge only the TONE (intensity is decided separately from the voice's arousal).

Reply as JSON only: {"emotional_tone": <tone>, "confidence": <0..1>,
"rationale": "<one short sentence>"}."""


class FusedEmotionResult(BaseModel):
    emotional_tone: Tone
    emotional_intensity: Intensity
    confidence: float
    source: Literal["fused", "acoustic_fallback"]
    rationale: str
    transcript: str
    acoustic: AcousticEmotionResult


def _build_user_prompt(acoustic: AcousticEmotionResult, transcript: Transcript) -> str:
    return (
        "ACOUSTIC SUMMARY (how it was said):\n"
        f"  arousal={acoustic.arousal:.2f} (0=calm, 1=activated)\n"
        f"  valence={acoustic.valence:.2f} (0=negative, 1=positive)\n"
        f"  provisional acoustic tone/intensity: {acoustic.emotional_tone} / "
        f"{acoustic.emotional_intensity}\n"
        f"  speaking rate: {transcript.speaking_rate_wpm:.0f} wpm\n\n"
        "TRANSCRIPT (what was said):\n"
        f'  "{transcript.text or "(no speech detected)"}"\n'
    )


def _acoustic_fallback(
    acoustic: AcousticEmotionResult, transcript: Transcript, reason: str
) -> FusedEmotionResult:
    return FusedEmotionResult(
        emotional_tone=acoustic.emotional_tone,
        emotional_intensity=acoustic.emotional_intensity,
        confidence=ACOUSTIC_FALLBACK_CONFIDENCE,
        source="acoustic_fallback",
        rationale=f"lexical channel unavailable ({reason}); acoustic-only verdict",
        transcript=transcript.text,
        acoustic=acoustic,
    )


def fuse_emotion(
    acoustic: AcousticEmotionResult, transcript: Transcript
) -> FusedEmotionResult:
    """Fuse acoustic + lexical channels into the final emotion verdict."""
    try:
        out = complete_json(_SYSTEM, _build_user_prompt(acoustic, transcript))
    except LLMUnavailable as err:
        return _acoustic_fallback(acoustic, transcript, str(err))

    tone = out.get("emotional_tone")
    if tone not in _TONES:
        # Malformed reply — don't trust a partial verdict; fall back.
        return _acoustic_fallback(acoustic, transcript, "malformed LLM output")

    try:
        confidence = max(0.0, min(1.0, float(out.get("confidence", 0.7))))
    except (TypeError, ValueError):
        confidence = 0.7

    # Intensity comes from the ACOUSTIC channel (arousal), which matched labels 3/3 in 5a.
    # The lexical channel owns TONE; letting the LLM also set intensity regressed it (it
    # overrode a correct acoustic 'high' with 'medium'). Prosody is the right signal for
    # "how strong", so acoustic intensity is authoritative here.
    return FusedEmotionResult(
        emotional_tone=tone,
        emotional_intensity=acoustic.emotional_intensity,
        confidence=round(confidence, 3),
        source="fused",
        rationale=str(out.get("rationale", ""))[:300],
        transcript=transcript.text,
        acoustic=acoustic,
    )
