"""Speech-to-text transcription via faster-whisper (self-hosted).

Produces the transcript that feeds the lexical emotion channel (stage 5b) and the
speaking-rate feature. Runs entirely on local infrastructure — customer audio never leaves
our environment (only the derived transcript is later sent to the fusion LLM, which is
disclosed per the trial's data-handling rules).

`small` int8 on CPU is the cost/quality sweet spot: the transcript is a *feature for
emotion fusion*, not a scored output, so we need "good enough to read intent," not perfect
ASR.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from app.audio.preprocessing import Preprocessed
from app.config import WHISPER_COMPUTE_TYPE, WHISPER_MODEL_SIZE


class Transcript(BaseModel):
    text: str
    language: str
    language_probability: float
    duration_sec: float           # speech duration covered by segments
    speaking_rate_wpm: float      # words per minute over the spoken portion
    num_words: int


@lru_cache(maxsize=2)
def _load_whisper(model_size: str, compute_type: str) -> Any:
    """Load and cache a faster-whisper model (int8 CPU)."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type=compute_type)


def transcribe(pre: Preprocessed) -> Transcript:
    """Transcribe a preprocessed clip's 16 kHz waveform."""
    model = _load_whisper(WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE)

    # faster-whisper accepts a float32 numpy array at 16 kHz directly.
    segments, info = model.transcribe(pre.waveform_16k, beam_size=5)
    segments = list(segments)  # generator -> materialize

    text = " ".join(s.text.strip() for s in segments).strip()
    spoken_sec = sum(max(0.0, s.end - s.start) for s in segments)
    num_words = len(text.split())
    wpm = (num_words / spoken_sec * 60.0) if spoken_sec > 0 else 0.0

    return Transcript(
        text=text,
        language=info.language,
        language_probability=round(float(info.language_probability), 3),
        duration_sec=round(spoken_sec, 2),
        speaking_rate_wpm=round(wpm, 1),
        num_words=num_words,
    )
