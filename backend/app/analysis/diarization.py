"""Speaker diarization + overlap (stage 6) via WhisperX (self-hosted).

These are AI-receptionist calls: a TTS agent ("Erica") + the human customer. This stage
separates the two and produces:
  * speaker_overlap_present  -- a required output field
  * customer-only transcript + time-ranges -- consumed later by the emotion stage so it
    analyzes the CUSTOMER only (the bot's flat voice was diluting emotion when mixed in)

WhisperX = faster-whisper transcription + word-level alignment + pyannote diarization, all
self-hosted; audio never leaves local infra. pyannote is gated -> needs HF_TOKEN (.env) and
one-time acceptance of the pyannote terms on Hugging Face.

Role heuristic: the agent answers first, so the earliest-starting turn's speaker is the
agent; the customer is the other speaker (or, if >2, the non-agent with the most speech).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseModel

from app.audio.preprocessing import Preprocessed
from app.config import (
    DIARIZATION_MAX_SPEAKERS,
    DIARIZATION_MIN_OVERLAP_SEC,
    DIARIZATION_MIN_TURN_SEC,
    DIARIZATION_OVERLAP_TOLERANCE_SEC,
    WHISPER_COMPUTE_TYPE,
    WHISPER_MODEL_SIZE,
)

_DEVICE = "cpu"

# Scripted lines the AI agent says — a reliable "this speaker is the agent" signal,
# far more robust than "who spoke first" (a stray caller utterance can precede the greeting).
_AGENT_MARKERS = (
    "how can i help", "how can i assist", "how may i help",
    "i'm eri", "i am eri", "this is eri",   # Erica / Erika
    "soy eri", "cómo puedo ayudar", "como puedo ayudar",
)


class Turn(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class DiarizationResult(BaseModel):
    turns: list[Turn]
    speakers: list[str]
    agent_speaker: Optional[str] = None
    customer_speaker: Optional[str] = None
    speaker_overlap_present: bool = False
    overlap_seconds: float = 0.0
    customer_transcript: str = ""
    customer_ranges: list[tuple[float, float]] = []
    full_transcript: str = ""


class DiarizationUnavailable(Exception):
    """Raised when WhisperX/pyannote can't run (missing token, un-accepted terms, error)."""


@lru_cache(maxsize=1)
def _asr_model() -> Any:
    import whisperx

    return whisperx.load_model(WHISPER_MODEL_SIZE, _DEVICE, compute_type=WHISPER_COMPUTE_TYPE)


@lru_cache(maxsize=4)
def _align_model(language: str) -> tuple[Any, Any]:
    import whisperx

    return whisperx.load_align_model(language_code=language, device=_DEVICE)


@lru_cache(maxsize=1)
def _diarize_pipeline() -> Any:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise DiarizationUnavailable("HF_TOKEN not set (pyannote is gated)")
    import whisperx

    # DiarizationPipeline location + auth kwarg vary across WhisperX versions.
    Pipe = getattr(whisperx, "DiarizationPipeline", None)
    if Pipe is None:
        from whisperx.diarize import DiarizationPipeline as Pipe  # type: ignore
    try:
        return Pipe(token=token, device=_DEVICE)            # newer whisperx
    except TypeError:
        return Pipe(use_auth_token=token, device=_DEVICE)   # older whisperx


def _compute_overlap(rows: list[tuple[float, float, str]]) -> tuple[bool, float]:
    """Return (meaningful_overlap_present, total_overlap_seconds).

    Sums cross-speaker overlaps that individually exceed the per-pair tolerance, then flags
    presence only when the total reaches DIARIZATION_MIN_OVERLAP_SEC ("enough to affect
    understanding" per the spec) — brief boundary blips don't count.
    """
    tol = DIARIZATION_OVERLAP_TOLERANCE_SEC
    total = 0.0
    for i in range(len(rows)):
        s1, e1, sp1 = rows[i]
        for j in range(i + 1, len(rows)):
            s2, e2, sp2 = rows[j]
            if sp1 == sp2:
                continue
            ov = min(e1, e2) - max(s1, s2)
            if ov > tol:
                total += ov
    return total >= DIARIZATION_MIN_OVERLAP_SEC, round(total, 3)


def _pick_agent(turns: list[Turn]) -> Optional[str]:
    """Agent = speaker of the scripted greeting; fallback to earliest-starting speaker."""
    if not turns:
        return None
    hits: dict[str, int] = {}
    for t in turns:
        tl = t.text.lower()
        if any(mk in tl for mk in _AGENT_MARKERS):
            hits[t.speaker] = hits.get(t.speaker, 0) + 1
    if hits:
        best = max(hits.values())
        cands = [sp for sp, v in hits.items() if v == best]
        return min(cands, key=lambda sp: min(t.start for t in turns if t.speaker == sp))
    return min(turns, key=lambda t: t.start).speaker


def diarize(pre: Preprocessed) -> DiarizationResult:
    """Diarize a preprocessed clip -> turns, roles, overlap, customer-only transcript/ranges."""
    import whisperx

    audio = pre.waveform_16k  # float32 @ 16 kHz — WhisperX's expected input format
    try:
        asr = _asr_model()
        result = asr.transcribe(audio, batch_size=16)
        language = result.get("language", "en")
        try:
            amodel, meta = _align_model(language)
            result = whisperx.align(result["segments"], amodel, meta, audio, _DEVICE,
                                    return_char_alignments=False)
        except Exception:  # noqa: BLE001 - alignment is best-effort; proceed without it
            pass
        if DIARIZATION_MAX_SPEAKERS:
            diarize_segments = _diarize_pipeline()(audio, max_speakers=DIARIZATION_MAX_SPEAKERS)
        else:
            diarize_segments = _diarize_pipeline()(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
    except DiarizationUnavailable:
        raise
    except Exception as err:  # noqa: BLE001
        msg = str(err).lower()
        if any(k in msg for k in ("401", "403", "gated", "authoriz", "terms", "user conditions")):
            raise DiarizationUnavailable(
                "pyannote access denied — with the HF_TOKEN account, accept the model terms at "
                "https://hf.co/pyannote/speaker-diarization-community-1 (and any segmentation "
                f"model it depends on), then retry: {err}"
            ) from err
        raise DiarizationUnavailable(str(err)) from err

    # Build turns from the speaker-assigned segments.
    turns: list[Turn] = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        spk = seg.get("speaker") or "UNKNOWN"
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end - start < DIARIZATION_MIN_TURN_SEC and not text:
            continue
        turns.append(Turn(start=start, end=end, speaker=spk, text=text))

    speakers = sorted({t.speaker for t in turns if t.speaker != "UNKNOWN"})

    # Roles: agent = greeting speaker; customer = most-talking other speaker.
    agent = _pick_agent(turns)
    customer = None
    if agent is not None:
        talk: dict[str, float] = {}
        for t in turns:
            if t.speaker != agent:
                talk[t.speaker] = talk.get(t.speaker, 0.0) + (t.end - t.start)
        customer = max(talk, key=talk.get) if talk else None

    # Overlap from raw diarization rows (has overlapping regions pyannote detected).
    rows = _diarize_rows(diarize_segments)
    overlap_present, overlap_sec = _compute_overlap(rows)

    cust_turns = [t for t in turns if t.speaker == customer] if customer else []
    customer_transcript = " ".join(t.text for t in cust_turns if t.text).strip()
    customer_ranges = [(t.start, t.end) for t in cust_turns]
    full_transcript = " ".join(t.text for t in turns if t.text).strip()

    return DiarizationResult(
        turns=turns,
        speakers=speakers,
        agent_speaker=agent,
        customer_speaker=customer,
        speaker_overlap_present=overlap_present,
        overlap_seconds=overlap_sec,
        customer_transcript=customer_transcript,
        customer_ranges=customer_ranges,
        full_transcript=full_transcript,
    )


def _diarize_rows(diarize_segments: Any) -> list[tuple[float, float, str]]:
    """Extract (start, end, speaker) rows from a pyannote/WhisperX diarization result."""
    rows: list[tuple[float, float, str]] = []
    # WhisperX returns a pandas DataFrame with columns start/end/speaker.
    try:
        for _, r in diarize_segments.iterrows():
            rows.append((float(r["start"]), float(r["end"]), str(r["speaker"])))
        return rows
    except AttributeError:
        pass
    # Fallback: a pyannote Annotation.
    try:
        for seg, _, label in diarize_segments.itertracks(yield_label=True):
            rows.append((float(seg.start), float(seg.end), str(label)))
    except Exception:  # noqa: BLE001
        pass
    return rows
