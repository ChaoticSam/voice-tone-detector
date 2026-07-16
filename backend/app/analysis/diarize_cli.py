"""CLI: run speaker diarization + overlap on audio files.

    python -m app.analysis.diarize_cli <file> [file ...]
"""

from __future__ import annotations

import sys

from app.analysis.diarization import DiarizationUnavailable, diarize
from app.audio.preprocessing import PreprocessError, preprocess_batch


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m app.analysis.diarize_cli <file> [file ...]")
        return 2

    for r in preprocess_batch(list(argv)):
        if isinstance(r, PreprocessError):
            print(f"{r.name}: PREPROCESS ERROR: {r.error}")
            continue
        print(f"\n=== {r.name} ===")
        try:
            d = diarize(r)
        except DiarizationUnavailable as err:
            print(f"  diarization unavailable: {err}")
            continue
        print(f"  speakers={d.speakers}  agent={d.agent_speaker}  customer={d.customer_speaker}")
        print(f"  speaker_overlap_present={d.speaker_overlap_present}  ({d.overlap_seconds}s)")
        print(f"  customer transcript: \"{d.customer_transcript[:160]}\"")
        for t in d.turns[:8]:
            role = "AGENT" if t.speaker == d.agent_speaker else (
                "CUST" if t.speaker == d.customer_speaker else t.speaker)
            print(f"    [{t.start:6.1f}-{t.end:6.1f}] {role:<5} {t.text[:64]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
