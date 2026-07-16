"""CLI: run the DSP analysis services (audio quality + silence) on audio files.

    python -m app.analysis <file> [file ...]
"""

from __future__ import annotations

import sys

from app.analysis.audio_quality import assess_quality
from app.analysis.silence import detect_silence
from app.audio.preprocessing import PreprocessError, preprocess_batch


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m app.analysis <file> [file ...]")
        return 2

    results = preprocess_batch(list(argv))
    name_w = max((len(r.name) for r in results), default=4)
    header = (
        f"{'FILE':<{name_w}}  {'QUALITY':<17}  {'SNRdB':>6}  {'CENTROID':>9}  "
        f"{'LONG_SIL':>8}  {'MAXSIL(s)':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if isinstance(r, PreprocessError):
            print(f"{r.name:<{name_w}}  ERROR: {r.error}")
            continue
        q = assess_quality(r)
        s = detect_silence(r.signal_original, r.original_sample_rate)
        print(
            f"{r.name:<{name_w}}  {q.audio_quality:<17}  {q.snr_db:>6.1f}  "
            f"{q.spectral_centroid_hz:>8.0f}Hz  {str(s.long_silence_present):>8}  "
            f"{s.longest_silence_sec:>9.2f}"
        )
        print(f"{'':<{name_w}}  reasons: {'; '.join(q.reasons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
