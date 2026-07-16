"""CLI: run the DSP analysis services (audio quality + silence) on audio files.

    python -m app.analysis <file> [file ...]
"""

from __future__ import annotations

import sys

from app.analysis.audio_quality import assess_quality
from app.analysis.noise import detect_noise
from app.analysis.silence import detect_silence
from app.audio.preprocessing import PreprocessError, preprocess_batch


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m app.analysis <file> [file ...]")
        return 2

    results = preprocess_batch(list(argv))
    name_w = max((len(r.name) for r in results), default=4)
    header = (
        f"{'FILE':<{name_w}}  {'QUALITY':<17}  {'LONG_SIL':>8}  "
        f"{'NOISE':>6}  {'TYPE':<16}  {'SEVERITY':<8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if isinstance(r, PreprocessError):
            print(f"{r.name:<{name_w}}  ERROR: {r.error}")
            continue
        q = assess_quality(r)
        s = detect_silence(r.signal_original, r.original_sample_rate)
        n = detect_noise(r)
        print(
            f"{r.name:<{name_w}}  {q.audio_quality:<17}  {str(s.long_silence_present):>8}  "
            f"{str(n.background_noise_present):>6}  {n.background_noise_type:<16}  "
            f"{n.background_noise_severity:<8}"
        )
        static_note = f" [static sfm={n.static_sfm}]" if n.static_detected else ""
        print(f"{'':<{name_w}}  noise top: {[(l, round(p, 3)) for l, p in n.top_classes[:4]]}{static_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
