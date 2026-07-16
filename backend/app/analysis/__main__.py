"""CLI: run the DSP analysis services (audio quality + silence) on audio files.

    python -m app.analysis <file> [file ...]
"""

from __future__ import annotations

import sys

from app.analysis.audio_quality import assess_quality
from app.analysis.emotion import detect_emotion
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
        f"{'FILE':<{name_w}}  {'TONE':<11}{'INTENS':<8}{'QUALITY':<17}  {'LONG_SIL':>8}  "
        f"{'NOISE':>6}  {'TYPE':<12}  {'SEV':<7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if isinstance(r, PreprocessError):
            print(f"{r.name:<{name_w}}  ERROR: {r.error}")
            continue
        e = detect_emotion(r)
        q = assess_quality(r)
        s = detect_silence(r.signal_original, r.original_sample_rate)
        n = detect_noise(r)
        print(
            f"{r.name:<{name_w}}  {e.emotional_tone:<11}{e.emotional_intensity:<8}"
            f"{q.audio_quality:<17}  {str(s.long_silence_present):>8}  "
            f"{str(n.background_noise_present):>6}  {n.background_noise_type:<12}  "
            f"{n.background_noise_severity:<7}"
        )
        print(f"{'':<{name_w}}  emotion A/V/D: {e.arousal}/{e.valence}/{e.dominance}  "
              f"(tone is acoustic-only, provisional -> fused with transcript in 5b)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
