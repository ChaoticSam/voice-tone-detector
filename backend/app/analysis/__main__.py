"""CLI: run the full DSP + emotion analysis pipeline on audio files (acoustic only, no LLM).

    python -m app.analysis <file> [file ...]
"""

from __future__ import annotations

import sys

from app.analysis.audio_quality import assess_quality
from app.analysis.emotion_pipeline import analyze_customer_emotion
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
        f"{'NOISE':>6}  {'TYPE':<12}  {'SEV':<7}  {'OVERLAP':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if isinstance(r, PreprocessError):
            print(f"{r.name:<{name_w}}  ERROR: {r.error}")
            continue
        emo = analyze_customer_emotion(r)
        q = assess_quality(r)
        s = detect_silence(r.signal_original, r.original_sample_rate)
        n = detect_noise(r)
        print(
            f"{r.name:<{name_w}}  {emo.emotional_tone:<11}{emo.emotional_intensity:<8}"
            f"{q.audio_quality:<17}  {str(s.long_silence_present):>8}  "
            f"{str(n.background_noise_present):>6}  {n.background_noise_type:<12}  "
            f"{n.background_noise_severity:<7}  {str(emo.speaker_overlap_present):>7}"
        )
        print(
            f"{'':<{name_w}}  emotion[categorical]: {emo.categorical.label} "
            f"(probs={emo.categorical.probs}) customer_arousal={emo.acoustic_customer.arousal} "
            f"whole_arousal={emo.acoustic_whole.arousal} -> tone={emo.emotional_tone} "
            f"intensity={emo.emotional_intensity} isolated={emo.customer_isolated}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
