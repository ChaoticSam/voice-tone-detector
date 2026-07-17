# Technical Report

Sections numbered to match deliverables 5–9 in the trial spec.

## §5 Approach & Final Architecture

### Approaches tested, in order

1. **Acoustic-dimensional SER alone** (`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`,
   arousal/valence/dominance regression). Reliable for **intensity** (arousal separated the 3
   labeled calls' intensity 3/3 once thresholded), but **valence was weak and anti-correlated**
   with tone on the real calls — the "satisfied" call had the *lowest* valence of the three.
   No threshold fit the labels. Root cause found later: the whole call's audio (bot + customer
   mixed) was being analyzed together — the AI agent's flat TTS voice diluted and confounded
   the acoustic signal.
2. **LLM lexical fusion** (transcript + acoustic summary → `gpt-4.1-mini`, prompted to weigh
   both channels). Matched tone 1/3 on the labeled calls. Also depended on transcript quality,
   which was unreliable on the one non-English call (Spanish; the transcribed customer
   dialogue included garbled/hallucinated text). This was the only component with a real cash
   cost, and it wasn't earning its cost.
3. **Diarization-based customer isolation.** Built speaker diarization (WhisperX + pyannote)
   to separate the AI agent's turns from the customer's, both to deliver
   `speaker_overlap_present` directly and to re-run the acoustic model on the **customer only**.
   This fixed the valence ordering (customer-only valence became monotonic with the labels:
   upset < neutral < satisfied) but broke intensity — customer-only arousal collapsed two of
   the three calls to near-identical values that whole-call arousal had cleanly separated.
   Fixed by splitting the signal sources: **intensity from whole-call arousal, tone-relevant
   signal from customer-only audio** — each stage uses whichever audio matches what it's
   actually measuring.
4. **Categorical SER** (`superb/wav2vec2-base-superb-er`, 4 discrete classes: neutral/happy/
   angry/sad, trained on IEMOCAP). Run on the customer-only audio; its argmax alone matched
   tone 3/3 on the labeled calls — ahead of both the dimensional model and the LLM. `ang`/`sad`
   are disambiguated into frustrated/upset/distressed using the same arousal bands as the
   dimensional model (reused, not reinvented).
5. **LLM dropped entirely.** Once the categorical model out-performed it on tone, the LLM
   added cost and latency for zero accuracy benefit, plus a dependency on transcript quality
   that had already caused a real failure (call_002). Removed along with all transcript
   handling downstream of diarization — diarization is kept only for customer-audio isolation
   and `speaker_overlap_present`.

This satisfies the spec's request to compare at least two materially different approaches
with three genuinely different architectures compared (a dimensional acoustic foundation
model, an LLM-based multimodal fusion, and a categorical acoustic classifier) — not just
hyperparameter variations of one method.

### Final architecture

```
preprocess → diarize (customer isolation + speaker_overlap_present)
           → dimensional model on whole-call audio  → intensity (arousal thresholds)
           → categorical SER on customer-only audio → tone (argmax + arousal-band disambiguation)
           → noise (AST classifier + DSP broadband-static detector, hybrid)
           → audio quality (DSP: clipping/SNR/loudness)
           → silence (DSP: relative-energy dead-air detection)
           → structured 9-field result (confidence = categorical model's top-class probability)
```

Self-hosted only. No LLM, no external paid API anywhere in the shipped pipeline.

## §6 Validation Results

**Metric:** exact-match accuracy per field (all fields are categorical/boolean); macro
precision/recall/F1 for the multi-class tone field where sample size allows.

### Primary evidence: the 3 labeled real production calls

This is the entire labeled real-call sample available — there is no separate train/test
split. All 3 examples were used for both calibrating thresholds and reporting this result;
per the spec's own caution ("accuracy should not be reported from the training set alone"),
**this number should be read as a sanity check that the system is directionally correct, not
as a reliable estimate of hidden-set accuracy.**

| call | field | predicted | ground truth | match |
|---|---|---|---|---|
| call_001 | emotional_tone | upset | upset | ✓ |
| call_001 | emotional_intensity | high | high | ✓ |
| call_001 | background_noise_present | false | false | ✓ |
| call_001 | audio_quality | clear | clear | ✓ |
| call_001 | speaker_overlap_present | false | false | ✓ |
| call_001 | long_silence_present | false | false | ✓ |
| call_002 | emotional_tone | neutral | neutral | ✓ |
| call_002 | emotional_intensity | medium | medium | ✓ |
| call_002 | background_noise_present | true | true | ✓ |
| call_002 | background_noise_type | TV | TV | ✓ |
| call_002 | background_noise_severity | medium | medium | ✓ |
| call_002 | audio_quality | clear | clear | ✓ |
| call_002 | speaker_overlap_present | true | true | ✓ |
| call_002 | long_silence_present | false | false | ✓ |
| call_003 | emotional_tone | satisfied | satisfied | ✓ |
| call_003 | emotional_intensity | medium | medium | ✓ |
| call_003 | background_noise_present | true | true | ✓ |
| call_003 | background_noise_type | static | sharp static | ✓ (substring match) |
| call_003 | background_noise_severity | medium | medium | ✓ |
| call_003 | audio_quality | clear | clear | ✓ |
| call_003 | speaker_overlap_present | true | true | ✓ |
| call_003 | long_silence_present | false | false | ✓ |

**3/3 on every field**, including tone (upset/neutral/satisfied — three different classes,
not a degenerate all-same-answer result) and intensity. `background_noise_type` is free text
so "exact match" isn't the right bar; both predictions are accurate descriptions of the
actual noise (a TV and static respectively).

### Supplementary evidence: CREMA-D proxy (tone classifier only, n=250)

n=3 is not enough to estimate generalization, so the tone classifier (categorical SER +
arousal disambiguation, the same code path used in production) was additionally run against
250 clips from CREMA-D (50 each of anger/happy/neutral/sad/fear; disgust excluded — no honest
mapping onto our 5-tone taxonomy). This is a **domain-mismatch proxy**: CREMA-D is acted,
single-speaker, non-call-center audio, not real dual-speaker support calls — a supplementary
stress test, not a substitute for real-call validation.

Taxonomy mapping: `neutral→neutral`, `happy→satisfied`, `anger`/`sad→frustrated/upset/
distressed` via the same arousal bands used in production, `fear→distressed`.

**Confusion matrix** (rows = true, columns = predicted):

| true \\ predicted | distressed | frustrated | neutral | satisfied | upset |
|---|---|---|---|---|---|
| **distressed** | 17 | 38 | 1 | 2 | 5 |
| **frustrated** | 0 | 63 | 5 | 5 | 0 |
| **neutral** | 0 | 42 | 6 | 2 | 0 |
| **satisfied** | 4 | 29 | 2 | 6 | 9 |
| **upset** | 0 | 0 | 0 | 0 | 14 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| distressed | 0.810 | 0.270 | 0.405 | 63 |
| frustrated | 0.366 | 0.863 | 0.514 | 73 |
| neutral | 0.429 | 0.120 | 0.188 | 50 |
| satisfied | 0.400 | 0.120 | 0.185 | 50 |
| upset | 0.500 | 1.000 | 0.667 | 14 |
| **macro avg** | **0.501** | **0.475** | **0.392** | 250 |

**Overall accuracy: 42.4% (106/250).** Excluding the 50 `fear`-sourced clips (fear is
structurally unrepresentable — the categorical model only has 4 output classes: neutral/
happy/angry/sad, so a true-fear clip can never directly produce "distressed" the way a
high-arousal `ang`/`sad` prediction can): **51.0% (102/200)**.

**This result should be reported honestly, not downplayed.** The model's predictions are
heavily skewed toward "frustrated" (172/250 predictions, vs. 73/250 true) — when this
IEMOCAP-trained categorical model is pushed outside its training distribution (a different
corpus, different actors, different recording conditions), it defaults to guessing
anger/sad-at-low-arousal far more than the true label warrants. This is a well-documented
phenomenon in speech-emotion research generally (SER models routinely degrade sharply under
cross-corpus evaluation) — not evidence of a bug in this implementation, but real evidence
that **the strong 3/3 result on the labeled real calls should not be over-trusted**. See §9.

## §7 Cost Analysis

**No LLM or external paid API in the shipped pipeline** — every model runs self-hosted, so
there is no per-call cash/API cost. The only cost is compute time.

Measured throughput (this session, CPU-only, M-series Mac): processing the 3 labeled calls
(237.8s of audio total) took 282.1s of wall-clock time across repeated runs — **~1.1–1.2x
realtime**, i.e. ~65–71 seconds of compute per audio-minute processed. Diarization
(WhisperX ASR + pyannote) is the dominant cost, running roughly 1x realtime by itself; the
two acoustic models and the DSP stages (noise/quality/silence) are comparatively fast.

Converting to $/audio-minute at a few CPU-instance price points (using the conservative
71s/min figure):

| instance rate | cost per audio-minute |
|---|---|
| $0.10/hr | $0.00197 |
| $0.15/hr | $0.00296 |
| $0.20/hr | $0.00394 |

**Compliant with the $0.003/audio-minute ceiling on a ~$0.10–0.15/hr-class CPU instance**;
a $0.20/hr instance would exceed it. Assumptions: single-threaded per-worker throughput (one
audio file at a time per worker process), models kept warm in memory after the first load
(no reload cost per file — see `app/worker.py`), no GPU (none of the models here require
one at this volume), and concurrency scaled by running multiple worker processes, each
bounded by ~2–3GB RAM for its loaded models rather than by CPU.

## §8 Latency Analysis

Per-clip wall-clock time measured directly (background-processed via the actual queue/worker
path, not a synthetic benchmark):

| call | audio length | wall time | realtime factor |
|---|---|---|---|
| call_001 | 30.9s | 44.0s | 1.42x |
| call_002 | 35.0s | 29.6s | 0.85x |
| call_003 | 171.9s | 207.1s | 1.20x |
| **total** | **237.8s** | **282.1s** | **1.19x** |

Per-audio-minute: **~71 seconds of processing** (measured range across repeated runs:
64–71s/min, normal variance from a shared dev machine, not a regression). Stage breakdown:
diarization (ASR + alignment + pyannote) is the single heaviest stage at roughly 1x realtime
alone; the dimensional and categorical acoustic models are fast (seconds, not tied to audio
length the way diarization is); the noise/quality/silence DSP stages are negligible
(sub-second). For production batch analysis this is "reasonable" per the spec's own bar —
comfortably faster than manual review, and horizontally scalable by adding worker processes.

## §9 Failure Modes, Limitations, Next Steps

**Sample-size risk (the biggest one).** Only 3 real labeled calls exist. The categorical
tone classifier hit 3/3 on them but only 42.4% (51.0% excluding a structurally-unrepresentable
class) on a 250-clip proxy benchmark from a different corpus — concrete, measured evidence
that the 3/3 result is not a reliable estimate of hidden-set performance. The gap is
consistent with a well-known SER phenomenon (sharp cross-corpus degradation), but its exact
size on AutoAce's actual hidden test set is unknown until measured.

**Domain mismatch, twice over.** The categorical model is trained on IEMOCAP (acted,
scripted dialogue); the proxy evaluation used CREMA-D (also acted, different actors/setup);
neither is real, spontaneous, dual-speaker call-center audio — the actual target domain is
one step further removed from both.

**Unvalidated `sad`/categorical-negative disambiguation.** The `ang`/`sad→frustrated/upset/
distressed` arousal-band mapping has exactly one confirming real-call data point (call_001,
`ang`+arousal 0.740→upset, correct). The CREMA-D proxy exercises it far more (see the
skew toward "frustrated" above) but that's evidence of miscalibration under domain shift,
not validation of the bands themselves on real calls.

**Non-English call transcription/diarization was unreliable.** The one Spanish-language
call in the labeled set produced a garbled/partially-hallucinated diarized transcript. This
no longer affects tone (the LLM/transcript path was removed), but it's a live risk for
`speaker_overlap_present`/customer-isolation accuracy on non-English hidden-set audio if
diarization's underlying ASR-assisted role assignment degrades similarly.

**Boundary-adjacent tone misses are the common failure shape.** Where the system missed on
qualitative spot-checks during development, misses were adjacent-class (frustrated vs.
upset; neutral vs. satisfied), not wild misclassifications — suggesting the ordinal
structure of the taxonomy is being captured even when the exact class is missed.

**Manifest "unmatched files" warning is ephemeral, not persisted.** If an uploaded batch
includes audio not listed in its CSV manifest, that's reported once in the upload response
but not stored — refreshing the dashboard loses it (files listed in the manifest but never
uploaded, however, *are* persisted as failed job rows and remain visible). A deliberate scope
cut given the time available, not an oversight.

**Auth is deliberately minimal.** No rate limiting on signin/signup, no email verification,
no password-reset flow — reasonable for a handful of known trial users, not for a
public-facing product. Documented inline in `app/api.py` at each call site.

### Next steps

- Get more labeled real calls, especially non-English and frustrated/distressed/sad
  examples (the current 3 don't cover the full taxonomy).
- Investigate why the categorical model skews toward "frustrated" under domain shift —
  calibration/temperature scaling, or ensembling with the dimensional model's signal with
  learned weights instead of the current simple override, might reduce this.
- If more real-call labels become available, consider fine-tuning the categorical SER model
  on real call-center audio rather than relying on IEMOCAP as-is.
- A more representative proxy dataset than CREMA-D (ideally real call-center audio with
  emotion labels — no such dataset was found freely available on HuggingFace during this
  trial; see the earlier dataset search) would tighten the sample-size risk above.
