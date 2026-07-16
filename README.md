# Voice Tone & Background Noise Detector

A modular, cost-efficient pipeline for classifying **emotional tone** and detecting
**background noise / audio quality** in production call audio.

Built stepwise as independent, replaceable stages:

```
Upload → Validation → Preprocessing → Parallel Analysis → Aggregation → Decision Engine → JSON
                                       (emotion · noise · transcription · quality · overlap · silence)
```

Design goals: self-hosted open models (no per-minute API billing, audio never leaves our
infrastructure), confidence-aware ensembling, and a deterministic-first approach that keeps
inference under a tight cost ceiling.

## Status

| Stage | Status |
|-------|--------|
| 1. Validation | ✅ done |
| 2. Preprocessing | ✅ done |
| 3+ Analysis services, decision engine, dashboard | in progress |

Each new stage is developed on its own branch.

## Getting started

See [`backend/README.md`](backend/README.md) for setup, usage, and tests.

## Note on data

Confidential trial audio and documents are intentionally excluded from version control
(see [`.gitignore`](.gitignore)); this repository contains code only.
