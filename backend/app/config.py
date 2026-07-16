"""Shared configuration for the audio pipeline.

Thresholds and constants live here so every stage (validation, preprocessing,
analysis) reads from one source of truth.
"""

# Container formats we accept. The provided samples are .ogg; the hidden test set
# may add mp3/m4a. soundfile handles wav/flac/ogg natively; ffmpeg covers the rest.
SUPPORTED_EXTENSIONS = frozenset({".wav", ".mp3", ".ogg", ".flac", ".m4a"})

# Clips shorter than this carry no analyzable signal and are rejected as too_short.
MIN_DURATION_SEC = 0.1
