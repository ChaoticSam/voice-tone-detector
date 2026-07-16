"""Shared configuration for the audio pipeline.

Thresholds and constants live here so every stage (validation, preprocessing,
analysis) reads from one source of truth.
"""

# Container formats we accept. The provided samples are .ogg; the hidden test set
# may add mp3/m4a. soundfile handles wav/flac/ogg natively; ffmpeg covers the rest.
SUPPORTED_EXTENSIONS = frozenset({".wav", ".mp3", ".ogg", ".flac", ".m4a"})

# Clips shorter than this carry no analyzable signal and are rejected as too_short.
MIN_DURATION_SEC = 0.1


# --- Silence detection (long_silence_present) --------------------------------
# Energy-based dead-air detection. Thresholds are relative to each file's own active
# level, so quiet calls are handled without an absolute-volume assumption.
SILENCE_FRAME_MS = 25          # analysis frame length
# A frame is "silent" when its RMS falls this many dB below the file's active level
# (the high-percentile frame energy). Dead air / digital silence sits far below this.
SILENCE_DROP_DB = 35.0
SILENCE_ACTIVE_PERCENTILE = 95  # percentile of frame RMS taken as the "active" level
# Longest silent run at/above this length flips long_silence_present to true.
#
# Calibrated against docs/labels.csv (all 3 real calls, all long_silence_present=false):
#   call_001 longest run 2.90s, call_002 3.18s (near-start "connecting" gap, not a
#   mid-call problem), call_003 7.33s (a quiet-but-not-silent stretch -- the "sharp
#   static" background noise persists at low level, so a human wouldn't call it dead
#   air even though relative energy drops ~50-70dB below the active level).
# At the old value (3.0s) both call_002 and call_003 were false positives. Any value
# in (7.33, ...] reproduces all 3 labels; 10.0 gives headroom above the boundary
# rather than snapping to it. CAVEAT: all 3 labeled examples are negative for this
# field, so this only bounds the threshold from below -- we have no positive example
# to validate sensitivity, and this should be revisited once positive examples exist.
LONG_SILENCE_SEC = 10.0


# --- Audio quality (clear | slightly_impaired | severely_impaired) -----------
# NOTE: uncalibrated principled defaults — no ground-truth labels available yet.
# Tune against labels.csv once provided. Final quality = worst-case across signals.
# Clipping: fraction of samples at/above full scale. A few samples touching 1.0 (ogg
# encode overshoot) is not audible distortion, so slight starts well above that.
CLIP_SLIGHT_RATIO = 0.001      # 0.1% of samples clipped
CLIP_SEVERE_RATIO = 0.01       # 1% of samples clipped
# SNR (rough percentile estimate): low SNR = static / hiss / noisy line.
SNR_SLIGHT_DB = 18.0
SNR_SEVERE_DB = 10.0
# Overall loudness. Very low level = hard-to-analyze quiet recording.
VOL_SLIGHT_DBFS = -32.0
VOL_SEVERE_DBFS = -45.0
# Percentiles for the rough SNR estimate (active vs noise-floor frame energy).
SNR_NOISE_PERCENTILE = 10
SNR_ACTIVE_PERCENTILE = 90
