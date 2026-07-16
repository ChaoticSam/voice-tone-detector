"""Regression test against the real labeled calls (docs/labels.csv).

Only checks the fields this stage produces (audio_quality, long_silence_present). Other
required-output fields (emotion, noise, overlap) aren't implemented yet and are skipped
here -- extend the assertions as each later stage lands.

CAVEAT: all 3 labeled calls are negative for long_silence_present, so this test can only
catch false positives / threshold regressions, not validate sensitivity to a true long
silence. Revisit once positive examples are available.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.analysis.audio_quality import assess_quality
from app.analysis.silence import detect_silence
from app.audio.preprocessing import preprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_PATH = REPO_ROOT / "docs" / "labels.csv"
AUDIO_DIR = REPO_ROOT / "audio_files"


def _load_labels() -> dict[str, dict]:
    with open(LABELS_PATH) as f:
        return {row["name"]: json.loads(row["result_json"]) for row in csv.DictReader(f)}


@pytest.mark.skipif(not LABELS_PATH.exists(), reason="docs/labels.csv not present")
@pytest.mark.parametrize("name", ["call_001.ogg", "call_002.ogg", "call_003.ogg"])
def test_matches_ground_truth(name: str):
    labels = _load_labels()
    if name not in labels or not (AUDIO_DIR / name).exists():
        pytest.skip(f"{name} audio or label not present")

    gt = labels[name]
    pre = preprocess(AUDIO_DIR / name)
    q = assess_quality(pre)
    s = detect_silence(pre.signal_original, pre.original_sample_rate)

    assert q.audio_quality == gt["audio_quality"]
    assert s.long_silence_present == gt["long_silence_present"]
