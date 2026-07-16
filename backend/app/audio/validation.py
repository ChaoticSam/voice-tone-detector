from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Literal, Optional

import soundfile as sf
from pydantic import BaseModel

from app.config import MIN_DURATION_SEC, SUPPORTED_EXTENSIONS

_PROBE_FRAMES = 1024

Status = Literal["valid", "invalid"]
Decoder = Literal["soundfile", "ffmpeg"]
ErrorCode = Literal[
    "empty_file",
    "unsupported_extension",
    "decode_failed",
    "no_audio_stream",
    "too_short",
]


class ValidationResult(BaseModel):
    """Structured outcome of validating a single audio file."""

    name: str
    status: Status
    error: Optional[ErrorCode] = None
    error_detail: Optional[str] = None
    duration_sec: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[str] = None
    decoder: Optional[Decoder] = None


def _invalid(name: str, error: ErrorCode, detail: str) -> ValidationResult:
    return ValidationResult(name=name, status="invalid", error=error, error_detail=detail)


def _probe_soundfile(path: Path) -> ValidationResult:
    """Validate via libsndfile. Raises on any format/decoding problem."""
    info = sf.info(str(path))
    # Force a real read of a small chunk to catch truncated / corrupt streams that
    # still present a valid-looking header.
    with sf.SoundFile(str(path)) as f:
        f.read(frames=_PROBE_FRAMES, dtype="float32")

    duration = info.frames / info.samplerate if info.samplerate else 0.0
    return ValidationResult(
        name=path.name,
        status="valid",
        duration_sec=round(duration, 3),
        sample_rate=info.samplerate,
        channels=info.channels,
        format=info.format,
        decoder="soundfile",
    )


def _probe_ffprobe(path: Path) -> ValidationResult:
    """Fallback validation via ffprobe (metadata only, no full decode).

    Returns an ``invalid`` result (rather than raising) for the cases ffprobe can
    report cleanly; raises only when ffprobe itself is unavailable or errors, so the
    caller can distinguish "decoder can't run" from "file has no audio".
    """
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-show_entries",
            "format=duration,format_name",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    if not streams:
        return _invalid(path.name, "no_audio_stream", "ffprobe found no audio stream")

    stream = streams[0]
    fmt = data.get("format", {})
    duration_raw = fmt.get("duration")
    sample_rate = int(stream["sample_rate"]) if stream.get("sample_rate") else None
    return ValidationResult(
        name=path.name,
        status="valid",
        duration_sec=round(float(duration_raw), 3) if duration_raw else None,
        sample_rate=sample_rate,
        channels=stream.get("channels"),
        format=fmt.get("format_name"),
        decoder="ffmpeg",
    )


def validate_file(path: str | Path) -> ValidationResult:
    """Validate a single audio file. Never raises — failures become ``invalid`` results."""
    path = Path(path)
    name = path.name

    # 1. Existence / emptiness.
    if not path.is_file():
        return _invalid(name, "empty_file", "file does not exist")
    if path.stat().st_size == 0:
        return _invalid(name, "empty_file", "file is empty (0 bytes)")

    # 2. Extension allowlist.
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return _invalid(
            name,
            "unsupported_extension",
            f"extension {path.suffix!r} not in {sorted(SUPPORTED_EXTENSIONS)}",
        )

    # 3. Decode: soundfile first, ffprobe fallback.
    result: ValidationResult
    try:
        result = _probe_soundfile(path)
    except Exception as sf_err:  # noqa: BLE001 - soundfile raises varied error types
        try:
            result = _probe_ffprobe(path)
        except FileNotFoundError:
            # ffprobe binary not installed — report the underlying decode failure.
            return _invalid(
                name,
                "decode_failed",
                f"soundfile could not decode file and ffprobe is unavailable: {sf_err}",
            )
        except Exception as ff_err:  # noqa: BLE001
            return _invalid(
                name,
                "decode_failed",
                f"soundfile and ffprobe both failed: {sf_err} | {ff_err}",
            )

    if result.status == "invalid":
        return result

    # 4. Duration sanity.
    if result.duration_sec is not None and result.duration_sec < MIN_DURATION_SEC:
        return _invalid(
            name,
            "too_short",
            f"duration {result.duration_sec}s below minimum {MIN_DURATION_SEC}s",
        )

    return result


def validate_batch(paths: list[str | Path]) -> list[ValidationResult]:
    """Validate many files with per-file isolation.

    A failure on one file can never abort the others — mirrors the queue's per-file
    failure handling required by the batch dashboard.
    """
    results: list[ValidationResult] = []
    for p in paths:
        try:
            results.append(validate_file(p))
        except Exception as err:  # noqa: BLE001 - last-resort safety net
            results.append(_invalid(Path(p).name, "decode_failed", f"unexpected error: {err}"))
    return results


def _main(argv: list[str]) -> int:
    if not argv:
        print("usage: python -m app.audio.validation <file> [file ...]")
        return 2

    results = validate_batch(list(argv))
    name_w = max((len(r.name) for r in results), default=4)
    header = f"{'FILE':<{name_w}}  {'STATUS':<7}  {'DUR(s)':>8}  {'SR':>6}  {'CH':>2}  DETAIL"
    print(header)
    print("-" * len(header))
    for r in results:
        dur = f"{r.duration_sec:.2f}" if r.duration_sec is not None else "-"
        sr = str(r.sample_rate) if r.sample_rate is not None else "-"
        ch = str(r.channels) if r.channels is not None else "-"
        detail = r.error or (r.decoder or "")
        print(f"{r.name:<{name_w}}  {r.status:<7}  {dur:>8}  {sr:>6}  {ch:>2}  {detail}")

    return 0 if all(r.status == "valid" for r in results) else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
