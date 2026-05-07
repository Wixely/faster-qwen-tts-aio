from __future__ import annotations

import io
import shutil
import subprocess
from typing import Tuple

import numpy as np
import soundfile as sf

# All formats produced by faster-qwen3-tts are mono int16 PCM at 24 kHz.
TARGET_CHANNELS = 1


CONTENT_TYPES = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "pcm": "audio/L16",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
}


SUPPORTED_FORMATS = tuple(CONTENT_TYPES.keys())


def _ensure_int16_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim == 2:
        # average channels down to mono
        arr = arr.mean(axis=1)
    if arr.dtype == np.int16:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767.0).astype(np.int16)
    return arr.astype(np.int16)


def encode_audio(audio: np.ndarray, sample_rate: int, fmt: str) -> Tuple[bytes, str]:
    """
    Encode the given audio array into the requested format.

    Returns (bytes, content_type).
    """
    fmt = (fmt or "wav").lower()
    if fmt not in CONTENT_TYPES:
        raise ValueError(f"Unsupported format: {fmt!r}. Supported: {sorted(CONTENT_TYPES)}")

    pcm = _ensure_int16_mono(audio)

    if fmt == "pcm":
        return pcm.tobytes(), CONTENT_TYPES[fmt]

    if fmt == "wav":
        buf = io.BytesIO()
        sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), CONTENT_TYPES[fmt]

    if fmt == "flac":
        buf = io.BytesIO()
        sf.write(buf, pcm, sample_rate, format="FLAC", subtype="PCM_16")
        return buf.getvalue(), CONTENT_TYPES[fmt]

    # mp3 / opus / aac via ffmpeg
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"format {fmt!r} requires ffmpeg on PATH; install it or request 'wav', 'flac', or 'pcm'."
        )
    return _ffmpeg_encode(pcm, sample_rate, fmt), CONTENT_TYPES[fmt]


def _ffmpeg_encode(pcm: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    codec_map = {"mp3": ("libmp3lame", "mp3"), "opus": ("libopus", "ogg"), "aac": ("aac", "adts")}
    codec, container = codec_map[fmt]
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-f", "s16le",
        "-ar", str(sample_rate),
        "-ac", str(TARGET_CHANNELS),
        "-i", "pipe:0",
        "-c:a", codec,
        "-f", container,
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=pcm.tobytes(), capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({fmt}): {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def load_reference_audio(path: str) -> Tuple[np.ndarray, int]:
    """Read any soundfile-supported audio file and return (audio, sr)."""
    return sf.read(path, always_2d=False)
