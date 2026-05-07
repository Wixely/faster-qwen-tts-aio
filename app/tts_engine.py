from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import Settings
from .voices import VoiceCatalog

log = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    audio: np.ndarray  # mono int16 PCM
    sample_rate: int


class EngineError(RuntimeError):
    """Raised for engine-level failures (model not loaded, voice unknown, etc.)."""


class TTSEngine:
    """
    Thin wrapper around faster-qwen3-tts.

    - The library is CUDA-only and synchronous; we serialise inference behind a
      lock and run blocking calls through asyncio.to_thread when called from the
      async event loop.
    - Two model variants can be loaded:
        * BASE (zero-shot voice cloning, used for custom uploaded voices)
        * CUSTOM_VOICE (built-in named speakers)
    """

    def __init__(self, settings: Settings, catalog: VoiceCatalog):
        self.settings = settings
        self.catalog = catalog
        self._base_model = None
        self._custom_model = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

    # ---- lifecycle ------------------------------------------------------------------

    def ensure_loaded(self) -> None:
        """Load any configured models. Safe to call repeatedly."""
        # Import lazily so that the package can be imported on machines without
        # CUDA / torch (useful for static checks and CI of unrelated bits).
        from faster_qwen3_tts import FasterQwen3TTS  # type: ignore

        with self._load_lock:
            if self.settings.base_model_id and self._base_model is None:
                log.info("Loading BASE model: %s", self.settings.base_model_id)
                self._base_model = FasterQwen3TTS.from_pretrained(self.settings.base_model_id)
                log.info("BASE model loaded.")

            if self.settings.custom_voice_model_id and self._custom_model is None:
                log.info("Loading CustomVoice model: %s", self.settings.custom_voice_model_id)
                self._custom_model = FasterQwen3TTS.from_pretrained(
                    self.settings.custom_voice_model_id
                )
                log.info("CustomVoice model loaded.")

            if self._base_model is None and self._custom_model is None:
                raise EngineError(
                    "No TTS model is configured. Set BASE_MODEL_ID and/or CUSTOM_VOICE_MODEL_ID."
                )

    def close(self) -> None:
        # Drop references so torch can free GPU memory at process exit.
        self._base_model = None
        self._custom_model = None

    # ---- info -----------------------------------------------------------------------

    @property
    def base_loaded(self) -> bool:
        return self._base_model is not None

    @property
    def custom_loaded(self) -> bool:
        return self._custom_model is not None

    @property
    def sample_rate(self) -> int:
        # Qwen3-TTS-12Hz models all output 24 kHz.
        return 24000

    def status(self) -> dict:
        return {
            "base_model_id": self.settings.base_model_id or None,
            "custom_voice_model_id": self.settings.custom_voice_model_id or None,
            "base_loaded": self.base_loaded,
            "custom_loaded": self.custom_loaded,
            "sample_rate": self.sample_rate,
        }

    # ---- voice resolution -----------------------------------------------------------

    def resolve_voice(self, name: Optional[str]) -> Tuple[str, str]:
        """
        Resolve a requested voice name to a backend.

        Returns (mode, voice_name) where mode is one of:
          - "custom":   custom uploaded voice; use BASE model voice cloning
          - "builtin":  one of settings.builtin_speakers; use CustomVoice model
        """
        wanted = (name or self.settings.default_voice or "").strip()
        if not wanted:
            raise EngineError("No voice specified and no default voice configured.")

        custom = self.catalog.get_custom(wanted)
        if custom is not None:
            if not self.base_loaded:
                raise EngineError(
                    f"Voice {wanted!r} is a custom (cloned) voice but BASE_MODEL_ID is not loaded."
                )
            return "custom", wanted

        builtin = self.catalog.get_builtin(wanted)
        if builtin is not None:
            if not self.custom_loaded:
                raise EngineError(
                    f"Voice {wanted!r} is a built-in speaker but CUSTOM_VOICE_MODEL_ID is not loaded."
                )
            return "builtin", wanted

        raise EngineError(f"Unknown voice: {wanted!r}")

    # ---- synthesis (sync; call via to_thread from async code) ----------------------

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> SynthesisResult:
        if not text or not text.strip():
            raise EngineError("text is empty")

        self.ensure_loaded()
        mode, voice_name = self.resolve_voice(voice)
        lang = language or self.settings.default_language

        with self._lock:
            if mode == "custom":
                voice_obj = self.catalog.get_custom(voice_name)
                assert voice_obj is not None  # resolved above
                audio, sr = self._base_model.generate_voice_clone(  # type: ignore[union-attr]
                    text=text,
                    language=lang,
                    ref_audio=str(voice_obj.audio_path),
                    ref_text=voice_obj.transcript,
                    append_silence=True,
                )
            elif mode == "builtin":
                audio, sr = self._custom_model.generate_custom_voice(  # type: ignore[union-attr]
                    speaker=voice_name,
                    text=text,
                    language=lang,
                )
            else:  # pragma: no cover - resolve_voice only returns the two modes above
                raise EngineError(f"unhandled mode: {mode}")

        audio_arr = _to_int16_mono(audio)
        return SynthesisResult(audio=audio_arr, sample_rate=int(sr))

    async def synthesize_async(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
    ) -> SynthesisResult:
        return await asyncio.to_thread(self.synthesize, text, voice, language)


def _to_int16_mono(audio) -> np.ndarray:
    """Coerce engine output (list of chunks, ndarray, possibly float) to mono int16."""
    # The README labels the non-streaming return as `audio_list` — handle both a
    # single ndarray and a list of ndarray chunks defensively.
    if isinstance(audio, (list, tuple)) and audio and hasattr(audio[0], "shape"):
        audio = np.concatenate([np.asarray(a).reshape(-1) for a in audio], axis=0)
    arr = np.asarray(audio)
    if arr.ndim == 2:
        # interleaved channels -> mono
        arr = arr.mean(axis=1)
    if arr.dtype == np.int16:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767.0).astype(np.int16)
    return arr.astype(np.int16)
