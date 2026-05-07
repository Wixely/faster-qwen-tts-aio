from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import List

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import (
    Attribution,
    Describe,
    Info,
    TtsProgram,
    TtsVoice,
    TtsVoiceSpeaker,
)
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

from . import __version__
from .config import Settings
from .tts_engine import EngineError, TTSEngine

log = logging.getLogger(__name__)

# Wyoming consumes raw 16-bit PCM. faster-qwen3-tts emits 24 kHz mono int16.
SAMPLE_WIDTH = 2
CHANNELS = 1


def _build_info(engine: TTSEngine, settings: Settings) -> Info:
    voice_names: List[str] = []
    voice_names.extend(v.name for v in engine.catalog.list_custom())
    if engine.custom_loaded:
        voice_names.extend(settings.builtin_speakers)
    # de-duplicate while preserving order
    seen = set()
    voices: List[TtsVoice] = []
    for name in voice_names:
        if name in seen:
            continue
        seen.add(name)
        voices.append(
            TtsVoice(
                name=name,
                description=f"Qwen3 TTS voice: {name}",
                attribution=Attribution(name="Qwen", url="https://github.com/QwenLM/Qwen3-TTS"),
                installed=True,
                version=None,
                languages=["en"],
                speakers=[TtsVoiceSpeaker(name=name)],
            )
        )

    if not voices:
        # Even if no voices are configured, advertise something so HA can connect.
        voices.append(
            TtsVoice(
                name="default",
                description="Qwen3 TTS default",
                attribution=Attribution(name="Qwen", url="https://github.com/QwenLM/Qwen3-TTS"),
                installed=True,
                version=None,
                languages=["en"],
            )
        )

    return Info(
        tts=[
            TtsProgram(
                name="faster-qwen-tts-aio",
                description="Qwen3-TTS via faster-qwen3-tts",
                attribution=Attribution(
                    name="faster-qwen-tts-aio",
                    url="https://github.com/",
                ),
                installed=True,
                version=__version__,
                voices=voices,
                supports_synthesize_streaming=False,
            )
        ]
    )


class QwenTtsHandler(AsyncEventHandler):
    def __init__(self, engine: TTSEngine, settings: Settings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.settings = settings

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            info = _build_info(self.engine, self.settings)
            await self.write_event(info.event())
            return True

        if Synthesize.is_type(event.type):
            syn = Synthesize.from_event(event)
            voice_name = None
            language = None
            if syn.voice is not None:
                # Prefer explicit speaker, else voice name.
                voice_name = syn.voice.speaker or syn.voice.name
                language = syn.voice.language

            try:
                result = await self.engine.synthesize_async(
                    text=syn.text,
                    voice=voice_name,
                    language=language or self.settings.default_language,
                )
            except EngineError as e:
                log.warning("Wyoming synth rejected: %s", e)
                # Send empty audio frame to terminate cleanly.
                await self.write_event(
                    AudioStart(rate=self.engine.sample_rate, width=SAMPLE_WIDTH, channels=CHANNELS).event()
                )
                await self.write_event(AudioStop().event())
                return True

            audio_bytes = result.audio.tobytes()
            rate = int(result.sample_rate)
            await self.write_event(
                AudioStart(rate=rate, width=SAMPLE_WIDTH, channels=CHANNELS).event()
            )
            chunk_bytes = 1024 * SAMPLE_WIDTH * CHANNELS
            for i in range(0, len(audio_bytes), chunk_bytes):
                await self.write_event(
                    AudioChunk(
                        audio=audio_bytes[i : i + chunk_bytes],
                        rate=rate,
                        width=SAMPLE_WIDTH,
                        channels=CHANNELS,
                    ).event()
                )
            await self.write_event(AudioStop().event())
            return True

        return True


async def run_wyoming_server(engine: TTSEngine, settings: Settings) -> None:
    log.info("Starting Wyoming server on %s", settings.wyoming_uri)
    try:
        server = AsyncServer.from_uri(settings.wyoming_uri)
        handler_factory = partial(QwenTtsHandler, engine, settings)
        await server.run(handler_factory)
    except asyncio.CancelledError:
        log.info("Wyoming server stopped")
        raise
    except Exception:
        log.exception("Wyoming server failed; HTTP API will continue running")
