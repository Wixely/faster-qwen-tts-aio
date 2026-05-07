from __future__ import annotations

import logging
import time
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .audio_utils import SUPPORTED_FORMATS, encode_audio
from .config import Settings
from .tts_engine import EngineError, TTSEngine

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])


# ---- auth -------------------------------------------------------------------

def _check_auth(request: Request, settings: Settings) -> None:
    """Validate Bearer token if OPENAI_API_KEY is configured. Otherwise allow."""
    expected = settings.openai_api_key
    if not expected:
        return
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth[len("bearer "):].strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---- /v1/models -------------------------------------------------------------

@router.get("/models")
async def list_models(request: Request) -> JSONResponse:
    settings: Settings = request.app.state.settings
    _check_auth(request, settings)
    created = int(time.time())
    data = [
        {"id": "qwen-tts", "object": "model", "created": created, "owned_by": "local"},
    ]
    return JSONResponse({"object": "list", "data": data})


# ---- /v1/audio/speech -------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str = "qwen-tts"
    input: str = Field(..., description="Text to synthesise")
    voice: Optional[str] = None
    response_format: Optional[
        Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
    ] = None
    speed: Optional[float] = Field(default=1.0, ge=0.25, le=4.0)
    instructions: Optional[str] = None
    # OpenAI also accepts a `stream` flag we silently ignore.


@router.post("/audio/speech")
async def create_speech(request: Request, body: SpeechRequest) -> Response:
    settings: Settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine
    _check_auth(request, settings)

    if not body.input or not body.input.strip():
        raise HTTPException(status_code=400, detail="`input` is required")
    if len(body.input) > settings.max_input_chars:
        raise HTTPException(
            status_code=400,
            detail=f"`input` exceeds max length of {settings.max_input_chars} chars",
        )

    fmt = (body.response_format or settings.default_response_format).lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=415, detail=f"Unsupported response_format: {fmt}")

    try:
        result = await engine.synthesize_async(
            text=body.input,
            voice=body.voice,
            language=settings.default_language,
        )
    except EngineError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:  # pragma: no cover
        log.exception("synthesis failed")
        raise HTTPException(status_code=500, detail="synthesis failed")

    try:
        data, content_type = encode_audio(result.audio, result.sample_rate, fmt)
    except RuntimeError as e:
        raise HTTPException(status_code=415, detail=str(e))

    headers = {}
    if fmt == "pcm":
        headers["X-Sample-Rate"] = str(result.sample_rate)
    return Response(content=data, media_type=content_type, headers=headers)
