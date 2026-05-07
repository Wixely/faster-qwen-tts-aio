from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from .audio_utils import SUPPORTED_FORMATS, encode_audio
from .config import Settings
from .tts_engine import EngineError, TTSEngine
from .voices import VoiceCatalog, is_valid_voice_name

log = logging.getLogger(__name__)

router = APIRouter()

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))


@router.get("/healthz", include_in_schema=False)
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


@router.get("/api/status")
async def api_status(request: Request) -> JSONResponse:
    engine: TTSEngine = request.app.state.engine
    settings: Settings = request.app.state.settings
    return JSONResponse(
        {
            "version": request.app.state.version,
            "settings": {
                "default_language": settings.default_language,
                "default_voice": settings.default_voice,
                "default_response_format": settings.default_response_format,
                "max_input_chars": settings.max_input_chars,
                "wyoming_uri": settings.wyoming_uri if settings.enable_wyoming else None,
                "auth_required": bool(settings.openai_api_key),
            },
            "engine": engine.status(),
            "supported_formats": list(SUPPORTED_FORMATS),
        }
    )


@router.get("/api/voices")
async def api_list_voices(request: Request) -> JSONResponse:
    catalog: VoiceCatalog = request.app.state.catalog
    engine: TTSEngine = request.app.state.engine
    custom = [v.to_public() for v in catalog.list_custom()]
    builtins = [v.to_public() for v in catalog.list_builtins()] if engine.custom_loaded else []
    return JSONResponse(
        {
            "custom": custom,
            "builtins": builtins,
            "custom_model_loaded": engine.custom_loaded,
            "base_model_loaded": engine.base_loaded,
        }
    )


@router.get("/api/voices/{name}/audio", include_in_schema=False)
async def api_get_voice_audio(name: str, request: Request) -> Response:
    catalog: VoiceCatalog = request.app.state.catalog
    voice = catalog.get_custom(name)
    if voice is None:
        raise HTTPException(status_code=404, detail="voice not found")
    return FileResponse(str(voice.audio_path), media_type="audio/wav")


@router.post("/api/voices")
async def api_create_voice(
    request: Request,
    name: str = Form(...),
    transcript: str = Form(...),
    language: str = Form(""),
    audio: UploadFile = File(...),
) -> JSONResponse:
    catalog: VoiceCatalog = request.app.state.catalog
    settings: Settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine

    if not engine.base_loaded:
        raise HTTPException(
            status_code=400,
            detail="Custom voice cloning requires the BASE model. BASE_MODEL_ID is not loaded.",
        )

    if not is_valid_voice_name(name):
        raise HTTPException(
            status_code=400,
            detail="Invalid voice name. Use 1-64 alphanumeric, underscore, or hyphen characters.",
        )
    if not transcript.strip():
        raise HTTPException(status_code=400, detail="transcript is required")

    contents = await audio.read()
    if not contents:
        raise HTTPException(status_code=400, detail="uploaded audio is empty")

    suffix = Path(audio.filename or "audio").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = Path(tmp.name)
    try:
        tmp.write(contents)
        tmp.close()
        try:
            voice = catalog.add(
                name=name,
                language=language or settings.default_language,
                transcript=transcript,
                src_audio=tmp_path,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:  # soundfile decode error etc.
            log.exception("voice upload failed")
            raise HTTPException(status_code=400, detail=f"could not process audio: {e}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return JSONResponse(voice.to_public(), status_code=201)


@router.delete("/api/voices/{name}")
async def api_delete_voice(name: str, request: Request) -> JSONResponse:
    catalog: VoiceCatalog = request.app.state.catalog
    if not catalog.delete(name):
        raise HTTPException(status_code=404, detail="voice not found")
    return JSONResponse({"deleted": name})


@router.post("/api/synthesize")
async def api_synthesize(
    request: Request,
    text: str = Form(...),
    voice: str = Form(""),
    language: str = Form(""),
    response_format: str = Form(""),
) -> Response:
    """Convenience endpoint for the test page. Same semantics as /v1/audio/speech."""
    settings: Settings = request.app.state.settings
    engine: TTSEngine = request.app.state.engine

    if not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > settings.max_input_chars:
        raise HTTPException(
            status_code=400,
            detail=f"text exceeds max length of {settings.max_input_chars} chars",
        )

    fmt = (response_format or settings.default_response_format).lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=415, detail=f"Unsupported response_format: {fmt}")

    try:
        result = await engine.synthesize_async(
            text=text,
            voice=voice or settings.default_voice,
            language=language or settings.default_language,
        )
    except EngineError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        log.exception("synthesis failed")
        raise HTTPException(status_code=500, detail="synthesis failed")

    try:
        data, content_type = encode_audio(result.audio, result.sample_rate, fmt)
    except RuntimeError as e:
        raise HTTPException(status_code=415, detail=str(e))

    return Response(content=data, media_type=content_type)
