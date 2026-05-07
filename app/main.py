from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .config import get_settings
from .openai_api import router as openai_router
from .tts_engine import TTSEngine
from .voices import VoiceCatalog
from .web import router as web_router
from .wyoming_server import run_wyoming_server

log = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    log.info("faster-qwen-tts-aio v%s starting", __version__)

    catalog = VoiceCatalog(settings)
    engine = TTSEngine(settings, catalog)

    app.state.version = __version__
    app.state.settings = settings
    app.state.catalog = catalog
    app.state.engine = engine

    if settings.eager_load:
        try:
            await asyncio.to_thread(engine.ensure_loaded)
        except Exception:
            log.exception("Eager model load failed; will retry on first request.")

    wyoming_task = None
    if settings.enable_wyoming:
        wyoming_task = asyncio.create_task(
            run_wyoming_server(engine, settings),
            name="wyoming-server",
        )

    try:
        yield
    finally:
        if wyoming_task is not None:
            wyoming_task.cancel()
            try:
                await wyoming_task
            except (asyncio.CancelledError, Exception):
                pass
        engine.close()
        log.info("faster-qwen-tts-aio shutdown complete")


app = FastAPI(
    title="faster-qwen-tts-aio",
    version=__version__,
    lifespan=lifespan,
)

app.include_router(openai_router)
app.include_router(web_router)


def main() -> None:  # pragma: no cover
    """Entrypoint for `python -m app.main`."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.http_host,
        port=settings.http_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
