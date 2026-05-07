from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Storage
    data_dir: Path = Field(default=Path("/data"))

    # HTTP server
    http_host: str = "0.0.0.0"
    http_port: int = 8080

    # Wyoming server
    enable_wyoming: bool = True
    wyoming_uri: str = "tcp://0.0.0.0:10200"

    # Models
    base_model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    custom_voice_model_id: str = ""
    builtin_speakers: List[str] = ["aiden", "serena"]

    # Defaults applied when a request omits the value
    default_language: str = "English"
    default_voice: str = "aiden"
    default_response_format: str = "wav"

    # Behaviour
    eager_load: bool = True
    max_input_chars: int = 4096
    log_level: str = "INFO"

    # Auth (optional)
    openai_api_key: str = ""

    @field_validator("builtin_speakers", mode="before")
    @classmethod
    def _split_speakers(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def config_file(self) -> Path:
        return self.data_dir / "config.json"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings
