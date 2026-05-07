from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import soundfile as sf

from .config import Settings


_VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")


def is_valid_voice_name(name: str) -> bool:
    return bool(_VOICE_NAME_RE.match(name or ""))


@dataclass
class CustomVoice:
    name: str
    language: str
    transcript: str
    audio_path: Path
    created_at: str

    @property
    def kind(self) -> str:
        return "custom"

    def to_public(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "language": self.language,
            "transcript": self.transcript,
            "created_at": self.created_at,
        }


@dataclass
class BuiltinVoice:
    name: str

    @property
    def kind(self) -> str:
        return "builtin"

    def to_public(self) -> dict:
        return {"name": self.name, "kind": self.kind}


class VoiceCatalog:
    """File-backed catalog of custom voices, plus a static list of built-in speakers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._custom: Dict[str, CustomVoice] = {}
        self.reload()

    # ---- discovery ------------------------------------------------------------------

    def reload(self) -> None:
        custom: Dict[str, CustomVoice] = {}
        root = self.settings.voices_dir
        if root.exists():
            for entry in sorted(root.iterdir()):
                if not entry.is_dir():
                    continue
                meta_file = entry / "meta.json"
                if not meta_file.is_file():
                    continue
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                audio_rel = meta.get("audio_filename") or "audio.wav"
                audio_path = entry / audio_rel
                if not audio_path.is_file():
                    continue
                voice = CustomVoice(
                    name=meta.get("name") or entry.name,
                    language=meta.get("language") or self.settings.default_language,
                    transcript=meta.get("transcript") or "",
                    audio_path=audio_path,
                    created_at=meta.get("created_at") or "",
                )
                custom[voice.name] = voice
        self._custom = custom

    # ---- queries --------------------------------------------------------------------

    def list_custom(self) -> List[CustomVoice]:
        return list(self._custom.values())

    def list_builtins(self) -> List[BuiltinVoice]:
        return [BuiltinVoice(name=n) for n in self.settings.builtin_speakers]

    def get_custom(self, name: str) -> Optional[CustomVoice]:
        return self._custom.get(name)

    def get_builtin(self, name: str) -> Optional[BuiltinVoice]:
        if name in self.settings.builtin_speakers:
            return BuiltinVoice(name=name)
        return None

    def all_names(self) -> List[str]:
        names = set(self._custom.keys()) | set(self.settings.builtin_speakers)
        return sorted(names)

    # ---- mutations ------------------------------------------------------------------

    def add(self, name: str, language: str, transcript: str, src_audio: Path) -> CustomVoice:
        if not is_valid_voice_name(name):
            raise ValueError(
                "voice name must be 1-64 chars: alphanumeric, underscore, or hyphen "
                "(must start with a letter or digit)."
            )
        if name in self._custom:
            raise ValueError(f"voice {name!r} already exists")
        if name in set(self.settings.builtin_speakers):
            raise ValueError(f"name {name!r} collides with a built-in speaker")
        if not transcript.strip():
            raise ValueError("transcript is required for voice cloning")

        # Re-encode the supplied audio to a clean mono 24 kHz WAV so the engine has
        # consistent input regardless of what the user uploaded.
        audio, sr = sf.read(str(src_audio), always_2d=False)
        if hasattr(audio, "ndim") and audio.ndim == 2:
            audio = audio.mean(axis=1)

        voice_dir = self.settings.voices_dir / name
        voice_dir.mkdir(parents=True, exist_ok=False)
        audio_path = voice_dir / "audio.wav"
        sf.write(str(audio_path), audio, sr, format="WAV", subtype="PCM_16")

        meta = {
            "name": name,
            "language": language or self.settings.default_language,
            "transcript": transcript.strip(),
            "audio_filename": "audio.wav",
            "sample_rate": int(sr),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (voice_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (voice_dir / "transcript.txt").write_text(meta["transcript"], encoding="utf-8")

        voice = CustomVoice(
            name=name,
            language=meta["language"],
            transcript=meta["transcript"],
            audio_path=audio_path,
            created_at=meta["created_at"],
        )
        self._custom[name] = voice
        return voice

    def delete(self, name: str) -> bool:
        voice = self._custom.pop(name, None)
        if voice is None:
            return False
        voice_dir = voice.audio_path.parent
        if voice_dir.exists() and voice_dir.is_dir():
            shutil.rmtree(voice_dir, ignore_errors=True)
        return True
