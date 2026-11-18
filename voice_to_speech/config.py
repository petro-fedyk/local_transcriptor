"""Local paths and validated persisted settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import os
from pathlib import Path
import threading


DATA_DIR = Path(os.environ.get("VTS_DATA_DIR") or "~/.local/share/voice-to-speech").expanduser()
RECORDINGS = DATA_DIR / "recordings"
MODELS = DATA_DIR / "models"
LLM_MODEL_REPO = "ggml-org/gemma-3-4b-it-GGUF"
LLM_MODEL_FILE = "gemma-3-4b-it-Q4_K_M.gguf"


@dataclass(frozen=True)
class Config:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("VTS_DATA_DIR") or "~/.local/share/voice-to-speech").expanduser())
    port: int = field(default_factory=lambda: int(os.environ.get("VTS_PORT") or "8765"))
    static_dir: Path = Path(__file__).resolve().parent.parent / "static"

    @property
    def recordings(self) -> Path:
        return self.data_dir / "recordings"

    @property
    def models(self) -> Path:
        return self.data_dir / "models"

    @property
    def llm_dir(self) -> Path:
        return self.data_dir / "llm"

    @property
    def llm_server_path(self) -> Path:
        return Path(os.environ.get("VTS_LLAMA_SERVER") or self.llm_dir / "bin" / "llama-server").expanduser()

    @property
    def llm_model_path(self) -> Path:
        return Path(os.environ.get("VTS_LLM_MODEL") or self.models / "llm" / LLM_MODEL_FILE).expanduser()

    @property
    def settings_path(self) -> Path:
        return self.data_dir / "settings.json"


@dataclass
class Settings:
    language: str = "uk"
    glossary: str = ""
    live_model: str = "large-v3-turbo"
    file_model: str = "large-v3"
    system_source: str = "default"
    mic_source: str = "default"
    echo_dedupe: bool = True
    auto_hq: bool = False
    idle_shutdown_minutes: int = 15


class SettingsError(ValueError):
    pass


def validate_settings(values: dict) -> dict:
    if not isinstance(values, dict):
        raise SettingsError("Потрібен об’єкт налаштувань.")
    allowed = {field.name for field in fields(Settings)}
    if set(values) - allowed:
        raise SettingsError("Невідомий параметр налаштувань.")
    result = dict(values)
    if "language" in result and result["language"] not in ("auto", "uk", "en"):
        raise SettingsError("Мова має бути auto, uk або en.")
    if "glossary" in result and (not isinstance(result["glossary"], str) or len(result["glossary"]) > 500):
        raise SettingsError("Словник має містити не більше 500 символів.")
    for key in ("live_model", "file_model"):
        if key in result and result[key] not in ("large-v3", "large-v3-turbo"):
            raise SettingsError("Невідома модель.")
    if "system_source" in result and (not isinstance(result["system_source"], str) or
       not result["system_source"] or len(result["system_source"]) > 250 or
       result["system_source"] == "app:" or
       any(ord(ch) < 32 for ch in result["system_source"])):
        raise SettingsError("Некоректне системне джерело звуку.")
    if "mic_source" in result and (not isinstance(result["mic_source"], str) or
       not result["mic_source"] or result["mic_source"].startswith("app:") or
       len(result["mic_source"]) > 250 or any(ord(ch) < 32 for ch in result["mic_source"])):
        raise SettingsError("Некоректне джерело мікрофона.")
    for key in ("echo_dedupe", "auto_hq"):
        if key in result and type(result[key]) is not bool:
            raise SettingsError("Параметр має бути так або ні.")
    if "idle_shutdown_minutes" in result and (type(result["idle_shutdown_minutes"]) is not int or
       not 0 <= result["idle_shutdown_minutes"] <= 240):
        raise SettingsError("Автовимкнення має бути від 0 до 240 хвилин.")
    return result


class SettingsStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.RLock()
        try:
            values = validate_settings(json.loads(self.path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            values = {}
        except (ValueError, OSError):
            values = {}
        self.value = Settings(**values)

    def snapshot(self) -> Settings:
        with self.lock:
            return Settings(**asdict(self.value))

    def update(self, patch: dict) -> Settings:
        patch = validate_settings(patch)
        with self.lock:
            next_value = Settings(**{**asdict(self.value), **patch})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(asdict(next_value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.path)
            self.value = next_value
            return self.snapshot()
