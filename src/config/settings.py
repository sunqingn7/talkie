"""
Pydantic-based configuration management for Talkie Voice Assistant.

Provides:
- Type-safe configuration with validation
- Auto-loading from YAML and .env files
- Default values for all settings
- Error messages for invalid config
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseModel):
    """Application settings."""

    name: str = "Talkie Voice Assistant"
    version: str = "1.0.0"
    debug: bool = False


class AudioSettings(BaseModel):
    """Audio settings."""

    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    chunk_size: int = Field(default=4096, ge=1024, le=8192)
    channels: int = Field(default=1, ge=1, le=2)
    format: str = "int16"


class WhisperSettings(BaseModel):
    """Whisper.cpp settings."""

    server_url: str = "http://localhost:8081"
    model: str = "base.en"
    language: str = "en"
    binary_path: Optional[str] = None
    model_path: Optional[str] = None


class STTSettings(BaseModel):
    """Speech-to-text settings."""

    primary: Literal["whisper"] = "whisper"
    whisper: WhisperSettings = WhisperSettings()


class TTSSettings(BaseModel):
    """Text-to-speech settings."""

    engine: Literal["qwen_tts", "edge_tts", "coqui", "pyttsx3"] = "edge_tts"
    edge_voice: str = "en-US-AriaNeural"
    coqui_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    coqui_device: Literal["auto", "cuda", "cpu"] = "auto"
    voice_output: Literal["web", "local"] = "web"
    voice_id: Optional[str] = None
    rate: int = Field(default=180, ge=60, le=360)
    volume: float = Field(default=1.0, ge=0.1, le=2.0)
    default_language: str = "auto"
    speaker_id: Optional[str] = None
    speakers_list: List[str] = []


class MusicSettings(BaseModel):
    """Music player settings."""

    output: Literal["web", "local"] = "web"
    stream_cache_expiry: int = Field(default=600, ge=60)
    max_concurrent_streams: int = Field(default=3, ge=1, le=10)


class LLMSettings(BaseModel):
    """LLM provider settings."""

    default_provider: Literal[
        "vllm", "llamacpp", "ollama", "google", "anthropic", "openai"
    ] = "llamacpp"
    base_url: str = "http://localhost:8080"
    model: str = "llama-3.2-3b-instruct"
    binary_path: Optional[str] = None
    model_path: Optional[str] = None
    auto_detect_models: bool = True
    timeout: int = Field(default=120, ge=10, le=300)
    max_tokens: int = Field(default=2048, ge=256, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class WeatherSettings(BaseModel):
    """Weather settings."""

    api_key: Optional[str] = None
    auto_detect_location: bool = True


class WebSearchSettings(BaseModel):
    """Web search settings."""

    tavily_api_key: Optional[str] = None
    use_duckduckgo_fallback: bool = True


class WakeWordSettings(BaseModel):
    """Wake word detection settings."""

    phrases: List[str] = ["hey talkie", "ok talkie", "talkie"]
    simulation_mode: bool = True
    sensitivity: float = Field(default=0.7, ge=0.1, le=1.0)


class VoiceActivitySettings(BaseModel):
    """Voice activity detection settings."""

    threshold: float = Field(default=0.5, ge=0.1, le=1.0)
    frame_duration_ms: int = Field(default=30, ge=10, le=100)
    padding_duration_ms: int = Field(default=300, ge=0, le=1000)


class LoggingSettings(BaseModel):
    """Logging settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file: Optional[str] = None


class WebSettings(BaseModel):
    """Web server settings."""

    host: str = "0.0.0.0"
    port: int = Field(default=8082, ge=1024, le=65535)


class Settings(BaseSettings):
    """
    Main settings class for Talkie Voice Assistant.

    Loads from:
    1. Environment variables (prefix: TALKIE_)
    2. YAML config file (config/settings.yaml)
    3. .env file
    4. Default values
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="TALKIE_", env_nested_delimiter="__", extra="ignore"
    )

    # Core settings
    app: AppSettings = AppSettings()
    audio: AudioSettings = AudioSettings()
    stt: STTSettings = STTSettings()
    tts: TTSSettings = TTSSettings()
    music: MusicSettings = MusicSettings()
    llm: LLMSettings = LLMSettings()
    weather: WeatherSettings = WeatherSettings()
    web_search: WebSearchSettings = WebSearchSettings()
    wake_word: WakeWordSettings = WakeWordSettings()
    voice_activity: VoiceActivitySettings = VoiceActivitySettings()
    logging: LoggingSettings = LoggingSettings()
    web: WebSettings = WebSettings()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Settings":
        """
        Load settings from a YAML file.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            Settings instance
        """
        import yaml

        if not Path(yaml_path).exists():
            print(f"Config file not found: {yaml_path}, using defaults")
            return cls()

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def validate(self) -> Dict[str, List[str]]:
        """
        Validate settings and return any errors.

        Returns:
            Dict mapping setting names to lists of error messages
        """
        errors: Dict[str, List[str]] = {}

        # Validate paths exist if specified
        if (
            self.stt.whisper.binary_path
            and not Path(self.stt.whisper.binary_path).exists()
        ):
            errors["stt.whisper.binary_path"] = [
                f"Path not found: {self.stt.whisper.binary_path}"
            ]

        if (
            self.stt.whisper.model_path
            and not Path(self.stt.whisper.model_path).exists()
        ):
            errors["stt.whisper.model_path"] = [
                f"Path not found: {self.stt.whisper.model_path}"
            ]

        if self.llm.binary_path and not Path(self.llm.binary_path).exists():
            errors["llm.binary_path"] = [f"Path not found: {self.llm.binary_path}"]

        if self.llm.model_path and not Path(self.llm.model_path).exists():
            errors["llm.model_path"] = [f"Path not found: {self.llm.model_path}"]

        # Validate API keys if features are enabled
        if (
            self.web_search.use_duckduckgo_fallback is False
            and not self.web_search.tavily_api_key
        ):
            errors["web_search"] = [
                "Tavily API key required when DuckDuckGo fallback is disabled"
            ]

        return errors


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Get or create the global settings instance.

    Args:
        config_path: Optional path to YAML config file

    Returns:
        Settings instance
    """
    global _settings

    if _settings is None:
        if config_path:
            _settings = Settings.from_yaml(config_path)
        else:
            # Try default config path
            default_path = Path(__file__).parent.parent / "config" / "settings.yaml"
            if default_path.exists():
                _settings = Settings.from_yaml(str(default_path))
            else:
                _settings = Settings()

        # Validate settings
        errors = _settings.validate()
        if errors:
            print("Configuration warnings:")
            for key, msgs in errors.items():
                for msg in msgs:
                    print(f"  - {key}: {msg}")

    return _settings


def reset_settings() -> None:
    """Reset the global settings instance."""
    global _settings
    _settings = None
