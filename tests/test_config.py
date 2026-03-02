"""Unit tests for Pydantic settings."""

import pytest
from pathlib import Path
import tempfile
import yaml
from src.config.settings import (
    Settings,
    TTSSettings,
    LLMSettings,
    LoggingSettings,
    get_settings,
    reset_settings,
)


class TestTTSSettings:
    """Tests for TTS settings."""

    def test_default_values(self):
        """Test default TTS settings."""
        tts = TTSSettings()
        assert tts.engine == "edge_tts"
        assert tts.edge_voice == "en-US-AriaNeural"
        assert tts.voice_output == "web"

    def test_valid_engines(self):
        """Test valid TTS engine values."""
        for engine in ["qwen_tts", "edge_tts", "coqui", "pyttsx3"]:
            tts = TTSSettings(engine=engine)
            assert tts.engine == engine

    def test_rate_bounds(self):
        """Test rate field bounds."""
        tts = TTSSettings(rate=60)
        assert tts.rate == 60

        tts = TTSSettings(rate=360)
        assert tts.rate == 360

    def test_volume_bounds(self):
        """Test volume field bounds."""
        tts = TTSSettings(volume=0.1)
        assert tts.volume == 0.1

        tts = TTSSettings(volume=2.0)
        assert tts.volume == 2.0


class TestLLMSettings:
    """Tests for LLM settings."""

    def test_default_values(self):
        """Test default LLM settings."""
        llm = LLMSettings()
        assert llm.default_provider == "llamacpp"
        assert llm.base_url == "http://localhost:8080"
        assert llm.auto_detect_models is True

    def test_valid_providers(self):
        """Test valid provider values."""
        for provider in ["vllm", "llamacpp", "ollama", "google", "anthropic", "openai"]:
            llm = LLMSettings(default_provider=provider)
            assert llm.default_provider == provider

    def test_timeout_bounds(self):
        """Test timeout field bounds."""
        llm = LLMSettings(timeout=10)
        assert llm.timeout == 10

        llm = LLMSettings(timeout=300)
        assert llm.timeout == 300

    def test_temperature_bounds(self):
        """Test temperature field bounds."""
        llm = LLMSettings(temperature=0.0)
        assert llm.temperature == 0.0

        llm = LLMSettings(temperature=2.0)
        assert llm.temperature == 2.0


class TestLoggingSettings:
    """Tests for logging settings."""

    def test_default_values(self):
        """Test default logging settings."""
        logging = LoggingSettings()
        assert logging.level == "INFO"
        assert logging.file is None

    def test_valid_levels(self):
        """Test valid log level values."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            logging = LoggingSettings(level=level)
            assert logging.level == level


class TestSettings:
    """Tests for main Settings class."""

    def test_default_settings(self):
        """Test creating settings with defaults."""
        settings = Settings()
        assert settings.app.name == "Talkie Voice Assistant"
        assert settings.tts.engine == "edge_tts"
        assert settings.llm.default_provider == "llamacpp"

    def test_from_yaml(self, temp_dir):
        """Test loading settings from YAML."""
        config = {
            "tts": {"engine": "coqui", "edge_voice": "en-GB-SoniaNeural"},
            "llm": {"default_provider": "vllm", "base_url": "http://localhost:8000"},
        }

        yaml_path = temp_dir / "test_settings.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)

        settings = Settings.from_yaml(str(yaml_path))
        assert settings.tts.engine == "coqui"
        assert settings.llm.default_provider == "vllm"
        assert settings.llm.base_url == "http://localhost:8000"

    def test_from_yaml_missing_file(self):
        """Test loading from non-existent file uses defaults."""
        settings = Settings.from_yaml("/nonexistent/path.yaml")
        assert settings.tts.engine == "edge_tts"  # Default

    def test_validate_missing_paths(self):
        """Test validation catches missing paths."""
        settings = Settings(
            stt={"whisper": {"binary_path": "/nonexistent/path"}},
            llm={"binary_path": "/also/nonexistent"},
        )

        errors = settings.validate()
        assert "stt.whisper.binary_path" in errors
        assert "llm.binary_path" in errors

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        settings = Settings()
        errors = settings.validate()
        # Should have no errors for default settings
        assert len(errors) == 0


class TestSettingsSingleton:
    """Tests for settings singleton pattern."""

    def test_get_settings_returns_same_instance(self):
        """Test that get_settings returns same instance."""
        reset_settings()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_reset_settings(self):
        """Test resetting settings."""
        reset_settings()
        settings1 = get_settings()
        reset_settings()
        settings2 = get_settings()
        assert settings1 is not settings2
