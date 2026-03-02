"""Unit tests for VoiceDaemon."""

import pytest
import threading
import time
from unittest.mock import Mock, MagicMock, patch
from src.core.voice_daemon import (
    VoiceDaemon,
    Priority,
    SpeechRequest,
    get_voice_daemon,
    reset_voice_daemon,
)


class TestSpeechRequest:
    """Tests for SpeechRequest dataclass."""

    def test_create_high_priority_request(self):
        """Test creating a high priority speech request."""
        request = SpeechRequest(text="Hello, this is important", priority=Priority.HIGH)

        assert request.text == "Hello, this is important"
        assert request.priority == Priority.HIGH
        assert request.language == "auto"
        assert request.audio_type == "chat"

    def test_create_normal_priority_request(self):
        """Test creating a normal priority speech request."""
        request = SpeechRequest(
            text="Reading file content",
            priority=Priority.NORMAL,
            audio_type="file"  # Must be set explicitly
        )
        
        assert request.priority == Priority.NORMAL
        assert request.audio_type == "file"

    def test_priority_comparison(self):
        """Test that HIGH priority is less than NORMAL."""
        high = SpeechRequest("high", Priority.HIGH)
        normal = SpeechRequest("normal", Priority.NORMAL)

        assert high < normal

    def test_custom_parameters(self):
        """Test creating request with custom parameters."""
        request = SpeechRequest(
            text="Custom speech",
            priority=Priority.HIGH,
            language="zh",
            speaker_id="custom_voice",
            speed=1.5,
            metadata={"custom": "data"},
        )

        assert request.language == "zh"
        assert request.speaker_id == "custom_voice"
        assert request.speed == 1.5
        assert request.metadata == {"custom": "data"}


class TestVoiceDaemon:
    """Tests for VoiceDaemon."""

    def test_init(self, mock_tts_tool):
        """Test daemon initialization."""
        daemon = VoiceDaemon(mock_tts_tool)

        assert daemon.tts_tool == mock_tts_tool
        assert daemon.is_running is False
        assert daemon.is_speaking is False
        assert daemon.queue_size == 0
        assert daemon.stats["total_speeches"] == 0

    def test_set_tts_tool(self):
        """Test setting TTS tool."""
        daemon = VoiceDaemon(None)
        mock_tool = Mock()

        daemon.set_tts_tool(mock_tool)

        assert daemon.tts_tool == mock_tool

    def test_enqueue_empty_text(self, mock_tts_tool):
        """Test enqueueing empty text fails."""
        daemon = VoiceDaemon(mock_tts_tool)

        result = daemon.enqueue(text="")

        assert result["success"] is False
        assert "No text" in result["error"]

    def test_enqueue_when_not_running(self, mock_tts_tool):
        """Test enqueueing when daemon is not running."""
        daemon = VoiceDaemon(mock_tts_tool)

        result = daemon.enqueue(text="Hello")

        assert result["success"] is False
        assert "not running" in result["error"]

    def test_get_status(self, mock_tts_tool):
        """Test getting daemon status."""
        daemon = VoiceDaemon(mock_tts_tool)

        status = daemon.get_status()

        assert "is_running" in status
        assert "is_speaking" in status
        assert "queue_size" in status
        assert "stats" in status
        assert status["is_running"] is False
        assert status["is_speaking"] is False

    def test_was_interrupted_by_high_priority(self, mock_tts_tool):
        """Test interruption flag."""
        daemon = VoiceDaemon(mock_tts_tool)

        assert daemon.was_interrupted_by_high_priority() is False
        daemon.interrupted_by_high_priority.set()
        assert daemon.was_interrupted_by_high_priority() is True
        daemon.clear_interruption_flag()
        assert daemon.was_interrupted_by_high_priority() is False

    def test_speak_immediately(self, mock_tts_tool):
        """Test speak_immediately convenience method."""
        daemon = VoiceDaemon(mock_tts_tool)

        result = daemon.speak_immediately("Hello")

        assert result["success"] is False  # Not running
        assert "not running" in result["error"]

    def test_speak_file_content(self, mock_tts_tool):
        """Test speak_file_content convenience method."""
        daemon = VoiceDaemon(mock_tts_tool)

        result = daemon.speak_file_content("File content")

        assert result["success"] is False  # Not running


class TestVoiceDaemonSingleton:
    """Tests for voice daemon singleton pattern."""

    def test_get_voice_daemon_creates_instance(self):
        """Test that get_voice_daemon creates instance."""
        reset_voice_daemon()

        daemon1 = get_voice_daemon()
        daemon2 = get_voice_daemon()

        assert daemon1 is daemon2  # Same instance

    def test_reset_voice_daemon(self):
        """Test resetting the singleton."""
        reset_voice_daemon()
        daemon1 = get_voice_daemon()
        reset_voice_daemon()
        daemon2 = get_voice_daemon()

        assert daemon1 is not daemon2  # Different instances after reset


class TestPriorityEnum:
    """Tests for Priority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        assert Priority.HIGH == 1
        assert Priority.NORMAL == 2
        assert Priority.LOW == 3

    def test_priority_ordering(self):
        """Test priority ordering."""
        assert Priority.HIGH < Priority.NORMAL
        assert Priority.NORMAL < Priority.LOW
        assert Priority.HIGH < Priority.LOW
