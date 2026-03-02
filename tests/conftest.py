"""Pytest configuration and fixtures for Talkie tests."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import Mock, AsyncMock, patch
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def mock_session_memory(temp_dir):
    """Create a mock session memory."""
    memory = Mock()
    memory.memory_dir = temp_dir
    memory.attachments = []
    memory.find_attachment_by_name = Mock(return_value=None)
    memory.get_last_attachment = Mock(return_value=None)
    memory.get_attachment_content = Mock(return_value=None)
    return memory


@pytest.fixture
def mock_voice_daemon():
    """Create a mock voice daemon."""
    daemon = Mock()
    daemon.is_running = True
    daemon.is_speaking = False
    daemon.speech_queue = Mock()
    daemon.speech_queue.qsize = Mock(return_value=0)
    daemon.stop_event = Mock()
    daemon.stop_event.is_set = Mock(return_value=False)
    daemon.on_audio_ready = None
    return daemon


@pytest.fixture
def mock_tts_tool():
    """Create a mock TTS tool."""
    tool = AsyncMock()
    tool.execute = AsyncMock(
        return_value={
            "success": True,
            "audio_file": "/tmp/test.wav",
            "audio_process": Mock(pid=12345, poll=Mock(return_value=None)),
        }
    )
    return tool


@pytest.fixture
def sample_config():
    """Return a sample configuration dictionary."""
    return {
        "tts": {
            "engine": "edge_tts",
            "voice": "en-US-AriaNeural",
            "voice_output": "web",
        },
        "llm": {"default_provider": "llamacpp", "auto_detect_models": True},
        "weather": {"auto_detect_location": True},
        "logging": {"level": "INFO"},
    }


@pytest.fixture
def sample_text():
    """Return sample text for testing."""
    return (
        "This is a sample text for testing. It contains multiple sentences. "
        "The quick brown fox jumps over the lazy dog."
    )


@pytest.fixture
def sample_chinese_text():
    """Return sample Chinese text for testing."""
    return "这是一个测试文本。它包含多个句子。快速brown 狐狸跳过懒惰的狗。"
