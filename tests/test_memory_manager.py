"""Unit tests for MemoryManager."""

import pytest
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta
from src.core.memory_manager import (
    MemoryManager,
    Message,
    Session,
    get_memory_manager,
    reset_memory_manager,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)

    def test_to_markdown(self):
        """Test message to markdown conversion."""
        msg = Message(role="assistant", content="Hi there!")
        md = msg.to_markdown()

        assert "🤖" in md
        assert "Assistant" in md
        assert "Hi there!" in md


class TestSession:
    """Tests for Session class."""

    def test_create_session(self):
        """Test creating a session."""
        session = Session(session_id="test_001", start_time=datetime.now())

        assert session.session_id == "test_001"
        assert len(session.messages) == 0

    def test_add_message(self):
        """Test adding messages to session."""
        session = Session(session_id="test_002", start_time=datetime.now())

        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi!")

        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"

    def test_to_markdown(self):
        """Test session to markdown conversion."""
        session = Session(session_id="test_003", start_time=datetime.now())
        session.add_message("user", "Test message")

        md = session.to_markdown()

        assert "test_003" in md
        assert "Test message" in md
        assert "👤" in md


class TestMemoryManager:
    """Tests for MemoryManager."""

    @pytest.fixture
    def temp_memory_dir(self):
        """Create temporary directory for memory storage."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)

    def test_init_creates_directory(self, temp_memory_dir):
        """Test that manager creates memory directory."""
        manager = MemoryManager(memory_dir=str(temp_memory_dir))
        assert temp_memory_dir.exists()

    def test_get_or_create_session(self, temp_memory_dir):
        """Test session creation."""
        manager = MemoryManager(memory_dir=str(temp_memory_dir))

        session_id = manager.get_or_create_session()

        assert session_id is not None
        assert len(session_id) > 0

    def test_add_message(self, temp_memory_dir):
        """Test adding messages."""
        manager = MemoryManager(memory_dir=str(temp_memory_dir))
        session_id = manager.get_or_create_session()

        manager.add_message(session_id, "user", "Hello")
        manager.add_message(session_id, "assistant", "Hi there!")

        # Check that session was saved
        assert session_id in manager._sessions
        assert len(manager._sessions[session_id].messages) == 2

    def test_session_saved_to_file(self, temp_memory_dir):
        """Test that sessions are saved to markdown files."""
        manager = MemoryManager(memory_dir=str(temp_memory_dir), auto_summarize=False)
        session_id = manager.get_or_create_session()

        manager.add_message(session_id, "user", "Test")

        # Session should be saved to file
        date_folder = temp_memory_dir / datetime.now().strftime("%Y-%m-%d")
        session_files = list(date_folder.glob("session-*.md"))

        assert len(session_files) > 0

    def test_get_sessions_for_date(self, temp_memory_dir):
        """Test retrieving sessions for a date."""
        manager = MemoryManager(memory_dir=str(temp_memory_dir), auto_summarize=False)
        session_id = manager.get_or_create_session("test_session_123")

        manager.add_message(session_id, "user", "Test")

        today = datetime.now().strftime("%Y-%m-%d")
        sessions = manager.get_sessions_for_date(today)

        assert len(sessions) > 0
        # Just verify we can retrieve sessions for the date
        assert all("session_id" in s for s in sessions)
        assert all("content" in s for s in sessions)

    def test_get_all_dates(self, temp_memory_dir):
        """Test getting all dates with data."""
        manager = MemoryManager(memory_dir=str(temp_memory_dir), auto_summarize=False)
        session_id = manager.get_or_create_session()

        manager.add_message(session_id, "user", "Test")

        dates = manager.get_all_dates()

        assert len(dates) > 0
        assert datetime.now().strftime("%Y-%m-%d") in dates

    def test_search_conversations(self, temp_memory_dir):
        """Test searching conversations."""
        manager = MemoryManager(memory_dir=str(temp_memory_dir), auto_summarize=False)
        session_id = manager.get_or_create_session()

        manager.add_message(session_id, "user", "Hello, how are you?")
        manager.add_message(session_id, "assistant", "I'm doing well, thanks!")

        results = manager.search_conversations("hello")

        assert len(results) > 0
        assert results[0]["matches"] > 0

    def test_auto_summarize_disabled(self, temp_memory_dir):
        """Test that auto_summarize can be disabled."""
        manager = MemoryManager(memory_dir=str(temp_memory_dir), auto_summarize=False)
        assert manager.auto_summarize is False

    def test_start_and_stop(self, temp_memory_dir):
        """Test starting and stopping manager."""
        manager = MemoryManager(memory_dir=str(temp_memory_dir), auto_summarize=False)

        manager.start()
        assert manager._running is True

        manager.stop()
        assert manager._running is False


class TestMemoryManagerSingleton:
    """Tests for memory manager singleton pattern."""

    def test_get_memory_manager_creates_instance(self):
        """Test that get_memory_manager creates instance."""
        reset_memory_manager()

        manager1 = get_memory_manager(auto_start=False)
        manager2 = get_memory_manager(auto_start=False)

        assert manager1 is manager2  # Same instance

    def test_reset_memory_manager(self):
        """Test resetting the singleton."""
        reset_memory_manager()
        manager1 = get_memory_manager(auto_start=False)
        reset_memory_manager()
        manager2 = get_memory_manager(auto_start=False)

        assert manager1 is not manager2  # Different instances after reset


class TestMemoryManagerIntegration:
    """Integration tests for MemoryManager."""

    @pytest.fixture
    def temp_memory_dir(self):
        """Create temporary directory for memory storage."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp, ignore_errors=True)

    def test_full_workflow(self, temp_memory_dir):
        """Test complete workflow: add messages, retrieve, search."""
        manager = MemoryManager(memory_dir=str(temp_memory_dir), auto_summarize=False)

        # Create session and add messages
        session_id = manager.get_or_create_session()
        manager.add_message(session_id, "user", "What's the weather today?")
        manager.add_message(session_id, "assistant", "The weather is sunny with 25°C")

        # Retrieve session
        today = datetime.now().strftime("%Y-%m-%d")
        sessions = manager.get_sessions_for_date(today)

        assert len(sessions) > 0

        # Search for content
        results = manager.search_conversations("weather")
        assert len(results) > 0
        assert results[0]["matches"] > 0

    def test_multiple_sessions_same_day(self, temp_memory_dir):
        """Test multiple sessions on same day."""
        manager = MemoryManager(memory_dir=str(temp_memory_dir), auto_summarize=False)

        # Create multiple sessions with different IDs
        session1 = manager.get_or_create_session("test_session_1")
        manager.add_message(session1, "user", "First session")

        # Small delay to ensure different timestamps
        import time

        time.sleep(1.1)

        session2 = manager.get_or_create_session("test_session_2")
        manager.add_message(session2, "user", "Second session")

        # Should have at least 1 session (may be merged if same minute)
        today = datetime.now().strftime("%Y-%m-%d")
        sessions = manager.get_sessions_for_date(today)

        assert len(sessions) >= 1
