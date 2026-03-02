"""Unit tests for ReadingPositionManager."""

import pytest
from pathlib import Path
import json
from src.core.reading_position_manager import ReadingPosition, ReadingPositionManager


class TestReadingPosition:
    """Tests for ReadingPosition dataclass."""

    def test_create_position(self):
        """Test creating a new reading position."""
        pos = ReadingPosition(
            file_id="test_123",
            file_path="/tmp/test.txt",
            file_name="test.txt",
            total_words=1000,
        )

        assert pos.file_id == "test_123"
        assert pos.total_words == 1000
        assert pos.current_word_index == 0
        assert pos.is_reading is False
        assert pos.is_paused is False

    def test_progress_percent(self):
        """Test progress percentage calculation."""
        pos = ReadingPosition(
            file_id="test_123",
            file_path="/tmp/test.txt",
            file_name="test.txt",
            total_words=1000,
            current_word_index=250,
        )

        assert pos.get_progress_percent() == 25

    def test_progress_percent_zero_total(self):
        """Test progress with zero total words."""
        pos = ReadingPosition(
            file_id="test_123",
            file_path="/tmp/test.txt",
            file_name="test.txt",
            total_words=0,
            current_word_index=0,
        )

        assert pos.get_progress_percent() == 0

    def test_progress_text(self):
        """Test progress text formatting."""
        pos = ReadingPosition(
            file_id="test_123",
            file_path="/tmp/test.txt",
            file_name="test.txt",
            total_words=1000,
            current_word_index=500,
        )

        text = pos.get_progress_text()
        assert "500" in text
        assert "1000" in text
        assert "50%" in text

    def test_to_dict(self):
        """Test serialization to dict."""
        pos = ReadingPosition(
            file_id="test_123",
            file_path="/tmp/test.txt",
            file_name="test.txt",
            total_words=1000,
            current_word_index=100,
        )

        data = pos.to_dict()
        assert data["file_id"] == "test_123"
        assert data["total_words"] == 1000
        assert data["current_word_index"] == 100

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "file_id": "test_456",
            "file_path": "/tmp/test2.txt",
            "file_name": "test2.txt",
            "total_words": 2000,
            "current_word_index": 500,
            "loaded_start_word": 0,
            "loaded_end_word": 0,
            "chunk_size": 100,
            "chunks_per_load": 3,
            "is_reading": True,
            "is_paused": False,
            "last_updated": "2024-01-01T00:00:00",
        }

        pos = ReadingPosition.from_dict(data)
        assert pos.file_id == "test_456"
        assert pos.total_words == 2000
        assert pos.current_word_index == 500
        assert pos.is_reading is True


class TestReadingPositionManager:
    """Tests for ReadingPositionManager."""

    def test_init_creates_directory(self, temp_dir):
        """Test that manager creates memory directory if it doesn't exist."""
        manager = ReadingPositionManager(str(temp_dir))
        assert temp_dir.exists()

    def test_save_and_get_position(self, temp_dir):
        """Test saving and retrieving a position."""
        manager = ReadingPositionManager(str(temp_dir))

        pos = ReadingPosition(
            file_id="test_789",
            file_path="/tmp/test3.txt",
            file_name="test3.txt",
            total_words=500,
            current_word_index=100,
        )

        manager.save_position(pos)
        retrieved = manager.get_position("test_789")

        assert retrieved is not None
        assert retrieved.file_id == "test_789"
        assert retrieved.current_word_index == 100

    def test_get_nonexistent_position(self, temp_dir):
        """Test getting a position that doesn't exist."""
        manager = ReadingPositionManager(str(temp_dir))

        result = manager.get_position("nonexistent")
        assert result is None

    def test_delete_position(self, temp_dir):
        """Test deleting a position."""
        manager = ReadingPositionManager(str(temp_dir))

        pos = ReadingPosition(
            file_id="test_delete",
            file_path="/tmp/test4.txt",
            file_name="test4.txt",
            total_words=300,
        )

        manager.save_position(pos)
        assert manager.has_position("test_delete")

        manager.delete_position("test_delete")
        assert not manager.has_position("test_delete")

    def test_get_unfinished_readings(self, temp_dir):
        """Test getting unfinished readings."""
        manager = ReadingPositionManager(str(temp_dir))

        # Create finished reading
        pos1 = ReadingPosition(
            file_id="finished",
            file_path="/tmp/finished.txt",
            file_name="finished.txt",
            total_words=100,
            current_word_index=100,
        )

        # Create unfinished reading
        pos2 = ReadingPosition(
            file_id="unfinished",
            file_path="/tmp/unfinished.txt",
            file_name="unfinished.txt",
            total_words=200,
            current_word_index=50,
        )

        manager.save_position(pos1)
        manager.save_position(pos2)

        unfinished = manager.get_unfinished_readings()
        assert len(unfinished) == 1
        assert unfinished[0].file_id == "unfinished"

    def test_update_word_index(self, temp_dir):
        """Test updating word index."""
        manager = ReadingPositionManager(str(temp_dir))

        pos = ReadingPosition(
            file_id="test_update",
            file_path="/tmp/test5.txt",
            file_name="test5.txt",
            total_words=1000,
        )

        manager.save_position(pos)
        manager.update_word_index("test_update", 250)

        updated = manager.get_position("test_update")
        assert updated.current_word_index == 250

    def test_mark_paused(self, temp_dir):
        """Test marking a file as paused."""
        manager = ReadingPositionManager(str(temp_dir))

        pos = ReadingPosition(
            file_id="test_pause",
            file_path="/tmp/test6.txt",
            file_name="test6.txt",
            total_words=500,
            is_reading=True,
        )

        manager.save_position(pos)
        manager.mark_paused("test_pause")

        updated = manager.get_position("test_pause")
        assert updated.is_paused is True
        assert updated.is_reading is False

    def test_mark_resumed(self, temp_dir):
        """Test marking a file as resumed."""
        manager = ReadingPositionManager(str(temp_dir))

        pos = ReadingPosition(
            file_id="test_resume",
            file_path="/tmp/test7.txt",
            file_name="test7.txt",
            total_words=500,
            is_paused=True,
            is_reading=False,
        )

        manager.save_position(pos)
        manager.mark_resumed("test_resume")

        updated = manager.get_position("test_resume")
        assert updated.is_paused is False
        assert updated.is_reading is True

    def test_mark_completed(self, temp_dir):
        """Test marking a file as completed."""
        manager = ReadingPositionManager(str(temp_dir))

        pos = ReadingPosition(
            file_id="test_complete",
            file_path="/tmp/test8.txt",
            file_name="test8.txt",
            total_words=1000,
            current_word_index=500,
        )

        manager.save_position(pos)
        manager.mark_completed("test_complete")

        updated = manager.get_position("test_complete")
        assert updated.current_word_index == 1000
        assert updated.is_reading is False

    def test_persistence_across_instances(self, temp_dir):
        """Test that positions persist across manager instances."""
        # Create first manager and save position
        manager1 = ReadingPositionManager(str(temp_dir))
        pos = ReadingPosition(
            file_id="persist_test",
            file_path="/tmp/persist.txt",
            file_name="persist.txt",
            total_words=800,
            current_word_index=200,
        )
        manager1.save_position(pos)

        # Create new manager instance and retrieve
        manager2 = ReadingPositionManager(str(temp_dir))
        retrieved = manager2.get_position("persist_test")

        assert retrieved is not None
        assert retrieved.current_word_index == 200
        assert retrieved.total_words == 800

    def test_update_loaded_range(self, temp_dir):
        """Test updating loaded range."""
        manager = ReadingPositionManager(str(temp_dir))

        pos = ReadingPosition(
            file_id="test_range",
            file_path="/tmp/test9.txt",
            file_name="test9.txt",
            total_words=1000,
        )

        manager.save_position(pos)
        manager.update_loaded_range("test_range", 100, 300)

        updated = manager.get_position("test_range")
        assert updated.loaded_start_word == 100
        assert updated.loaded_end_word == 300
