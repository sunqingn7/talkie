import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReadingPosition:
    """Represents a reading position for a file."""

    file_id: str
    file_path: str
    file_name: str
    total_words: int

    current_word_index: int = 0
    loaded_start_word: int = 0
    loaded_end_word: int = 0

    chunk_size: int = 100
    chunks_per_load: int = 3

    is_reading: bool = False
    is_paused: bool = False

    last_updated: Optional[str] = None

    def __post_init__(self) -> None:
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReadingPosition":
        """Create from dictionary for deserialization."""
        return cls(**data)

    def get_progress_percent(self) -> int:
        """Get reading progress as percentage."""
        if self.total_words == 0:
            return 0
        return int((self.current_word_index / self.total_words) * 100)

    def get_progress_text(self) -> str:
        """Get human-readable progress text."""
        percent = self.get_progress_percent()
        return f"At word {self.current_word_index} of {self.total_words} ({percent}%)"


class ReadingPositionManager:
    """
    Manages reading positions for files across sessions.
    Supports persistence across system restarts.
    """

    def __init__(self, memory_dir: Optional[str] = None):
        import tempfile

        self.memory_dir = Path(
            memory_dir or os.path.join(tempfile.gettempdir(), "talkie_sessions")
        )
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.positions_file = self.memory_dir / "reading_positions.json"
        self._positions: Dict[str, ReadingPosition] = {}

        self._load_positions()

    def _load_positions(self) -> None:
        """Load positions from persistent storage."""
        if self.positions_file.exists():
            try:
                with open(self.positions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for file_id, pos_data in data.items():
                        self._positions[file_id] = ReadingPosition.from_dict(pos_data)
                logger.info("Loaded %d reading positions", len(self._positions))
            except Exception as e:
                logger.error("Failed to load positions: %s", e)

    def _save_positions(self) -> None:
        """Save positions to persistent storage."""
        try:
            data = {file_id: pos.to_dict() for file_id, pos in self._positions.items()}
            with open(self.positions_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save positions: %s", e)

    def save_position(self, position: ReadingPosition) -> None:
        """Save or update a reading position."""
        position.last_updated = datetime.now().isoformat()
        self._positions[position.file_id] = position
        self._save_positions()
        logger.debug(
            "Saved position for %s: word %d",
            position.file_name,
            position.current_word_index,
        )

    def get_position(self, file_id: str) -> Optional[ReadingPosition]:
        """Get reading position for a file."""
        return self._positions.get(file_id)

    def delete_position(self, file_id: str) -> None:
        """Delete reading position for a file."""
        if file_id in self._positions:
            del self._positions[file_id]
            self._save_positions()
            logger.debug("Deleted position for %s", file_id)

    def get_all_positions(self) -> List[ReadingPosition]:
        """Get all saved reading positions."""
        return list(self._positions.values())

    def get_unfinished_readings(self) -> List[ReadingPosition]:
        """Get all positions that have not been fully read."""
        return [
            pos
            for pos in self._positions.values()
            if pos.current_word_index < pos.total_words
        ]

    def has_position(self, file_id: str) -> bool:
        """Check if a position exists for a file."""
        return file_id in self._positions

    def update_word_index(self, file_id: str, word_index: int):
        """Update the current word index for a file."""
        if file_id in self._positions:
            self._positions[file_id].current_word_index = word_index
            self._positions[file_id].last_updated = datetime.now().isoformat()
            self._save_positions()

    def update_loaded_range(self, file_id: str, start_word: int, end_word: int):
        """Update the loaded range for a file."""
        if file_id in self._positions:
            self._positions[file_id].loaded_start_word = start_word
            self._positions[file_id].loaded_end_word = end_word
            self._save_positions()

    def mark_paused(self, file_id: str):
        """Mark a file as paused."""
        if file_id in self._positions:
            self._positions[file_id].is_paused = True
            self._positions[file_id].is_reading = False
            self._save_positions()

    def mark_resumed(self, file_id: str):
        """Mark a file as resumed/reading."""
        if file_id in self._positions:
            self._positions[file_id].is_paused = False
            self._positions[file_id].is_reading = True
            self._save_positions()

    def mark_completed(self, file_id: str):
        """Mark a file reading as completed."""
        if file_id in self._positions:
            self._positions[file_id].current_word_index = self._positions[
                file_id
            ].total_words
            self._positions[file_id].is_reading = False
            self._positions[file_id].is_paused = False
            self._save_positions()
