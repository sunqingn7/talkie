"""
Memory Manager - Persistent conversation storage with auto-summarization.

Structure:
Memory/
  2024-03-02/
    summary-2024-03-02.md       # Daily summary
    session-2024-03-02-09-15.md # Individual sessions
    session-2024-03-02-14-30.md
  2024-03-03/
    summary-2024-03-03.md
    session-2024-03-03-10-00.md

Features:
- Auto-save conversations to date-based markdown files
- Hourly summarization of all chats into daily summary
- Search and retrieval of past conversations
"""

import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from queue import Queue
import time

from src.utils.logger import get_logger
from src.core.llm_client import LLMClient

logger = get_logger(__name__)


@dataclass
class Message:
    """Represents a single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_markdown(self) -> str:
        """Convert message to markdown format."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        role_icon = "👤" if self.role == "user" else "🤖"
        return f"{role_icon} **{self.role.title()}** ({time_str})\n\n{self.content}\n"


@dataclass
class Session:
    """Represents a conversation session."""

    session_id: str
    start_time: datetime
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the session."""
        message = Message(role=role, content=content)
        self.messages.append(message)
        logger.debug("Added %s message to session %s", role, self.session_id)

    def to_markdown(self) -> str:
        """Convert session to markdown format."""
        lines = [
            f"# Session: {self.session_id}",
            f"**Started:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Duration:** {len(self.messages)} messages",
            "",
            "## Conversation",
            "",
        ]

        for msg in self.messages:
            lines.append(msg.to_markdown())
            lines.append("---")
            lines.append("")

        if self.metadata:
            lines.append("## Metadata")
            lines.append("")
            for key, value in self.metadata.items():
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        return "\n".join(lines)

    def get_summary_text(self) -> str:
        """Get text suitable for summarization."""
        lines = []
        for msg in self.messages:
            lines.append(f"{msg.role}: {msg.content}")
        return "\n".join(lines)


class MemoryManager:
    """
    Manages persistent conversation memory with auto-summarization.

    Directory structure:
    Memory/
      YYYY-MM-DD/
        summary-YYYY-MM-DD.md
        session-YYYY-MM-DD-HH-MM.md
    """

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        summary_interval_hours: float = 1.0,
        auto_summarize: bool = True,
        llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialize memory manager.

        Args:
            memory_dir: Base directory for memory storage
            summary_interval_hours: Hours between auto-summarization
            auto_summarize: Enable automatic summarization
            llm_client: Optional LLM client for intelligent summarization
        """
        self.memory_base = Path(memory_dir or "Memory")
        self.memory_base.mkdir(parents=True, exist_ok=True)

        self.summary_interval = summary_interval_hours * 3600
        self.auto_summarize = auto_summarize
        self._llm_client = llm_client

        # Current sessions (session_id -> Session)
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

        # Summarization
        self._summarization_queue: Queue = Queue()
        self._summarization_thread: Optional[threading.Thread] = None
        self._last_summary_time: Optional[datetime] = None
        self._running = False

        # Callbacks
        self.on_summary_ready: Optional[callable] = None

        logger.info("MemoryManager initialized at %s", self.memory_base)

    def start(self) -> None:
        """Start the memory manager (auto-summarization thread)."""
        if self._running:
            logger.warning("MemoryManager already running")
            return

        self._running = True
        self._summarization_thread = threading.Thread(
            target=self._summarization_loop, name="MemorySummarizer", daemon=True
        )
        self._summarization_thread.start()
        logger.info(
            "MemoryManager started (summary interval: %.1f hours)",
            self.summary_interval / 3600,
        )

    def stop(self) -> None:
        """Stop the memory manager."""
        if not self._running:
            return

        logger.info("Stopping MemoryManager...")
        self._running = False

        # Do a final summary before stopping
        self._summarize_today()

        if self._summarization_thread and self._summarization_thread.is_alive():
            self._summarization_thread.join(timeout=5.0)

        logger.info("MemoryManager stopped")

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """
        Get existing session or create a new one.

        Args:
            session_id: Optional session ID (auto-generated if not provided)

        Returns:
            Session ID
        """
        now = datetime.now()

        if session_id is None:
            # Generate session ID based on time
            session_id = now.strftime("%Y%m%d-%H%M%S")

        with self._lock:
            if session_id not in self._sessions:
                session = Session(session_id=session_id, start_time=now)
                self._sessions[session_id] = session
                logger.info("Created new session: %s", session_id)

            return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to a session and auto-save.

        Args:
            session_id: Session identifier
            role: "user" or "assistant"
            content: Message content
        """
        with self._lock:
            if session_id not in self._sessions:
                # Create session if it doesn't exist
                session = Session(session_id=session_id, start_time=datetime.now())
                self._sessions[session_id] = session

            self._sessions[session_id].add_message(role, content)

        # Auto-save session to file
        self._save_session(session_id)

    def _save_session(self, session_id: str) -> None:
        """Save a session to markdown file."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return

        # Determine date folder
        date_str = session.start_time.strftime("%Y-%m-%d")
        date_folder = self.memory_base / date_str
        date_folder.mkdir(parents=True, exist_ok=True)

        # Create filename
        time_str = session.start_time.strftime("%Y-%m-%d-%H-%M")
        filename = f"session-{time_str}.md"
        filepath = date_folder / filename

        # Write to file
        try:
            content = session.to_markdown()
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.debug("Saved session %s to %s", session_id, filepath)
        except Exception as e:
            logger.error("Failed to save session %s: %s", session_id, e)

    def _summarization_loop(self) -> None:
        """Background loop for periodic summarization."""
        logger.info("Summarization loop started")

        while self._running:
            try:
                # Check if it's time to summarize
                now = datetime.now()

                if self._last_summary_time is None:
                    # First summary
                    self._summarize_today()
                    self._last_summary_time = now
                elif (
                    now - self._last_summary_time
                ).total_seconds() >= self.summary_interval:
                    # Time for next summary
                    self._summarize_today()
                    self._last_summary_time = now

                # Sleep briefly before next check
                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error("Error in summarization loop: %s", e)
                time.sleep(60)

        logger.info("Summarization loop stopped")

    def _summarize_today(self) -> None:
        """Create summary for today's conversations."""
        today = datetime.now().strftime("%Y-%m-%d")
        date_folder = self.memory_base / today

        if not date_folder.exists():
            logger.debug("No conversations for %s", today)
            return

        # Collect all sessions for today
        sessions = []
        session_files = sorted(date_folder.glob("session-*.md"))

        for filepath in session_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                sessions.append((filepath.stem, content))
            except Exception as e:
                logger.error("Failed to read session file %s: %s", filepath, e)

        if not sessions:
            logger.debug("No sessions to summarize for %s", today)
            return

        # Create summary content
        summary_content = self._create_summary(today, sessions)

        # Save summary
        summary_file = date_folder / f"summary-{today}.md"
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary_content)
            logger.info("Created summary for %s (%d sessions)", today, len(sessions))

            # Trigger callback if set
            if self.on_summary_ready:
                try:
                    self.on_summary_ready(today, summary_file)
                except Exception as e:
                    logger.error("Error in summary ready callback: %s", e)

        except Exception as e:
            logger.error("Failed to save summary for %s: %s", today, e)

    def _create_summary(self, date_str: str, sessions: List[tuple]) -> str:
        """
        Create summary content from sessions.

        Uses LLM for intelligent summarization if available.
        """
        lines = [
            f"# Daily Summary - {date_str}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Sessions:** {len(sessions)}",
            "",
        ]

        all_conversations = []
        for session_id, content in sessions:
            session_time = re.search(r"\d{2}:\d{2}:\d{2}", content)
            time_str = session_time.group(0) if session_time else "Unknown"
            msg_count = content.count("👤")
            all_conversations.append(
                {
                    "id": session_id,
                    "time": time_str,
                    "messages": msg_count,
                    "content": content,
                }
            )

        if self._llm_client and sessions:
            try:
                llm_summary = self._create_llm_summary(date_str, all_conversations)
                if llm_summary:
                    lines.append(llm_summary)
                    return "\n".join(lines)
            except Exception as e:
                logger.warning(f"LLM summarization failed, falling back to basic: {e}")

        lines.append("## Sessions")
        lines.append("")

        for session_id, content in sessions:
            session_time = re.search(r"\d{2}:\d{2}:\d{2}", content)
            time_str = session_time.group(0) if session_time else "Unknown"

            msg_count = content.count("👤")

            preview_lines = content.split("\n")[:5]
            preview = " ".join(preview_lines).replace("\n", " ")[:200] + "..."

            lines.append(f"### {session_id} ({time_str})")
            lines.append(f"- **Messages:** {msg_count}")
            lines.append(f"- **Preview:** {preview}")
            lines.append("")

        lines.append("## Key Topics")
        lines.append("")
        lines.append(
            "*This section will be enhanced with LLM-powered topic extraction.*"
        )
        lines.append("")

        return "\n".join(lines)

    def _create_llm_summary(self, date_str: str, sessions: List[Dict]) -> Optional[str]:
        """Generate intelligent summary using LLM."""
        if not self._llm_client:
            return None

        prompt_lines = [
            f"You are a helpful assistant summarizing a day's worth of conversations.",
            f"Summarize the following {len(sessions)} chat sessions from {date_str} into a coherent daily summary.",
            "",
            "For each session, provide:",
            "1. A brief description of what was discussed",
            "2. Key topics or decisions made",
            "",
            "Finally, extract the main themes/topics of the day.",
            "",
            "Sessions:",
        ]

        for i, session in enumerate(sessions, 1):
            preview = session.get("content", "")[:800]
            prompt_lines.append(
                f"\n--- Session {i}: {session.get('id')} at {session.get('time')} ---"
            )
            prompt_lines.append(preview)

        prompt_lines.extend(
            [
                "",
                "Now provide your summary in this format:",
                "## Overview",
                "[Brief summary of the day's conversations]",
                "",
                "## Key Discussions",
                "- [Topic 1]: [Brief description]",
                "- [Topic 2]: [Brief description]",
                "",
                "## Main Themes",
                "- Theme 1",
                "- Theme 2",
            ]
        )

        prompt = "\n".join(prompt_lines)

        try:
            response = self._llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}], stream=False
            )

            if response.get("choices"):
                content = response["choices"][0]["message"]["content"]
                return content
        except Exception as e:
            logger.error(f"LLM summary generation error: {e}")

        return None

    def get_sessions_for_date(
        self, date_str: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all sessions for a specific date.

        Args:
            date_str: Date in YYYY-MM-DD format (today if not specified)

        Returns:
            List of session info dictionaries
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        date_folder = self.memory_base / date_str
        if not date_folder.exists():
            return []

        sessions = []
        session_files = sorted(date_folder.glob("session-*.md"))

        for filepath in session_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract metadata
                session_id = filepath.stem.replace("session-", "")

                sessions.append(
                    {
                        "file": str(filepath),
                        "session_id": session_id,
                        "content": content,
                        "lines": len(content.split("\n")),
                    }
                )
            except Exception as e:
                logger.error("Failed to read session %s: %s", filepath, e)

        return sessions

    def get_summary(self, date_str: Optional[str] = None) -> Optional[str]:
        """
        Get summary for a specific date.

        Args:
            date_str: Date in YYYY-MM-DD format (today if not specified)

        Returns:
            Summary content or None if not found
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        date_folder = self.memory_base / date_str
        summary_file = date_folder / f"summary-{date_str}.md"

        if not summary_file.exists():
            return None

        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error("Failed to read summary %s: %s", summary_file, e)
            return None

    def search_conversations(
        self, query: str, date_from: Optional[str] = None, date_to: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search through conversations.

        Args:
            query: Search query
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)

        Returns:
            List of matching results with context
        """
        results = []

        # Determine date range
        if date_to is None:
            date_to = datetime.now().strftime("%Y-%m-%d")
        if date_from is None:
            date_from = date_to  # Search only one day if not specified

        current_date = datetime.strptime(date_from, "%Y-%m-%d")
        end_date = datetime.strptime(date_to, "%Y-%m-%d")

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            date_folder = self.memory_base / date_str

            if date_folder.exists():
                session_files = date_folder.glob("session-*.md")

                for filepath in session_files:
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Simple text search (case-insensitive)
                        if query.lower() in content.lower():
                            # Find matching lines
                            lines = content.split("\n")
                            matching_lines = []

                            for i, line in enumerate(lines):
                                if query.lower() in line.lower():
                                    # Get context (2 lines before and after)
                                    start = max(0, i - 2)
                                    end = min(len(lines), i + 3)
                                    context = lines[start:end]

                                    matching_lines.append(
                                        {"line_num": i + 1, "context": context}
                                    )

                            if matching_lines:
                                results.append(
                                    {
                                        "file": str(filepath),
                                        "date": date_str,
                                        "session_id": filepath.stem.replace(
                                            "session-", ""
                                        ),
                                        "matches": len(matching_lines),
                                        "matching_lines": matching_lines,
                                    }
                                )
                    except Exception as e:
                        logger.error("Failed to search file %s: %s", filepath, e)

            # Move to next day
            current_date += timedelta(days=1)

        return results

    def get_all_dates(self) -> List[str]:
        """Get all dates with conversation data."""
        dates = []

        if self.memory_base.exists():
            for item in sorted(self.memory_base.iterdir(), reverse=True):
                if item.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", item.name):
                    dates.append(item.name)

        return dates

    def export_session(self, session_id: str, filepath: str) -> bool:
        """
        Export a session to a specific file.

        Args:
            session_id: Session identifier
            filepath: Output file path

        Returns:
            True if successful
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                # Try to load from file
                return self._load_and_export_session(session_id, filepath)

        try:
            content = session.to_markdown()
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("Exported session %s to %s", session_id, filepath)
            return True
        except Exception as e:
            logger.error("Failed to export session %s: %s", session_id, e)
            return False

    def _load_and_export_session(self, session_id: str, filepath: str) -> bool:
        """Load session from file and export it."""
        # Search for session file
        pattern = f"session-{session_id}*.md"

        for date_folder in self.memory_base.iterdir():
            if not date_folder.is_dir():
                continue

            session_files = list(date_folder.glob(pattern))
            if session_files:
                source_file = session_files[0]
                try:
                    with open(source_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)

                    logger.info(
                        "Exported session %s from %s to %s",
                        session_id,
                        source_file,
                        filepath,
                    )
                    return True
                except Exception as e:
                    logger.error("Failed to export session %s: %s", session_id, e)
                    return False

        logger.error("Session %s not found", session_id)
        return False


# Global instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(
    memory_dir: Optional[str] = None,
    auto_start: bool = True,
    llm_client: Optional[LLMClient] = None,
) -> MemoryManager:
    """Get or create the global memory manager instance."""
    global _memory_manager

    if _memory_manager is None:
        _memory_manager = MemoryManager(memory_dir=memory_dir, llm_client=llm_client)
        if auto_start:
            _memory_manager.start()

    return _memory_manager


def reset_memory_manager() -> None:
    """Reset the global memory manager instance."""
    global _memory_manager

    if _memory_manager:
        _memory_manager.stop()
    _memory_manager = None
