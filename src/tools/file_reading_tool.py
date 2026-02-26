"""
File Reading Tool - MCP-based chunk-by-chunk file reading with pause/resume support.

This tool gives the LLM explicit control over file reading, allowing
natural stopping by simply not calling the tool again.
Supports pause/resume and large file streaming.
"""

import asyncio
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import BaseTool
from core.reading_position_manager import ReadingPosition, ReadingPositionManager
from utils.file_stream_reader import (
    read_chunks_from_file,
    count_total_words,
    get_file_info,
    split_into_chunks,
)

# Global event loop reference for thread-safe async operations
_main_event_loop: Optional[asyncio.AbstractEventLoop] = None


def set_main_event_loop(loop: asyncio.AbstractEventLoop):
    """Set the main event loop for thread-safe async operations."""
    global _main_event_loop
    _main_event_loop = loop


def get_main_event_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Get the main event loop."""
    global _main_event_loop
    return _main_event_loop


class FileReadingTool(BaseTool):
    """
    Read a file chunk by chunk with automatic background reading.

    Each call starts reading the file in background chunks (~100 words each).
    The reading continues automatically until:
    - All content is read
    - User says "stop reading" (which calls stop_reading)
    - A new file is loaded (restarts reading)

    Supports:
    - Pause/resume reading
    - Large file streaming (loads 3 chunks at a time by default)
    - Position persistence across sessions
    """

    def __init__(
        self, config: dict, session_memory=None, voice_daemon=None, web_interface=None
    ):
        super().__init__(config)
        self.session_memory = session_memory
        self.voice_daemon = voice_daemon
        self.web_interface = web_interface

        self.position_manager = ReadingPositionManager(
            session_memory.memory_dir if session_memory else None
        )

        self.chunk_size: int = 300  # Characters for Chinese, words for English
        self.chunks_per_load: int = 3

        self.current_file_id: Optional[str] = None
        self.current_file_path: Optional[str] = None
        self.current_file_name: Optional[str] = None

        self.current_word_index: int = 0
        self.loaded_start_word: int = 0
        self.loaded_end_word: int = 0

        self.total_words: int = 0
        self.current_chunks: list = []
        self.current_chunk_index: int = 0

        self.is_reading: bool = False
        self.is_paused: bool = False

        # Buffer queue for pull-based model
        # Buffer loader thread loads chunks on-demand, voice daemon pulls when ready
        import queue

        self._chunk_buffer: queue.Queue = queue.Queue(maxsize=5)
        self._buffer_loader_thread: Optional[threading.Thread] = None

        # Track reading position for progressive reading
        self._read_position: int = 0  # Character position in content
        self._total_chars: int = 0

        # Current chunk being processed
        self._current_chunk_text: str = ""
        self._current_chunk_index: int = 0

        self._reading_thread: Optional[threading.Thread] = None

        self.web_fetch_tool = None
        self._current_url: Optional[str] = None

    def set_session_memory(self, session_memory):
        """Set the session memory reference."""
        self.session_memory = session_memory
        self.position_manager = ReadingPositionManager(
            session_memory.memory_dir if session_memory else None
        )

    def set_voice_daemon(self, voice_daemon):
        """Set the voice daemon reference."""
        self.voice_daemon = voice_daemon

        # Set up audio ready callback for web broadcast
        if voice_daemon:
            voice_daemon.on_audio_ready = self._on_audio_ready
            print(
                f"[FileReading] Set voice_daemon callback, voice_output={self.config.get('tts', {}).get('voice_output', 'local') if self.config else 'unknown'}"
            )
            print(f"[FileReading] Callback is: {voice_daemon.on_audio_ready}")

    def _on_audio_ready(self, audio_file: str, audio_type: str, request=None):
        """Callback when audio is ready - broadcast to web if voice_output is 'web'."""
        wi = self.web_interface

        # Get the chunk index from request metadata if available
        chunk_index = 0
        if request and hasattr(request, "metadata"):
            chunk_index = request.metadata.get("paragraph", 0)

        # Debug: print the state of web_interface
        print(f"[FileReading] DEBUG: web_interface exists: {wi is not None}")
        print(f"[FileReading] DEBUG: web_interface id: {id(wi) if wi else 'None'}")
        print(
            f"[FileReading] DEBUG: has manager: {hasattr(wi, 'manager') if wi else 'N/A'}"
        )

        manager = None
        if wi:
            try:
                manager = wi.manager
                print(
                    f"[FileReading] _on_audio_ready: got manager, id={id(manager)}, connections={len(manager.active_connections)}, chunk_index={chunk_index}"
                )
            except Exception as e:
                print(f"[FileReading] Could not get web_interface.manager: {e}")
        else:
            print(f"[FileReading] DEBUG: web_interface not set on FileReadingTool!")

        print(
            f"[FileReading] >>> _on_audio_ready ENTERED: audio_file={audio_file}, audio_type={audio_type}, chunk_index={chunk_index}, manager_id={id(manager) if manager else 'None'}, connections={len(manager.active_connections) if manager else 0}"
        )

        voice_output = "local"
        if hasattr(self, "config") and self.config:
            voice_output = self.config.get("tts", {}).get("voice_output", "local")

        print(
            f"[FileReading] _on_audio_ready: voice_output={voice_output}, audio_type={audio_type}"
        )

        if voice_output == "web":
            loop = get_main_event_loop()
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_audio(audio_file, audio_type, chunk_index), loop
                )
            else:
                print(f"[FileReading] Main loop not available, trying fallback...")
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(
                            self._broadcast_audio(audio_file, audio_type, chunk_index)
                        )
                    else:
                        print(f"[FileReading] Loop not running, skipping broadcast")
                except RuntimeError as e:
                    print(f"[FileReading] No event loop: {e}, skipping broadcast")

        # After audio finishes, play next chunk from buffer
        if self.is_reading and not self.is_paused:
            print(f"[FileReading] Audio finished, playing next chunk...")
            self._play_next_chunk()

    async def _broadcast_audio(
        self, audio_file: str, audio_type: str, chunk_index: int = 0
    ):
        """Broadcast audio to web clients."""
        # Use the stored web_interface reference
        wi = self.web_interface
        manager = wi.manager if wi and hasattr(wi, "manager") else None

        if not audio_file:
            return

        try:
            import os
            import base64

            if not manager:
                print(f"[FileReading] manager not ready, skipping broadcast")
                return

            print(
                f"[FileReading] _broadcast_audio: connections={len(manager.active_connections)}, manager_id={id(manager)}, chunk_index={chunk_index}"
            )

            if not os.path.exists(audio_file):
                print(f"[FileReading] Audio file not found: {audio_file}")
                return

            with open(audio_file, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("utf-8")

            message = {
                "type": "audio",
                "audio_data": audio_data,
                "audio_type": audio_type,
                "format": "mp3" if audio_file.endswith(".mp3") else "wav",
                "chunk_index": chunk_index,
            }

            connections = manager.active_connections
            print(f"[FileReading] About to broadcast to {len(connections)} connections")
            for connection in connections:
                await connection.send_json(message)
            print(f"[FileReading] Broadcast complete to {len(connections)} clients")
        except Exception as e:
            import traceback

            print(f"[FileReading] Error broadcasting audio: {e}")
            traceback.print_exc()

    def set_web_fetch_tool(self, web_fetch_tool):
        """Set the web fetch tool for URL reading."""
        self.web_fetch_tool = web_fetch_tool

    def _get_description(self) -> str:
        return (
            "Read a file or webpage aloud in background chunks. Supports pause, resume, and continuing from where you left off. "
            "Use url= to read webpages, file_id= or filename= for uploaded files. "
            "Use action='read' to start, action='pause' to pause, action='resume' to continue, "
            "action='stop' to stop completely, action='status' to get current position."
        )

    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Optional: specific file ID to read. If not provided, uses the most recent file.",
                },
                "filename": {
                    "type": "string",
                    "description": "Optional: partial filename to search for.",
                },
                "url": {
                    "type": "string",
                    "description": "Optional: URL of a webpage to read. Will fetch content and read it aloud with pause/resume support.",
                },
                "action": {
                    "type": "string",
                    "enum": ["read", "pause", "resume", "stop", "status"],
                    "description": "Action to perform: read (default), pause, resume, stop, or status",
                },
                "chunks_per_load": {
                    "type": "number",
                    "description": "Number of chunks to load at once (default: 3). For large files, lower values reduce memory.",
                },
                "chunk_size": {
                    "type": "number",
                    "description": "Number of words per chunk (default: 100)",
                },
            },
            "required": [],
        }

    def _find_attachment(
        self, file_id: str = None, filename: str = None
    ) -> Optional[Dict]:
        """Find an attachment by file_id, filename, or return most recent."""
        attachment = None

        if file_id:
            attachment = next(
                (a for a in self.session_memory.attachments if a["id"] == file_id), None
            )

        if not attachment and filename:
            attachment = self.session_memory.find_attachment_by_name(filename)

        if not attachment:
            attachment = self.session_memory.get_last_attachment()

        return attachment

    def _get_file_path(self, attachment: Dict) -> Optional[str]:
        """Get the file path for streaming reads."""
        if attachment.get("file_path") and attachment["file_path"]:
            return attachment["file_path"]

        if attachment.get("content_ref") and attachment["content_ref"]:
            return attachment["content_ref"]

        return None

    async def _load_url(self, url: str, start_word: int = 0) -> tuple[bool, str]:
        """Fetch URL content, save to temp file, and initialize reading state."""
        import asyncio
        import tempfile
        import os

        if not self.web_fetch_tool:
            return False, "Web fetch tool not available. Cannot read URLs."

        print(f"[FileReading] Fetching URL: {url}")

        try:
            result = await self.web_fetch_tool.execute(url=url, read_aloud=False)

            if not result.get("success"):
                error = result.get("error", "Failed to fetch URL")
                return False, f"Could not fetch URL: {error}"

            content = result.get("content", "")
            if not content:
                return False, "No content extracted from URL"

            # Debug: show content info
            print(
                f"[DEBUG] WebFetch returned content: {len(content)} chars, split() gives {len(content.split())} items"
            )
            print(f"[DEBUG] Content first 200 chars: {content[:200]}")

            # Generate a file_id for the URL
            import uuid

            url_file_id = f"url_{uuid.uuid4().hex[:8]}"

            # Extract title from content or use URL
            title = result.get("title", "")
            if not title:
                title = url.split("/")[-1] or url

            # Sanitize title for filename
            safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:50]

            # Save content to temp file (unify with file reading)
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=".txt", prefix=f"webfetch_{safe_title}_"
            )
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"[FileReading] Saved URL content to temp file: {temp_path}")

            # Initialize reading state - use file path (same as file reading)
            self.current_file_id = url_file_id
            self.current_file_path = temp_path
            self.current_file_name = title
            self._current_url = url

            # Clear in-memory content (we're now using file)
            self.current_content = None

            # For URLs, use character count since the file is Chinese text
            self._total_chars = len(content)
            self.total_words = (
                self._total_chars
            )  # Store as total_words for compatibility
            self.current_word_index = start_word
            self.current_chunk_index = 0
            self.is_reading = True
            self.is_paused = False

            position = ReadingPosition(
                file_id=self.current_file_id,
                file_path=temp_path,  # Store temp file path
                file_name=self.current_file_name,
                total_words=self.total_words,
                current_word_index=start_word,
                loaded_start_word=start_word,
                loaded_end_word=start_word,
                chunk_size=self.chunk_size,
                chunks_per_load=self.chunks_per_load,
                is_reading=True,
                is_paused=False,
            )
            self.position_manager.save_position(position)

            return True, f"Loaded webpage ({self.total_words} words)"

        except Exception as e:
            print(f"[FileReading] Error loading URL: {e}")
            return False, f"Error loading URL: {str(e)}"

    def _load_file(
        self, file_id: str = None, filename: str = None, start_word: int = 0
    ) -> tuple[bool, str]:
        """Load file content and initialize reading state."""
        attachment = self._find_attachment(file_id, filename)

        if not attachment:
            return False, "No file found. Please upload a file first."

        file_path = self._get_file_path(attachment)

        if not file_path:
            content = self.session_memory.get_attachment_content(attachment["id"])
            if not content:
                return False, f"Could not load content for {attachment['filename']}"
            self.total_words = len(content.split())
            self.current_content = content
        else:
            self.total_words = count_total_words(file_path)
            self.current_file_path = file_path

        self.current_file_id = attachment["id"]
        self.current_file_name = attachment["filename"]
        self.current_word_index = start_word
        self.current_chunk_index = 0
        self.is_reading = True
        self.is_paused = False

        position = ReadingPosition(
            file_id=self.current_file_id,
            file_path=self.current_file_path or "",
            file_name=self.current_file_name,
            total_words=self.total_words,
            current_word_index=start_word,
            loaded_start_word=start_word,
            loaded_end_word=start_word,
            chunk_size=self.chunk_size,
            chunks_per_load=self.chunks_per_load,
            is_reading=True,
            is_paused=False,
        )
        self.position_manager.save_position(position)

        return True, f"Loaded {attachment['filename']} ({self.total_words} words)"

    def _load_next_chunks(self) -> bool:
        """Load the next set of chunks from current position."""

        # Handle in-memory content (URLs)
        if self.current_content and not self.current_file_path:
            return self._load_chunks_from_content()

        # Handle file path streaming
        if not self.current_file_path:
            return False

        start_word = self.current_word_index
        chunks, actual_start, actual_end = read_chunks_from_file(
            self.current_file_path, start_word, self.chunk_size, self.chunks_per_load
        )

        if not chunks:
            return False

        self.current_chunks = chunks
        self.loaded_start_word = actual_start
        self.loaded_end_word = actual_end
        self.current_chunk_index = 0

        self.position_manager.update_loaded_range(
            self.current_file_id, actual_start, actual_end
        )

        print(
            f"[FileReading] Loaded chunks: words {actual_start}-{actual_end} ({len(chunks)} chunks)"
        )
        return True

    def _load_chunks_from_content(self) -> bool:
        """Load chunks from in-memory content (for URLs)."""
        if not self.current_content:
            return False

        # Use character-based indexing for Chinese text
        content = self.current_content
        total_chars = len(content)
        start_char = min(
            self.current_word_index, total_chars
        )  # word_index now means char_index for Chinese
        end_char = min(
            start_char + (self.chunk_size * self.chunks_per_load), total_chars
        )

        if start_char >= end_char:
            return False

        content_slice = content[start_char:end_char]
        chunks = split_into_chunks(content_slice, self.chunk_size, min_chunk_size=100)

        if not chunks:
            return False

        # Debug: print all chunks
        is_chinese = (
            len(self.current_content) > 0
            and self.current_content.count(" ") < len(self.current_content) * 0.1
        )
        print(
            f"[DEBUG] Total chars: {len(self.current_content)}, chunk_size={self.chunk_size}, chunks_per_load={self.chunks_per_load}, is_chinese={is_chinese}"
        )
        for i, chunk in enumerate(chunks):
            print(
                f"[DEBUG] Chunk {i + 1}: {len(chunk)} chars, first 50 chars: '{chunk[:50]}...'"
            )

        self.current_chunks = chunks
        self.loaded_start_word = start_char
        self.loaded_end_word = end_char
        self.current_chunk_index = 0

        self.position_manager.update_loaded_range(
            self.current_file_id, start_char, end_char
        )

        total_chars = sum(len(c) for c in chunks)
        print(
            f"[FileReading] Loaded URL chunks: chars {start_char}-{end_char} ({len(chunks)} chunks, ~{total_chars} total chars)"
        )
        return True

    def _buffer_loader_loop(self):
        """Buffer loader thread: maintains ~5 chunks in buffer, loads on-demand."""
        print(f"[BufferLoader] Loader thread started, is_reading={self.is_reading}")

        # Use file-based reading if we have a file path, otherwise use in-memory content
        if self.current_file_path:
            self._buffer_loader_from_file()
        elif self.current_content:
            self._buffer_loader_from_content()
        else:
            print("[BufferLoader] No content or file to read")

    def _buffer_loader_from_file(self):
        """Buffer loader for file-based reading."""
        print(f"[BufferLoader] Using file-based reading: {self.current_file_path}")

        # Calculate total characters from file
        self._total_chars = count_total_words(self.current_file_path)
        print(f"[BufferLoader] Total chars in file: {self._total_chars}")

        # Track chunk number for dynamic sizing
        chunk_number = 0
        last_chunk_index = -1

        while self.is_reading and self._read_position < self._total_chars:
            if self.is_paused:
                print("[BufferLoader] Paused, waiting...")
                time.sleep(0.1)
                continue

            # Check if buffer has room for more chunks (max 5)
            qsize = self._chunk_buffer.qsize()
            if qsize >= 5:
                # Buffer full, wait a bit
                time.sleep(0.2)
                continue

            # Determine target size based on chunk number (before reading)
            chunk_number += 1
            if chunk_number == 1:
                target_size = 20
            elif chunk_number == 2:
                target_size = 40
            else:
                target_size = 50

            # Read chunks from file starting at current position
            chunks = []
            try:
                print(
                    f"[BufferLoader] Calling read_chunks_from_file: path={self.current_file_path}, pos={self._read_position}, target_size={target_size}"
                )
                chunks, actual_start, actual_end = read_chunks_from_file(
                    self.current_file_path,
                    self._read_position,
                    target_size,
                    1,  # chunks_per_load
                )
                print(
                    f"[BufferLoader] Got {len(chunks)} chunks: {actual_start}-{actual_end}"
                )
            except Exception as e:
                print(f"[BufferLoader] Error reading file: {e}")
                import traceback

                traceback.print_exc()
                break

            if not chunks:
                print("[BufferLoader] No more chunks from file, finished")
                break

            # Get the chunk
            chunk_text = chunks[0]
            if not chunk_text.strip():
                print("[BufferLoader] Empty chunk, finished")
                break

            actual_size = len(chunk_text)

            print(
                f"[BufferLoader] Loaded chunk {chunk_number}: target={target_size}, actual={actual_size}, qsize={qsize + 1}"
            )

            # Put chunk in buffer
            self._chunk_buffer.put(chunk_text)

            # Move position forward (by character count approximation)
            self._read_position += actual_size
            self.current_word_index = self._read_position

            # Save position periodically
            if chunk_number % 5 == 0:
                self.position_manager.update_word_index(
                    self.current_file_id, self._read_position
                )

            progress = min(
                100, int((self._read_position / (self.total_words * 2)) * 100)
            )  # rough estimate
            print(
                f"[BufferLoader] Buffer position={self._read_position}, qsize={self._chunk_buffer.qsize()}"
            )

        # Signal done
        self._chunk_buffer.put(None)
        print(f"[BufferLoader] Finished loading from file")

    def _buffer_loader_from_content(self):
        """Buffer loader for in-memory content (legacy)."""
        print(f"[BufferLoader] Using in-memory content")

        content = self.current_content
        if not content:
            print("[BufferLoader] No content to read")
            return

        self._total_chars = len(content)

        # Determine if Chinese or English
        is_chinese = content.count(" ") < len(content) * 0.1

        # Track chunk number for dynamic sizing
        chunk_number = 0

        while self.is_reading and self._read_position < self._total_chars:
            if self.is_paused:
                print("[BufferLoader] Paused, waiting...")
                time.sleep(0.1)
                continue

            # Check if buffer has room for more chunks (max 5)
            qsize = self._chunk_buffer.qsize()
            if qsize >= 5:
                # Buffer full, wait a bit
                time.sleep(0.2)
                continue

            # Determine target size based on chunk number
            chunk_number += 1
            if chunk_number == 1:
                target_size = 20
            elif chunk_number == 2:
                target_size = 40
            else:
                target_size = 50

            # Get sentences until we exceed target size
            chunk_text = self._get_sentences_until_size(
                content, self._read_position, target_size, is_chinese
            )

            if not chunk_text.strip():
                print("[BufferLoader] Empty chunk, finished reading")
                break

            actual_size = len(chunk_text)
            print(
                f"[BufferLoader] Loaded chunk {chunk_number}: target={target_size}, actual={actual_size}, qsize={qsize + 1}"
            )

            # Put chunk in buffer
            self._chunk_buffer.put(chunk_text)

            # Move position forward
            self._read_position += len(chunk_text)
            self.current_word_index = self._read_position

            # Save position periodically (every 5 chunks)
            if chunk_number % 5 == 0:
                self.position_manager.update_word_index(
                    self.current_file_id, self._read_position
                )

            progress = (
                int((self._read_position / self._total_chars) * 100)
                if self._total_chars > 0
                else 0
            )
            print(
                f"[BufferLoader] Buffer position={self._read_position}/{self._total_chars}, progress={progress}%, qsize={self._chunk_buffer.qsize()}"
            )

        # Signal that we're done - put None marker
        self._chunk_buffer.put(None)
        print(
            f"[BufferLoader] Finished loading from content, position={self._read_position}/{self._total_chars}"
        )

    def _get_sentences_until_size(
        self, content: str, start_pos: int, target_size: int, is_chinese: bool
    ) -> str:
        """Get sentences until we exceed target size."""
        if is_chinese:
            return self._get_chinese_sentences(content, start_pos, target_size)
        else:
            return self._get_english_sentences(content, start_pos, target_size)

    def _get_chinese_sentences(
        self, content: str, start_pos: int, target_size: int
    ) -> str:
        """Get Chinese sentences: read one sentence at a time, accumulate until target size."""
        CHINESE_EOS = "。！？"
        CHINESE_CLAUSE = "，；："
        NEWLINE = "\n"

        text_len = len(content)
        pos = start_pos
        chunk = ""

        while pos < text_len:
            # Find next sentence boundary
            eos_pos = -1
            clause_pos = -1
            newline_pos = -1

            # Look for EOS first
            for i in range(pos, min(pos + 100, text_len)):
                if content[i] in CHINESE_EOS:
                    eos_pos = i + 1
                    break

            # Look for clause delimiter
            for i in range(pos, min(pos + 100, text_len)):
                if content[i] in CHINESE_CLAUSE:
                    clause_pos = i + 1
                    break

            # Look for newline
            for i in range(pos, min(pos + 100, text_len)):
                if content[i] in NEWLINE:
                    newline_pos = i + 1
                    break

            # Determine split position - prefer EOS, then clause, then newline
            split_pos = -1
            if eos_pos > pos:
                split_pos = eos_pos
            elif clause_pos > pos:
                split_pos = clause_pos
            elif newline_pos > pos:
                split_pos = newline_pos

            if split_pos == -1:
                # No boundary found, take remaining
                split_pos = text_len

            # Extract one sentence
            sentence = content[pos:split_pos]
            sentence_len = len(sentence.strip())

            if sentence_len > 0:
                # Add sentence to chunk
                if chunk:
                    chunk += sentence
                else:
                    chunk = sentence

                # Check if we've reached target size
                if len(chunk) >= target_size:
                    return chunk.strip()

            # Move to next position
            pos = split_pos

            # Safety: if chunk is way past target, return it
            if len(chunk) > target_size * 1.5:
                return chunk.strip()

        # Return whatever we have
        return chunk.strip()

    def _get_english_sentences(
        self, content: str, start_pos: int, target_size: int
    ) -> str:
        """Get English sentences: read one sentence at a time, accumulate until target size."""
        ENGLISH_EOS = ".!?"
        ENGLISH_CLAUSE = ",;:"
        NEWLINE = "\n"

        text_len = len(content)
        pos = start_pos
        chunk = ""

        while pos < text_len:
            # Find next sentence boundary
            eos_pos = -1
            clause_pos = -1
            newline_pos = -1

            # Look for EOS first
            for i in range(pos, min(pos + 100, text_len)):
                if content[i] in ENGLISH_EOS:
                    eos_pos = i + 1
                    break

            # Look for clause delimiter
            for i in range(pos, min(pos + 100, text_len)):
                if content[i] in ENGLISH_CLAUSE:
                    clause_pos = i + 1
                    break

            # Look for newline
            for i in range(pos, min(pos + 100, text_len)):
                if content[i] in NEWLINE:
                    newline_pos = i + 1
                    break

            # Determine split position - prefer EOS, then clause, then newline
            split_pos = -1
            if eos_pos > pos:
                split_pos = eos_pos
            elif clause_pos > pos:
                split_pos = clause_pos
            elif newline_pos > pos:
                split_pos = newline_pos

            if split_pos == -1:
                # No boundary found, take remaining
                split_pos = text_len

            # Extract one sentence
            sentence = content[pos:split_pos]
            sentence_len = len(sentence.strip())

            if sentence_len > 0:
                # Add sentence to chunk
                if chunk:
                    chunk += sentence
                else:
                    chunk = sentence

                # Check if we've reached target size (by characters)
                if len(chunk) >= target_size:
                    return chunk.strip()

            # Move to next position
            pos = split_pos

            # Safety: if chunk is way past target, return it
            if len(chunk) > target_size * 1.5:
                return chunk.strip()

        # Return whatever we have
        return chunk.strip()

    def _play_next_chunk(self):
        """Get next chunk from buffer and send to TTS. Called when audio finishes."""
        if not self.is_reading or self.is_paused:
            print("[FileReading] Not playing - paused or stopped")
            return

        # Get chunk from buffer
        try:
            chunk = self._chunk_buffer.get_nowait()
        except queue.Empty:
            print("[FileReading] Buffer empty, no more chunks to play")
            self.is_reading = False
            return

        # None = end of content
        if chunk is None:
            print("[FileReading] No more chunks - finished reading")
            self.is_reading = False
            return

        # Send chunk to TTS
        self._current_chunk_text = chunk

        progress = (
            int((self._read_position / self._total_chars) * 100)
            if self._total_chars > 0
            else 0
        )
        print(f"[FileReading] Playing chunk: len={len(chunk)}, progress={progress}%")

        result = self.voice_daemon.speak_file_content(
            text=chunk, paragraph_num=self._current_chunk_index, language="auto"
        )

        self._current_chunk_index += 1

        if not result.get("success"):
            print(f"[FileReading] TTS failed: {result.get('error')}")

    def _read_background(self):
        """Background thread to read chunks with prefetching."""
        print(f"[FileReading] _read_background() started, is_reading={self.is_reading}")

        if not self.current_chunks:
            if not self._load_next_chunks():
                self.is_reading = False
                print(f"[FileReading] Failed to load initial chunks")
                return

        total_chunks = len(self.current_chunks)
        min_queue_size = 2  # Maintain at least 2 chunks in queue

        while self.is_reading and self.current_word_index < self.total_words:
            if not self.is_reading:
                print(f"[FileReading] Stop flag detected, stopping background reading")
                break

            if self.is_paused:
                print(f"[FileReading] Paused, stopping background reading")
                break

            # Check if we need to load more chunks
            if self.current_chunk_index >= len(self.current_chunks):
                if not self._load_next_chunks():
                    print(
                        f"[FileReading] No more chunks to load, finishing remaining queue"
                    )
                total_chunks = len(self.current_chunks)

            # Get current state from voice daemon
            # We need to check if TTS is currently processing (not just if queue is empty)
            # Because queue can be empty but TTS might still be playing previous chunk
            current_queue_size = 0
            is_currently_speaking = False
            if self.voice_daemon:
                current_queue_size = (
                    self.voice_daemon.speech_queue.qsize()
                    if hasattr(self.voice_daemon, "speech_queue")
                    else 0
                )
                is_currently_speaking = getattr(self.voice_daemon, "is_speaking", False)

            # Only add next chunk if:
            # 1. Queue is empty (no pending chunks)
            # 2. Not currently speaking (previous TTS finished)
            # This ensures correct ordering - wait for each chunk to COMPLETE before adding next
            if (
                current_queue_size == 0
                and not is_currently_speaking
                and self.current_chunk_index < len(self.current_chunks)
            ):
                chunk = self.current_chunks[self.current_chunk_index]
                chunk_index = self.current_chunk_index
                self.current_chunk_index += 1

                chunk_words = len(chunk.split())
                self.current_word_index += chunk_words

                self.position_manager.update_word_index(
                    self.current_file_id, self.current_word_index
                )

                if self.voice_daemon and self.is_reading and not self.is_paused:
                    progress = (
                        int((self.current_word_index / self.total_words) * 100)
                        if self.total_words > 0
                        else 0
                    )
                    print(
                        f"[FileReading] Queued chunk {self.current_chunk_index}/{total_chunks} ({chunk_words} words) - {progress}%, queue={current_queue_size}, speaking={is_currently_speaking}"
                    )
                    result = self.voice_daemon.speak_file_content(
                        text=chunk, paragraph_num=chunk_index, language="auto"
                    )

                    if not result.get("success"):
                        print(
                            f"[FileReading] TTS failed for chunk {self.current_chunk_index}: {result.get('error')}"
                        )
            else:
                # Small sleep to avoid busy loop, but much shorter than before
                time.sleep(0.05)

        if self.current_word_index >= self.total_words:
            self.position_manager.mark_completed(self.current_file_id)

        self.is_reading = False
        print(
            f"[FileReading] Background reading finished. Read {self.current_word_index}/{self.total_words} words"
        )

    async def execute(
        self,
        file_id: str = None,
        filename: str = None,
        url: str = None,
        action: str = "read",
        chunks_per_load: int = 3,
        chunk_size: int = 300,
    ) -> Dict[str, Any]:
        """Execute reading action."""
        print(
            f"[FileReading] execute() called with file_id={file_id}, url={url}, action={action}, is_reading={self.is_reading}"
        )

        self.chunks_per_load = chunks_per_load if chunks_per_load else 3
        self.chunk_size = chunk_size if chunk_size else 300

        if action == "status":
            return self.get_status()

        if action == "pause":
            return self.pause_reading()

        if action == "resume":
            return self.resume_reading()

        if action == "stop":
            return self.stop_reading()

        if action == "read":
            # Handle URL reading
            if url:
                return await self._start_reading_url(url)

            # Handle file reading
            existing_position = None
            if file_id:
                existing_position = self.position_manager.get_position(file_id)
            elif self.current_file_id:
                existing_position = self.position_manager.get_position(
                    self.current_file_id
                )

            if existing_position and existing_position.current_word_index > 0:
                resume_msg = (
                    f"Resuming from word {existing_position.current_word_index}"
                )
                print(f"[FileReading] {resume_msg}")

            return self._start_reading(file_id, filename, existing_position)

        return {"success": False, "error": f"Unknown action: {action}"}

    async def _start_reading_url(self, url: str) -> Dict[str, Any]:
        """Start reading a URL."""
        import asyncio

        # Check for existing position for this URL
        url_file_id = f"url_{hash(url) % 10000000}"
        existing_position = self.position_manager.get_position(url_file_id)

        start_word = 0
        if existing_position and existing_position.current_word_index > 0:
            start_word = existing_position.current_word_index
            print(f"[FileReading] Resuming URL from word {start_word}")

        # Load the URL
        success, message = await self._load_url(url, start_word)

        if not success:
            return {"success": False, "error": message, "status": "load_failed"}

        # Update file_id to match position manager
        self.current_file_id = url_file_id

        # Clear any stale stop signals before starting new file reading
        if self.voice_daemon and hasattr(self.voice_daemon, "stop_event"):
            self.voice_daemon.stop_event.clear()

        # Start reading in background
        if self._reading_thread and self._reading_thread.is_alive():
            return self._get_reading_status()

        self.is_reading = True
        self.is_paused = False

        # Initialize buffer loader model
        self._chunk_buffer = queue.Queue(maxsize=5)
        self._current_chunk_index = 0
        self._read_position = self.current_word_index

        # Clear buffer
        while not self._chunk_buffer.empty():
            try:
                self._chunk_buffer.get_nowait()
            except queue.Empty:
                break

        # Start buffer loader thread
        self._buffer_loader_thread = threading.Thread(
            target=self._buffer_loader_loop, daemon=True
        )
        self._buffer_loader_thread.start()

        # Play first chunk after loader fills buffer a bit
        time.sleep(0.5)  # Give loader time to fill some chunks
        if self.is_reading and not self.is_paused:
            self._play_next_chunk()

        print(f"[FileReading] Started buffer loader, position={self._read_position}")

        return self._get_reading_status()

    def _start_reading(
        self,
        file_id: str = None,
        filename: str = None,
        existing_position: ReadingPosition = None,
    ) -> Dict[str, Any]:
        """Start or continue reading."""
        # Clear any stale stop signals before starting new file reading
        if self.voice_daemon and hasattr(self.voice_daemon, "stop_event"):
            self.voice_daemon.stop_event.clear()

        start_word = 0

        if existing_position and existing_position.current_word_index > 0:
            start_word = existing_position.current_word_index
            self.current_file_id = existing_position.file_id
            self.current_file_path = existing_position.file_path
            self.current_file_name = existing_position.file_name
            self.total_words = existing_position.total_words
            self.current_word_index = start_word
            self.is_reading = True
            self.is_paused = False

            if self._reading_thread and self._reading_thread.is_alive():
                print(f"[FileReading] Reading thread already running")
                return self._get_reading_status()
        else:
            success, message = self._load_file(file_id, filename, start_word)
            if not success:
                return {"success": False, "error": message, "status": "no_file"}

        if self._reading_thread and self._reading_thread.is_alive():
            print(f"[FileReading] Thread already running")
            return self._get_reading_status()

        print(f"[FileReading] Starting background thread from word {start_word}")
        self.is_reading = True
        self.is_paused = False

        self._reading_thread = threading.Thread(
            target=self._read_background, daemon=True
        )
        self._reading_thread.start()

        return self._get_reading_status()

    def _get_reading_status(self) -> Dict[str, Any]:
        """Get current reading status."""
        progress_percent = (
            int((self.current_word_index / self.total_words) * 100)
            if self.total_words > 0
            else 0
        )

        remaining_words = self.total_words - self.current_word_index
        remaining_chunks = remaining_words // self.chunk_size + (
            1 if remaining_words % self.chunk_size else 0
        )

        return {
            "success": True,
            "status": "reading"
            if self.is_reading
            else ("paused" if self.is_paused else "stopped"),
            "message": f"Reading {self.current_file_name}"
            if self.current_file_name
            else "Reading",
            "total_words": self.total_words,
            "current_word_index": self.current_word_index,
            "progress_percent": progress_percent,
            "remaining_chunks": remaining_chunks,
            "more_content": self.current_word_index < self.total_words,
            "is_paused": self.is_paused,
        }

    def pause_reading(self) -> Dict[str, Any]:
        """Pause the current reading session."""
        if not self.is_reading:
            return {
                "success": False,
                "error": "Not currently reading",
                "status": "not_reading",
            }

        print(f"[FileReading] pause_reading() called")
        self.is_paused = True

        if self.voice_daemon:
            self.voice_daemon.stop_file_reading()

        if self.current_file_id:
            self.position_manager.mark_paused(self.current_file_id)
            self.position_manager.update_word_index(
                self.current_file_id, self.current_word_index
            )

        progress_percent = (
            int((self.current_word_index / self.total_words) * 100)
            if self.total_words > 0
            else 0
        )

        return {
            "success": True,
            "status": "paused",
            "message": f"Paused at word {self.current_word_index} of {self.total_words} ({progress_percent}%)",
            "current_word_index": self.current_word_index,
            "total_words": self.total_words,
            "progress_percent": progress_percent,
        }

    def resume_reading(self) -> Dict[str, Any]:
        """Resume the paused reading session."""
        if not self.is_paused:
            if self.is_reading:
                return {
                    "success": False,
                    "error": "Already reading",
                    "status": "reading",
                }
            return {
                "success": False,
                "error": "No paused reading to resume",
                "status": "not_paused",
            }

        if not self.current_file_id:
            return {"success": False, "error": "No file to resume", "status": "no_file"}

        print(f"[FileReading] Resuming from word {self.current_word_index}")

        self.is_reading = True
        self.is_paused = False

        self.position_manager.mark_resumed(self.current_file_id)

        # For buffer-based reading, just trigger the next chunk
        # The buffer loader thread will continue automatically
        self._play_next_chunk()

        return self._get_reading_status()

    def stop_reading(self) -> Dict[str, Any]:
        """Stop the current reading session completely."""
        was_reading = self.is_reading or self.is_paused
        words_read = self.current_word_index
        total_words = self.total_words

        self.is_reading = False
        self.is_paused = False

        # Join buffer loader thread
        if self._buffer_loader_thread and self._buffer_loader_thread.is_alive():
            self._buffer_loader_thread.join(timeout=2.0)

        # Reset thread variables
        self._buffer_loader_thread = None

        # Clear buffer
        while not self._chunk_buffer.empty():
            try:
                self._chunk_buffer.get_nowait()
            except queue.Empty:
                break

        if self.voice_daemon:
            self.voice_daemon.stop_file_reading()

        if self.current_file_id:
            self.position_manager.delete_position(self.current_file_id)

        self.current_file_id = None
        self.current_file_path = None
        self.current_file_name = None
        self.current_word_index = 0
        self.loaded_start_word = 0
        self.loaded_end_word = 0
        self.total_words = 0
        self.current_chunks = []
        self.current_chunk_index = 0

        return {
            "success": True,
            "was_reading": was_reading,
            "words_read": words_read,
            "total_words": total_words,
            "message": f"Stopped reading. Read {words_read}/{total_words} words."
            if was_reading
            else "Was not reading anything.",
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current reading status."""
        if not self.current_file_id:
            unfinished = self.position_manager.get_unfinished_readings()
            if unfinished:
                return {
                    "is_reading": False,
                    "is_paused": False,
                    "has_unfinished": True,
                    "unfinished_readings": [
                        {
                            "file_id": pos.file_id,
                            "file_name": pos.file_name,
                            "current_word_index": pos.current_word_index,
                            "total_words": pos.total_words,
                            "progress_percent": pos.get_progress_percent(),
                        }
                        for pos in unfinished
                    ],
                    "message": f"You have {len(unfinished)} unfinished reading(s)",
                }
            return {
                "is_reading": False,
                "is_paused": False,
                "message": "No file currently being read",
            }

        if self.is_paused:
            progress_percent = (
                int((self.current_word_index / self.total_words) * 100)
                if self.total_words > 0
                else 0
            )
            return {
                "is_reading": False,
                "is_paused": True,
                "file_id": self.current_file_id,
                "file_name": self.current_file_name,
                "current_word_index": self.current_word_index,
                "total_words": self.total_words,
                "progress_percent": progress_percent,
                "message": f"Paused at word {self.current_word_index} of {self.total_words} ({progress_percent}%)",
            }

        if self.is_reading:
            return self._get_reading_status()

        progress_percent = (
            int((self.current_word_index / self.total_words) * 100)
            if self.total_words > 0
            else 0
        )

        return {
            "is_reading": False,
            "is_paused": False,
            "file_id": self.current_file_id,
            "file_name": self.current_file_name,
            "current_word_index": self.current_word_index,
            "total_words": self.total_words,
            "progress_percent": progress_percent,
            "message": f"At word {self.current_word_index} of {self.total_words} ({progress_percent}%)",
        }

    def get_progress_text(self) -> str:
        """Get human-readable progress text."""
        if not self.current_file_id:
            return "Not reading any file"

        progress_percent = (
            int((self.current_word_index / self.total_words) * 100)
            if self.total_words > 0
            else 0
        )

        if self.is_paused:
            return f"Paused at word {self.current_word_index} of {self.total_words} ({progress_percent}%) - {self.current_file_name}"

        if self.is_reading:
            return f"Reading at word {self.current_word_index} of {self.total_words} ({progress_percent}%) - {self.current_file_name}"

        return f"At word {self.current_word_index} of {self.total_words} ({progress_percent}%) - {self.current_file_name}"
