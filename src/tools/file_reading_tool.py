"""
File Reading Tool - MCP-based chunk-by-chunk file reading with pause/resume support.

This tool gives the LLM explicit control over file reading, allowing
natural stopping by simply not calling the tool again.
Supports pause/resume and large file streaming.
"""

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
    split_into_chunks
)


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
    
    def __init__(self, config: dict, session_memory=None, voice_daemon=None):
        super().__init__(config)
        self.session_memory = session_memory
        self.voice_daemon = voice_daemon
        
        self.position_manager = ReadingPositionManager(
            session_memory.memory_dir if session_memory else None
        )
        
        self.chunk_size: int = 100
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
                    "description": "Optional: specific file ID to read. If not provided, uses the most recent file."
                },
                "filename": {
                    "type": "string",
                    "description": "Optional: partial filename to search for."
                },
                "url": {
                    "type": "string",
                    "description": "Optional: URL of a webpage to read. Will fetch content and read it aloud with pause/resume support."
                },
                "action": {
                    "type": "string",
                    "enum": ["read", "pause", "resume", "stop", "status"],
                    "description": "Action to perform: read (default), pause, resume, stop, or status"
                },
                "chunks_per_load": {
                    "type": "number",
                    "description": "Number of chunks to load at once (default: 3). For large files, lower values reduce memory."
                },
                "chunk_size": {
                    "type": "number",
                    "description": "Number of words per chunk (default: 100)"
                }
            },
            "required": []
        }
    
    def _find_attachment(self, file_id: str = None, filename: str = None) -> Optional[Dict]:
        """Find an attachment by file_id, filename, or return most recent."""
        attachment = None
        
        if file_id:
            attachment = next(
                (a for a in self.session_memory.attachments if a['id'] == file_id),
                None
            )
        
        if not attachment and filename:
            attachment = self.session_memory.find_attachment_by_name(filename)
        
        if not attachment:
            attachment = self.session_memory.get_last_attachment()
        
        return attachment
    
    def _get_file_path(self, attachment: Dict) -> Optional[str]:
        """Get the file path for streaming reads."""
        if attachment.get('file_path') and attachment['file_path']:
            return attachment['file_path']
        
        if attachment.get('content_ref') and attachment['content_ref']:
            return attachment['content_ref']
        
        return None
    
    async def _load_url(self, url: str, start_word: int = 0) -> tuple[bool, str]:
        """Fetch URL content and initialize reading state."""
        import asyncio
        
        if not self.web_fetch_tool:
            return False, "Web fetch tool not available. Cannot read URLs."
        
        print(f"[FileReading] Fetching URL: {url}")
        
        try:
            result = await self.web_fetch_tool.execute(url=url, read_aloud=False)
            
            if not result.get('success'):
                error = result.get('error', 'Failed to fetch URL')
                return False, f"Could not fetch URL: {error}"
            
            content = result.get('content', '')
            if not content:
                return False, "No content extracted from URL"
            
            # Generate a file_id for the URL
            import uuid
            self.current_file_id = f"url_{uuid.uuid4().hex[:8]}"
            self._current_url = url
            
            # Store content in memory (not to disk for URLs)
            self.current_content = content
            self.current_file_path = None  # URLs don't have file paths
            self.current_file_name = url.split('/')[-1] or url
            
            self.total_words = len(content.split())
            self.current_word_index = start_word
            self.current_chunk_index = 0
            self.is_reading = True
            self.is_paused = False
            
            position = ReadingPosition(
                file_id=self.current_file_id,
                file_path=url,  # Store URL as "file_path" for URLs
                file_name=self.current_file_name,
                total_words=self.total_words,
                current_word_index=start_word,
                loaded_start_word=start_word,
                loaded_end_word=start_word,
                chunk_size=self.chunk_size,
                chunks_per_load=self.chunks_per_load,
                is_reading=True,
                is_paused=False
            )
            self.position_manager.save_position(position)
            
            return True, f"Loaded webpage ({self.total_words} words)"
            
        except Exception as e:
            print(f"[FileReading] Error loading URL: {e}")
            return False, f"Error loading URL: {str(e)}"
    
    def _load_file(self, file_id: str = None, filename: str = None, start_word: int = 0) -> tuple[bool, str]:
        """Load file content and initialize reading state."""
        attachment = self._find_attachment(file_id, filename)
        
        if not attachment:
            return False, "No file found. Please upload a file first."
        
        file_path = self._get_file_path(attachment)
        
        if not file_path:
            content = self.session_memory.get_attachment_content(attachment['id'])
            if not content:
                return False, f"Could not load content for {attachment['filename']}"
            self.total_words = len(content.split())
            self.current_content = content
        else:
            self.total_words = count_total_words(file_path)
            self.current_file_path = file_path
        
        self.current_file_id = attachment['id']
        self.current_file_name = attachment['filename']
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
            is_paused=False
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
            self.current_file_path,
            start_word,
            self.chunk_size,
            self.chunks_per_load
        )
        
        if not chunks:
            return False
        
        self.current_chunks = chunks
        self.loaded_start_word = actual_start
        self.loaded_end_word = actual_end
        self.current_chunk_index = 0
        
        self.position_manager.update_loaded_range(
            self.current_file_id, 
            actual_start, 
            actual_end
        )
        
        print(f"[FileReading] Loaded chunks: words {actual_start}-{actual_end} ({len(chunks)} chunks)")
        return True
    
    def _load_chunks_from_content(self) -> bool:
        """Load chunks from in-memory content (for URLs)."""
        if not self.current_content:
            return False
        
        words = self.current_content.split()
        start_word = min(self.current_word_index, len(words))
        end_word = min(start_word + (self.chunk_size * self.chunks_per_load), len(words))
        
        if start_word >= end_word:
            return False
        
        content_slice = ' '.join(words[start_word:end_word])
        chunks = split_into_chunks(content_slice, self.chunk_size)
        
        if not chunks:
            return False
        
        self.current_chunks = chunks
        self.loaded_start_word = start_word
        self.loaded_end_word = end_word
        self.current_chunk_index = 0
        
        self.position_manager.update_loaded_range(
            self.current_file_id,
            start_word,
            end_word
        )
        
        print(f"[FileReading] Loaded URL chunks: words {start_word}-{end_word} ({len(chunks)} chunks)")
        return True
        return True
    
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
                    print(f"[FileReading] No more chunks to load, finishing remaining queue")
                total_chunks = len(self.current_chunks)
            
            # Get current queue size from voice daemon
            current_queue_size = 0
            if self.voice_daemon:
                current_queue_size = getattr(self.voice_daemon, 'queue_size', 0)
            
            # Only add chunk if queue has room
            if current_queue_size <= min_queue_size and self.current_chunk_index < len(self.current_chunks):
                chunk = self.current_chunks[self.current_chunk_index]
                self.current_chunk_index += 1
                
                chunk_words = len(chunk.split())
                self.current_word_index += chunk_words
                
                self.position_manager.update_word_index(self.current_file_id, self.current_word_index)
                
                if self.voice_daemon and self.is_reading and not self.is_paused:
                    progress = int((self.current_word_index / self.total_words) * 100) if self.total_words > 0 else 0
                    print(f"[FileReading] Queued chunk {self.current_chunk_index}/{total_chunks} ({chunk_words} words) - {progress}%, queue={current_queue_size}")
                    result = self.voice_daemon.speak_file_content(
                        text=chunk,
                        paragraph_num=self.current_chunk_index,
                        language="auto"
                    )
                    
                    if not result.get('success'):
                        print(f"[FileReading] TTS failed for chunk {self.current_chunk_index}: {result.get('error')}")
            else:
                # Small sleep to avoid busy loop, but much shorter than before
                time.sleep(0.05)
        
        if self.current_word_index >= self.total_words:
            self.position_manager.mark_completed(self.current_file_id)
        
        self.is_reading = False
        print(f"[FileReading] Background reading finished. Read {self.current_word_index}/{self.total_words} words")
    
    async def execute(
        self, 
        file_id: str = None, 
        filename: str = None,
        url: str = None,
        action: str = "read",
        chunks_per_load: int = 3,
        chunk_size: int = 100
    ) -> Dict[str, Any]:
        """Execute reading action."""
        print(f"[FileReading] execute() called with file_id={file_id}, url={url}, action={action}, is_reading={self.is_reading}")
        
        self.chunks_per_load = chunks_per_load if chunks_per_load else 3
        self.chunk_size = chunk_size if chunk_size else 100
        
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
                existing_position = self.position_manager.get_position(self.current_file_id)
            
            if existing_position and existing_position.current_word_index > 0:
                resume_msg = f"Resuming from word {existing_position.current_word_index}"
                print(f"[FileReading] {resume_msg}")
            
            return self._start_reading(file_id, filename, existing_position)
        
        return {
            "success": False,
            "error": f"Unknown action: {action}"
        }
    
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
            return {
                "success": False,
                "error": message,
                "status": "load_failed"
            }
        
        # Update file_id to match position manager
        self.current_file_id = url_file_id
        
        # Start reading in background
        if self._reading_thread and self._reading_thread.is_alive():
            return self._get_reading_status()
        
        self.is_reading = True
        self.is_paused = False
        
        self._reading_thread = threading.Thread(target=self._read_background, daemon=True)
        self._reading_thread.start()
        
        return self._get_reading_status()
    
    def _start_reading(self, file_id: str = None, filename: str = None, existing_position: ReadingPosition = None) -> Dict[str, Any]:
        """Start or continue reading."""
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
                return {
                    "success": False,
                    "error": message,
                    "status": "no_file"
                }
        
        if self._reading_thread and self._reading_thread.is_alive():
            print(f"[FileReading] Thread already running")
            return self._get_reading_status()
        
        print(f"[FileReading] Starting background thread from word {start_word}")
        self.is_reading = True
        self.is_paused = False
        
        self._reading_thread = threading.Thread(target=self._read_background, daemon=True)
        self._reading_thread.start()
        
        return self._get_reading_status()
    
    def _get_reading_status(self) -> Dict[str, Any]:
        """Get current reading status."""
        progress_percent = int((self.current_word_index / self.total_words) * 100) if self.total_words > 0 else 0
        
        remaining_words = self.total_words - self.current_word_index
        remaining_chunks = remaining_words // self.chunk_size + (1 if remaining_words % self.chunk_size else 0)
        
        return {
            "success": True,
            "status": "reading" if self.is_reading else ("paused" if self.is_paused else "stopped"),
            "message": f"Reading {self.current_file_name}" if self.current_file_name else "Reading",
            "total_words": self.total_words,
            "current_word_index": self.current_word_index,
            "progress_percent": progress_percent,
            "remaining_chunks": remaining_chunks,
            "more_content": self.current_word_index < self.total_words,
            "is_paused": self.is_paused
        }
    
    def pause_reading(self) -> Dict[str, Any]:
        """Pause the current reading session."""
        if not self.is_reading:
            return {
                "success": False,
                "error": "Not currently reading",
                "status": "not_reading"
            }
        
        self.is_reading = False
        self.is_paused = True
        
        if self.voice_daemon:
            self.voice_daemon.stop_current()
        
        if self.current_file_id:
            self.position_manager.mark_paused(self.current_file_id)
            self.position_manager.update_word_index(self.current_file_id, self.current_word_index)
        
        progress_percent = int((self.current_word_index / self.total_words) * 100) if self.total_words > 0 else 0
        
        return {
            "success": True,
            "status": "paused",
            "message": f"Paused at word {self.current_word_index} of {self.total_words} ({progress_percent}%)",
            "current_word_index": self.current_word_index,
            "total_words": self.total_words,
            "progress_percent": progress_percent
        }
    
    def resume_reading(self) -> Dict[str, Any]:
        """Resume the paused reading session."""
        if not self.is_paused:
            if self.is_reading:
                return {
                    "success": False,
                    "error": "Already reading",
                    "status": "reading"
                }
            return {
                "success": False,
                "error": "No paused reading to resume",
                "status": "not_paused"
            }
        
        if not self.current_file_id:
            return {
                "success": False,
                "error": "No file to resume",
                "status": "no_file"
            }
        
        print(f"[FileReading] Resuming from word {self.current_word_index}")
        
        self.is_reading = True
        self.is_paused = False
        
        self.position_manager.mark_resumed(self.current_file_id)
        
        if self._reading_thread and self._reading_thread.is_alive():
            print(f"[FileReading] Thread already running")
            return self._get_reading_status()
        
        self._reading_thread = threading.Thread(target=self._read_background, daemon=True)
        self._reading_thread.start()
        
        return self._get_reading_status()
    
    def stop_reading(self) -> Dict[str, Any]:
        """Stop the current reading session completely."""
        was_reading = self.is_reading or self.is_paused
        words_read = self.current_word_index
        total_words = self.total_words
        
        self.is_reading = False
        self.is_paused = False
        
        if self.voice_daemon:
            self.voice_daemon.stop_current()
        
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
            "message": f"Stopped reading. Read {words_read}/{total_words} words." if was_reading else "Was not reading anything."
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
                            "progress_percent": pos.get_progress_percent()
                        }
                        for pos in unfinished
                    ],
                    "message": f"You have {len(unfinished)} unfinished reading(s)"
                }
            return {
                "is_reading": False,
                "is_paused": False,
                "message": "No file currently being read"
            }
        
        if self.is_paused:
            progress_percent = int((self.current_word_index / self.total_words) * 100) if self.total_words > 0 else 0
            return {
                "is_reading": False,
                "is_paused": True,
                "file_id": self.current_file_id,
                "file_name": self.current_file_name,
                "current_word_index": self.current_word_index,
                "total_words": self.total_words,
                "progress_percent": progress_percent,
                "message": f"Paused at word {self.current_word_index} of {self.total_words} ({progress_percent}%)"
            }
        
        if self.is_reading:
            return self._get_reading_status()
        
        progress_percent = int((self.current_word_index / self.total_words) * 100) if self.total_words > 0 else 0
        
        return {
            "is_reading": False,
            "is_paused": False,
            "file_id": self.current_file_id,
            "file_name": self.current_file_name,
            "current_word_index": self.current_word_index,
            "total_words": self.total_words,
            "progress_percent": progress_percent,
            "message": f"At word {self.current_word_index} of {self.total_words} ({progress_percent}%)"
        }
    
    def get_progress_text(self) -> str:
        """Get human-readable progress text."""
        if not self.current_file_id:
            return "Not reading any file"
        
        progress_percent = int((self.current_word_index / self.total_words) * 100) if self.total_words > 0 else 0
        
        if self.is_paused:
            return f"Paused at word {self.current_word_index} of {self.total_words} ({progress_percent}%) - {self.current_file_name}"
        
        if self.is_reading:
            return f"Reading at word {self.current_word_index} of {self.total_words} ({progress_percent}%) - {self.current_file_name}"
        
        return f"At word {self.current_word_index} of {self.total_words} ({progress_percent}%) - {self.current_file_name}"
