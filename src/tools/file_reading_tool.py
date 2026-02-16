"""
File Reading Tool - MCP-based chunk-by-chunk file reading.

This tool gives the LLM explicit control over file reading, allowing
natural stopping by simply not calling the tool again.
"""

import asyncio
import threading
import time
from typing import Any, Dict, Optional
from . import BaseTool


class FileReadingTool(BaseTool):
    """
    Read a file chunk by chunk with automatic background reading.
    
    Each call starts reading the file in background chunks (~100 words each).
    The reading continues automatically until:
    - All content is read
    - User says "stop reading" (which calls stop_reading)
    - A new file is loaded (restarts reading)
    """
    
    def __init__(self, config: dict, session_memory=None, voice_daemon=None):
        super().__init__(config)
        self.session_memory = session_memory
        self.voice_daemon = voice_daemon
        
        # Reading state
        self.current_file_id: Optional[str] = None
        self.current_content: str = ""
        self.current_position: int = 0
        self.chunk_size: int = 100  # words per chunk
        self.is_reading: bool = False
        self.words_read: int = 0
        self.total_words: int = 0
        
    def set_session_memory(self, session_memory):
        """Set the session memory reference."""
        self.session_memory = session_memory
        
    def set_voice_daemon(self, voice_daemon):
        """Set the voice daemon reference."""
        self.voice_daemon = voice_daemon
    
    def _get_description(self) -> str:
        return (
            "Read a file aloud in background chunks. Each call starts reading ~100 words at a time. "
            "The reading continues automatically until finished. "
            "Use stop_file_reading to stop at any time."
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
                }
            },
            "required": []
        }
    
    def _split_into_chunks(self, text: str, chunk_size: int = 100) -> list:
        """Split text into chunks of approximately chunk_size words."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_count = 0
        
        for word in words:
            current_chunk.append(word)
            current_count += 1
            
            # Check if we should end this chunk
            if current_count >= chunk_size:
                # Try to end at a sentence boundary
                if word.endswith(('.', '!', '?')) or current_count >= chunk_size + 20:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_count = 0
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _load_file(self, file_id: str = None, filename: str = None) -> tuple[bool, str]:
        """Load file content. Returns (success, message)."""
        attachment = None
        
        # Try to find by file_id
        if file_id:
            attachment = next(
                (a for a in self.session_memory.attachments if a['id'] == file_id),
                None
            )
        
        # Try to find by filename
        if not attachment and filename:
            attachment = self.session_memory.find_attachment_by_name(filename)
        
        # Fall back to most recent
        if not attachment:
            attachment = self.session_memory.get_last_attachment()
        
        if not attachment:
            return False, "No file found. Please upload a file first."
        
        # Load content
        content = self.session_memory.get_attachment_content(attachment['id'])
        if not content:
            return False, f"Could not load content for {attachment['filename']}"
        
        # Update state
        self.current_file_id = attachment['id']
        self.current_content = content
        self.current_position = 0
        self.total_words = len(content.split())
        self.words_read = 0
        self.is_reading = True
        
        return True, f"Loaded {attachment['filename']} ({self.total_words} words)"
    
    def _read_background(self):
        """Background thread to read all chunks."""
        print(f"[FileReading] _read_background() started, is_reading={self.is_reading}")
        chunks = self._split_into_chunks(self.current_content, self.chunk_size)
        total_chunks = len(chunks)
        
        print(f"[FileReading] Background reading started: {total_chunks} chunks")
        
        while self.current_position < total_chunks and self.is_reading:
            # Check stop flag
            if not self.is_reading:
                print(f"[FileReading] Stop flag detected, stopping background reading")
                break
            
            chunk = chunks[self.current_position]
            self.current_position += 1
            chunk_words = len(chunk.split())
            self.words_read += chunk_words
            
            # Speak via VoiceDaemon (NORMAL priority - doesn't interrupt)
            if self.voice_daemon and self.is_reading:
                print(f"[FileReading] Speaking chunk {self.current_position}/{total_chunks} ({chunk_words} words)")
                result = self.voice_daemon.speak_file_content(
                    text=chunk,
                    paragraph_num=self.current_position,
                    language="auto"
                )
                
                if not result.get('success'):
                    print(f"[FileReading] TTS failed for chunk {self.current_position}: {result.get('error')}")
            
            # Small delay between chunks
            time.sleep(0.5)
        
        self.is_reading = False
        print(f"[FileReading] Background reading finished. Read {self.words_read}/{self.total_words} words")
    
    async def execute(self, file_id: str = None, filename: str = None) -> Dict[str, Any]:
        """Start reading the file in background, chunk by chunk."""
        print(f"[FileReading] execute() called with file_id={file_id}, filename={filename}, is_reading={self.is_reading}")
        
        # Check if we need to load a file (BEFORE setting is_reading flag)
        needs_load = not self.is_reading or (file_id and file_id != self.current_file_id)
        
        if needs_load:
            print(f"[FileReading] Need to load file")
            success, message = self._load_file(file_id, filename)
            if not success:
                return {
                    "success": False,
                    "error": message,
                    "status": "no_file"
                }
            print(f"[FileReading] {message}")
        
        # Check if we already started the background thread for this file
        # Use current_content to determine if we need to start thread
        if not self.current_content:
            print(f"[FileReading] No content loaded, returning error")
            return {
                "success": False,
                "error": "No file content loaded",
                "status": "no_file"
            }
        
        # Check if we need to start the background thread
        # We start it if we just loaded the file (needs_load=True) OR if no thread was started yet (current_position=0)
        if needs_load or self.current_position == 0:
            print(f"[FileReading] Starting background thread... (needs_load={needs_load}, position={self.current_position})")
            self.is_reading = True
            self.current_position = 0
            self.words_read = 0
            
            reading_thread = threading.Thread(target=self._read_background, daemon=True)
            reading_thread.start()
            print(f"[FileReading] Thread started, is_alive={reading_thread.is_alive()}")
        
        chunks = self._split_into_chunks(self.current_content, self.chunk_size)
        
        return {
            "success": True,
            "status": "reading_started",
            "message": f"Started reading ({len(chunks)} chunks)",
            "total_words": self.total_words,
            "chunks_total": len(chunks),
            "progress_percent": 0,
            "more_content": True,
            "instruction": "Say 'stop reading' to stop at any time"
        }
    
    def stop_reading(self) -> Dict[str, Any]:
        """Stop the current reading session."""
        was_reading = self.is_reading
        words_read = self.words_read
        total_words = self.total_words
        
        # Just set the flag - the background thread will check it
        self.is_reading = False
        
        # Also stop voice daemon to stop immediately
        if self.voice_daemon:
            self.voice_daemon.stop_current()
        
        # Reset state
        self.current_file_id = None
        self.current_content = ""
        self.current_position = 0
        self.words_read = 0
        self.total_words = 0
        
        return {
            "success": True,
            "was_reading": was_reading,
            "words_read": words_read,
            "total_words": total_words,
            "message": f"Stopped reading. Read {words_read}/{total_words} words." if was_reading else "Was not reading anything."
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current reading status."""
        if not self.is_reading:
            return {
                "is_reading": False,
                "message": "No file currently being read"
            }
        
        progress_percent = int((self.words_read / self.total_words) * 100) if self.total_words > 0 else 0
        
        return {
            "is_reading": True,
            "file_id": self.current_file_id,
            "position": self.current_position,
            "words_read": self.words_read,
            "total_words": self.total_words,
            "progress_percent": progress_percent
        }
