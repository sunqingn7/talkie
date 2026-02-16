"""
File Reading Tool - MCP-based chunk-by-chunk file reading.

This tool gives the LLM explicit control over file reading, allowing
natural stopping by simply not calling the tool again.
"""

import asyncio
from typing import Any, Dict, Optional
from . import BaseTool


class FileReadingTool(BaseTool):
    """
    Read a file chunk by chunk with explicit LLM control.
    
    Unlike the old read_file_aloud which queued everything at once,
    this tool reads ONE chunk per call, giving the LLM control over:
    - When to start reading
    - How fast to progress (chunk by chunk)
    - When to stop (simply don't call again)
    
    Usage:
    1. First call: Automatically loads the most recent file
    2. Subsequent calls: Continue from where we left off
    3. Stop: LLM simply stops calling the tool
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
            "Read a file aloud chunk by chunk. Each call reads and speaks the next "
            "portion of the file (~100 words). The LLM controls the pacing by calling "
            "this tool repeatedly. To stop reading, simply don't call this tool again. "
            "If no file_id is provided, reads the most recently uploaded file."
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
                    "description": "Optional: partial filename to search for. Used if file_id not provided."
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
    
    async def execute(self, file_id: str = None, filename: str = None) -> Dict[str, Any]:
        """Read and speak the next chunk of the file."""
        
        # Check if we need to load a file
        if not self.is_reading or (file_id and file_id != self.current_file_id):
            success, message = self._load_file(file_id, filename)
            if not success:
                return {
                    "success": False,
                    "error": message,
                    "status": "no_file"
                }
            print(f"[FileReading] {message}")
        
        # Split content into chunks
        chunks = self._split_into_chunks(self.current_content, self.chunk_size)
        
        # Check if we've finished
        if self.current_position >= len(chunks):
            self.is_reading = False
            return {
                "success": True,
                "status": "finished",
                "message": f"Finished reading. Total: {self.words_read}/{self.total_words} words",
                "progress_percent": 100,
                "more_content": False
            }
        
        # Get next chunk
        chunk = chunks[self.current_position]
        self.current_position += 1
        chunk_words = len(chunk.split())
        self.words_read += chunk_words
        
        # Speak the chunk via VoiceDaemon (HIGH priority for immediate response)
        if self.voice_daemon:
            try:
                print(f"[FileReading] Speaking chunk {self.current_position}/{len(chunks)} ({chunk_words} words)")
                result = self.voice_daemon.speak_immediately(
                    text=chunk,
                    language="auto"
                )
                
                if not result.get('success'):
                    return {
                        "success": False,
                        "error": f"TTS failed: {result.get('error')}",
                        "status": "tts_error"
                    }
            except Exception as e:
                print(f"[FileReading] TTS error: {e}")
                return {
                    "success": False,
                    "error": f"TTS error: {str(e)}",
                    "status": "tts_error"
                }
        else:
            return {
                "success": False,
                "error": "Voice daemon not available",
                "status": "no_voice_daemon"
            }
        
        # Calculate progress
        progress_percent = int((self.words_read / self.total_words) * 100) if self.total_words > 0 else 0
        more_content = self.current_position < len(chunks)
        
        return {
            "success": True,
            "status": "reading",
            "message": f"Read chunk {self.current_position} of {len(chunks)} ({progress_percent}% complete)",
            "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk,
            "words_read": self.words_read,
            "total_words": self.total_words,
            "progress_percent": progress_percent,
            "more_content": more_content,
            "chunks_total": len(chunks),
            "chunks_read": self.current_position
        }
    
    def stop_reading(self) -> Dict[str, Any]:
        """Stop the current reading session and reset state."""
        was_reading = self.is_reading
        chunks_read = self.current_position
        
        # Reset state
        self.current_file_id = None
        self.current_content = ""
        self.current_position = 0
        self.is_reading = False
        self.words_read = 0
        self.total_words = 0
        
        # Also stop voice daemon
        if self.voice_daemon:
            self.voice_daemon.stop_current()
        
        return {
            "success": True,
            "was_reading": was_reading,
            "chunks_read": chunks_read,
            "message": "Reading stopped." if was_reading else "Was not reading anything."
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
