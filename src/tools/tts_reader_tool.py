"""
TTS Reader Tool - Read text files with paragraph-by-paragraph TTS in a separate thread.
Supports stopping the reading at any time.
"""

import asyncio
import threading
import time
import re
import os
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import BaseTool


class TTSReaderTool(BaseTool):
    """Tool to read text content with TTS in a separate thread.
    
    Features:
    - Reads text paragraph by paragraph
    - Runs in a separate thread (non-blocking)
    - Can be stopped at any time
    - Integrates with file attachments
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.reading_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.is_reading = False
        self.current_paragraph = 0
        self.total_paragraphs = 0
        self.tts_tool = None  # Will be set by MCP server
        
    def _get_description(self) -> str:
        return (
            "Read text content aloud using text-to-speech, paragraph by paragraph. "
            "Runs in background so you can stop it anytime by saying 'stop reading'. "
            "Supports reading attached files or any text content."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Text content to read aloud"
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to a file to read (alternative to content)"
                },
                "start_paragraph": {
                    "type": "integer",
                    "description": "Paragraph number to start from (1-indexed)",
                    "default": 1
                },
                "language": {
                    "type": "string",
                    "description": "Language code for TTS (e.g., 'en', 'zh-cn')",
                    "default": "auto"
                }
            },
            "required": []
        }
    
    def set_tts_tool(self, tts_tool):
        """Set the TTS tool reference."""
        self.tts_tool = tts_tool
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on multiple newlines or paragraph breaks
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        # Clean up and filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def is_reading_active(self) -> bool:
        """Check if currently reading."""
        return self.is_reading and self.reading_thread and self.reading_thread.is_alive()
    
    def stop_reading(self) -> Dict[str, Any]:
        """Stop the current reading session."""
        if not self.is_reading:
            return {
                "success": False,
                "message": "Not currently reading anything",
                "was_reading": False
            }
        
        # Signal stop
        self.stop_event.set()
        self.is_reading = False
        
        # Wait a bit for thread to finish
        if self.reading_thread and self.reading_thread.is_alive():
            self.reading_thread.join(timeout=2.0)
        
        return {
            "success": True,
            "message": f"Stopped reading at paragraph {self.current_paragraph} of {self.total_paragraphs}",
            "was_reading": True,
            "paragraphs_read": self.current_paragraph,
            "total_paragraphs": self.total_paragraphs
        }
    
    def get_reading_status(self) -> Dict[str, Any]:
        """Get current reading status."""
        return {
            "is_reading": self.is_reading_active(),
            "current_paragraph": self.current_paragraph,
            "total_paragraphs": self.total_paragraphs,
            "progress_percent": (self.current_paragraph / self.total_paragraphs * 100) if self.total_paragraphs > 0 else 0
        }
    
    def read_paragraphs_sync(self, paragraphs: List[str], start_idx: int = 0, language: str = "auto"):
        """Synchronous function to read paragraphs one by one in a thread."""
        self.is_reading = True
        self.total_paragraphs = len(paragraphs)
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            for i in range(start_idx, len(paragraphs)):
                # Check if stop was requested
                if self.stop_event.is_set():
                    break
                
                self.current_paragraph = i + 1
                paragraph = paragraphs[i]
                
                # Skip very short paragraphs
                if len(paragraph) < 10:
                    continue
                
                # Check stop event again before speaking
                if self.stop_event.is_set():
                    break
                
                # Speak the paragraph - run synchronously in this thread's event loop
                if self.tts_tool:
                    try:
                        # Run the async TTS execute and wait for it to complete
                        future = asyncio.run_coroutine_threadsafe(
                            self.tts_tool.execute(
                                text=paragraph[:500],  # Limit to 500 chars per paragraph
                                language=language if language != "auto" else None
                            ),
                            loop
                        )
                        # Wait for TTS to complete (with timeout to prevent hanging)
                        result = future.result(timeout=120)  # 2 minute timeout per paragraph
                        
                        # If TTS was successful, wait a bit for audio to finish playing
                        if result.get('success'):
                            # Calculate approximate duration (rough estimate: ~3 chars per second)
                            estimated_duration = len(paragraph) / 15  # ~15 chars per second
                            time.sleep(min(estimated_duration + 0.5, 30))  # Cap at 30 seconds
                    except Exception as e:
                        print(f"   ⚠️  Error speaking paragraph {i+1}: {e}")
                
                # Check stop event again
                if self.stop_event.is_set():
                    break
        
        except Exception as e:
            print(f"   ⚠️  Error in reading thread: {e}")
        
        finally:
            self.is_reading = False
            self.stop_event.clear()
            # Clean up the event loop
            try:
                loop.close()
            except:
                pass
    
    async def execute(self, content: str = None, file_path: str = None, 
                      start_paragraph: int = 1, language: str = "auto") -> Dict[str, Any]:
        """Start reading text or file content with TTS."""
        
        # If already reading, return status
        if self.is_reading_active():
            return {
                "success": False,
                "message": "Already reading content. Say 'stop reading' to stop first.",
                "status": self.get_reading_status()
            }
        
        # Get content from file if provided
        if file_path and not content:
            try:
                path = Path(file_path)
                if not path.exists():
                    return {
                        "success": False,
                        "error": f"File not found: {file_path}"
                    }
                content = path.read_text(encoding='utf-8')
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to read file: {str(e)}"
                }
        
        if not content or not content.strip():
            return {
                "success": False,
                "error": "No content to read"
            }
        
        # Split into paragraphs
        paragraphs = self.split_into_paragraphs(content)
        
        if not paragraphs:
            return {
                "success": False,
                "error": "No readable paragraphs found in content"
            }
        
        # Validate start_paragraph
        start_idx = max(0, start_paragraph - 1)
        if start_idx >= len(paragraphs):
            return {
                "success": False,
                "error": f"Start paragraph {start_paragraph} is beyond total paragraphs ({len(paragraphs)})"
            }
        
        # Reset stop event
        self.stop_event.clear()
        
        # Start reading in a separate thread using the synchronous method
        self.reading_thread = threading.Thread(
            target=self.read_paragraphs_sync,
            args=(paragraphs, start_idx, language),
            daemon=False  # Non-daemon so it won't be killed when main thread continues
        )
        self.reading_thread.start()
        
        # Calculate preview info
        paragraphs_to_read = len(paragraphs) - start_idx
        preview_text = paragraphs[start_idx][:200] + "..." if len(paragraphs[start_idx]) > 200 else paragraphs[start_idx]
        
        return {
            "success": True,
            "message": f"Started reading {paragraphs_to_read} paragraphs (from paragraph {start_paragraph})",
            "total_paragraphs": len(paragraphs),
            "starting_from": start_paragraph,
            "preview": preview_text,
            "instruction": "Say 'stop reading' at any time to stop"
        }


class StopReadingTool(BaseTool):
    """Tool to stop the TTS reader."""
    
    def __init__(self, config: dict, reader_tool: TTSReaderTool = None):
        super().__init__(config)
        self.reader_tool = reader_tool
    
    def set_reader_tool(self, reader_tool: TTSReaderTool):
        """Set the reader tool reference."""
        self.reader_tool = reader_tool
    
    def _get_description(self) -> str:
        return "Stop the current text-to-speech reading session."
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    async def execute(self) -> Dict[str, Any]:
        """Stop the current reading."""
        if not self.reader_tool:
            return {
                "success": False,
                "error": "Reader tool not available"
            }
        
        return self.reader_tool.stop_reading()
