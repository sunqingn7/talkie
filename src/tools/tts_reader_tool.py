"""
TTS Reader Tool - Read text files with paragraph-by-paragraph TTS via Voice Daemon.
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
    """Tool to read text content with TTS via Voice Daemon.
    
    Features:
    - Reads text paragraph by paragraph
    - Queues content via Voice Daemon (NORMAL priority)
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
        self.voice_daemon = None  # Will be set by MCP server
        
    def _get_description(self) -> str:
        return (
            "Read text content aloud using text-to-speech, paragraph by paragraph. "
            "Runs via Voice Daemon so you can stop it anytime by saying 'stop reading'. "
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
    
    def set_voice_daemon(self, voice_daemon):
        """Set the voice daemon reference."""
        self.voice_daemon = voice_daemon
    
    def set_tts_tool(self, tts_tool):
        """Legacy method for backward compatibility."""
        # No longer used directly, voice daemon handles TTS
        pass
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on multiple newlines or paragraph breaks
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        # Clean up and filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def is_reading_active(self) -> bool:
        """Check if currently reading."""
        return self.is_reading
    
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
        
        # Also stop the voice daemon's current speech
        if self.voice_daemon:
            self.voice_daemon.stop_current()
        
        return {
            "success": True,
            "message": f"Stopped reading at paragraph {self.current_paragraph} of {self.total_paragraphs}",
            "was_reading": True,
            "paragraphs_read": self.current_paragraph,
            "total_paragraphs": self.total_paragraphs
        }
    
    def get_reading_status(self) -> Dict[str, Any]:
        """Get current reading status."""
        # Also get voice daemon status
        daemon_status = {}
        if self.voice_daemon:
            daemon_status = self.voice_daemon.get_status()
        
        return {
            "is_reading": self.is_reading_active(),
            "current_paragraph": self.current_paragraph,
            "total_paragraphs": self.total_paragraphs,
            "progress_percent": (self.current_paragraph / self.total_paragraphs * 100) if self.total_paragraphs > 0 else 0,
            "voice_daemon": daemon_status
        }
    
    def read_paragraphs_sync(self, paragraphs: List[str], start_idx: int = 0, language: str = "auto"):
        """Synchronous function to queue paragraphs via Voice Daemon."""
        self.is_reading = True
        self.total_paragraphs = len(paragraphs)
        
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
                
                # Check stop event again before queuing
                if self.stop_event.is_set():
                    break
                
                # Queue paragraph via Voice Daemon (NORMAL priority)
                if self.voice_daemon:
                    try:
                        result = self.voice_daemon.speak_file_content(
                            text=paragraph[:500],  # Limit to 500 chars per paragraph
                            paragraph_num=i + 1,
                            language=language if language != "auto" else "auto"
                        )
                        
                        if not result.get('success'):
                            print(f"   ⚠️  Failed to queue paragraph {i+1}: {result.get('error')}")
                    except Exception as e:
                        print(f"   ⚠️  Error queuing paragraph {i+1}: {e}")
                else:
                    print(f"   ⚠️  Voice daemon not available")
                    break
                
                # Check stop event again
                if self.stop_event.is_set():
                    break
                
                # Small delay between queuing paragraphs to allow processing
                time.sleep(0.5)
        
        except Exception as e:
            print(f"   ⚠️  Error in reading thread: {e}")
        
        finally:
            self.is_reading = False
            self.stop_event.clear()
    
    async def execute(self, content: str = None, file_path: str = None, 
                      start_paragraph: int = 1, language: str = "auto") -> Dict[str, Any]:
        """Start reading text or file content with TTS via Voice Daemon."""
        
        # Check if voice daemon is available
        if not self.voice_daemon:
            return {
                "success": False,
                "error": "Voice daemon not available",
                "message": "Voice daemon not initialized. Please restart the application."
            }
        
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
        
        # Start reading in a separate thread
        self.reading_thread = threading.Thread(
            target=self.read_paragraphs_sync,
            args=(paragraphs, start_idx, language),
            daemon=True
        )
        self.reading_thread.start()
        
        # Calculate preview info
        paragraphs_to_read = len(paragraphs) - start_idx
        preview_text = paragraphs[start_idx][:200] + "..." if len(paragraphs[start_idx]) > 200 else paragraphs[start_idx]
        
        # Get daemon status
        daemon_status = self.voice_daemon.get_status() if self.voice_daemon else {}
        
        return {
            "success": True,
            "message": f"Started reading {paragraphs_to_read} paragraphs via Voice Daemon (from paragraph {start_paragraph})",
            "total_paragraphs": len(paragraphs),
            "starting_from": start_paragraph,
            "preview": preview_text,
            "instruction": "Say 'stop reading' at any time to stop",
            "voice_daemon_status": daemon_status
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
