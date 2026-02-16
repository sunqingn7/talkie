"""
TTS Reader Tool - Read text files with paragraph-by-paragraph TTS via Voice Daemon.
Supports stopping the reading at any time.
"""

import asyncio
import concurrent.futures
import subprocess
import tempfile
import threading
import time
import re
import os
from typing import Any, Dict, List, Optional, Tuple
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
        self.voice_daemon = None # Will be set by MCP server
        
        # Parallel audio generation
        self.PRE_GENERATE_COUNT = 3  # Number of audio files to pre-generate
        self.audio_cache: Dict[int, Tuple[str, threading.Event]] = {}  # batch_idx -> (audio_file_path, ready_event)
        self.generation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="talkie_audio_cache_"))
        self.batches: List[str] = []  # Store batches for generation
        self.language = "auto"
        self.batch_lock = threading.Lock()

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
    
    def batch_paragraphs(self, paragraphs: List[str], max_words: int = 100) -> List[str]:
        """
        Batch paragraphs together to reach approximately max_words per batch.
        
        This reduces the number of TTS calls and makes speech flow better.
        """
        batches = []
        current_batch = []
        current_word_count = 0
        
        for paragraph in paragraphs:
            word_count = len(paragraph.split())
            
            # If adding this paragraph would exceed max_words and we already have content,
            # finalize the current batch and start a new one
            if current_word_count > 0 and current_word_count + word_count > max_words:
                batches.append('\n\n'.join(current_batch))
                current_batch = [paragraph]
                current_word_count = word_count
            else:
                current_batch.append(paragraph)
                current_word_count += word_count
        
        # Don't forget the last batch
        if current_batch:
            batches.append('\n\n'.join(current_batch))
        
        return batches

    def _generate_audio_for_batch(self, batch_idx: int, batch_text: str) -> str:
        """Generate audio file for a batch in background.
        
        Args:
            batch_idx: Index of the batch
            batch_text: Text content to convert to speech
            
        Returns:
            Path to generated audio file or empty string if failed
        """
        if self.stop_event.is_set():
            return ""
        
        try:
            # Use Edge TTS if available
            if self.voice_daemon and self.voice_daemon.tts_tool:
                tts_tool = self.voice_daemon.tts_tool
                
                # Check if using Edge TTS
                if hasattr(tts_tool, 'edge_tts_tool') and tts_tool.edge_tts_tool:
                    edge_tool = tts_tool.edge_tts_tool
                    
                    # Generate audio file
                    import asyncio
                    audio_path = self.temp_dir / f"batch_{batch_idx}_{int(time.time())}.wav"
                    
                    # Get event loop or create one
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Run TTS generation
                    voice = edge_tool.get_current_voice()
                    result = loop.run_until_complete(
                        edge_tool.execute(text=batch_text, voice=voice, speed=1.0)
                    )
                    
                    if result.get('success'):
                        output_file = result.get('output_file')
                        # Convert to WAV if needed
                        if output_file and os.path.exists(output_file):
                            wav_file = str(audio_path)
                            import subprocess
                            subprocess.run([
                                "ffmpeg", "-y", "-i", output_file,
                                "-ar", "24000", "-ac", "1", "-sample_fmt", "s16",
                                wav_file
                            ], check=False, capture_output=True, timeout=30)
                            
                            if os.path.exists(wav_file):
                                print(f"[TTSReader] ✅ Generated audio for batch {batch_idx+1}: {wav_file}")
                                return wav_file
                    
                    print(f"[TTSReader] ⚠️ Failed to generate audio for batch {batch_idx+1}")
                    return ""
            
            print(f"[TTSReader] ⚠️ No TTS tool available for batch {batch_idx+1}")
            return ""
            
        except Exception as e:
            print(f"[TTSReader] ⚠️ Error generating audio for batch {batch_idx+1}: {e}")
            return ""

    def _pre_generate_initial_files(self, batches: List[str], language: str):
        """Pre-generate the first 3 audio files before starting to play.
        
        Args:
            batches: List of text batches
            language: Language code
        """
        self.batches = batches
        self.language = language
        
        count = min(self.PRE_GENERATE_COUNT, len(batches))
        print(f"[TTSReader] Pre-generating {count} initial audio files...")
        
        for i in range(count):
            if self.stop_event.is_set():
                break
            
            # Create ready event
            ready_event = threading.Event()
            self.audio_cache[i] = ("", ready_event)
            
            # Submit generation task
            future = self.generation_executor.submit(
                self._generate_and_store,
                i,
                batches[i]
            )

    def _generate_and_store(self, batch_idx: int, batch_text: str):
        """Generate audio and store in cache.
        
        This is called by the thread pool executor.
        """
        audio_file = self._generate_audio_for_batch(batch_idx, batch_text)
        
        with self.batch_lock:
            if batch_idx in self.audio_cache:
                _, ready_event = self.audio_cache[batch_idx]
                self.audio_cache[batch_idx] = (audio_file, ready_event)
                ready_event.set()
                print(f"[TTSReader] Batch {batch_idx+1} audio ready: {audio_file if audio_file else 'FAILED'}")

    def _get_audio_for_batch(self, batch_idx: int) -> str:
        """Get audio file for a batch, waiting if necessary.
        
        Args:
            batch_idx: Index of the batch
            
        Returns:
            Path to audio file or empty string
        """
        if batch_idx not in self.audio_cache:
            # Not pre-generated, generate on demand
            if batch_idx < len(self.batches):
                print(f"[TTSReader] Generating audio on-demand for batch {batch_idx+1}")
                return self._generate_audio_for_batch(batch_idx, self.batches[batch_idx])
            return ""
        
        audio_file, ready_event = self.audio_cache[batch_idx]
        
        # Wait for generation to complete
        if not ready_event.is_set():
            print(f"[TTSReader] Waiting for batch {batch_idx+1} audio generation...")
            ready_event.wait(timeout=30)  # Wait up to 30 seconds
        
        return audio_file if audio_file else ""

    def _trigger_next_generation(self, current_batch_idx: int):
        """Trigger generation of next batch to maintain buffer.
        
        Args:
            current_batch_idx: Current batch being played
        """
        next_idx = current_batch_idx + self.PRE_GENERATE_COUNT
        
        if next_idx < len(self.batches) and next_idx not in self.audio_cache:
            print(f"[TTSReader] Triggering generation for batch {next_idx+1} to maintain buffer")
            
            ready_event = threading.Event()
            self.audio_cache[next_idx] = ("", ready_event)
            
            self.generation_executor.submit(
                self._generate_and_store,
                next_idx,
                self.batches[next_idx]
            )

    def _cleanup_audio_cache(self):
        """Clean up old audio files from cache."""
        with self.batch_lock:
            for batch_idx, (audio_file, _) in list(self.audio_cache.items()):
                if audio_file and os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                        print(f"[TTSReader] Cleaned up audio file for batch {batch_idx+1}")
                    except Exception as e:
                        print(f"[TTSReader] Error cleaning up audio file: {e}")
            self.audio_cache.clear()

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
        """Synchronous function to queue batched paragraphs via Voice Daemon.
        
        Uses parallel audio generation with a 3-file buffer for smoother playback.
        """
        # Clear previous audio cache
        self._cleanup_audio_cache()
        
        # Batch paragraphs by word count for better flow
        batches = self.batch_paragraphs(paragraphs[start_idx:], max_words=100)
        print(f"[TTSReader] Starting to read {len(paragraphs)} paragraphs from index {start_idx}")
        print(f"[TTSReader] Batched into {len(batches)} chunks (max ~100 words each)")
        self.is_reading = True
        self.total_paragraphs = len(paragraphs)
        
        # Pre-generate initial audio files
        self._pre_generate_initial_files(batches, language)

        try:
            for batch_idx, batch in enumerate(batches):
                # Calculate actual paragraph number for tracking
                paragraphs_in_batch = batch.count('\n\n') + 1
                self.current_paragraph = start_idx + sum(len(self.batch_paragraphs([paragraphs[start_idx + i]], max_words=100)[0].split('\n\n')) for i in range(batch_idx)) + paragraphs_in_batch

                # Check if stop was requested
                if self.stop_event.is_set():
                    print(f"[TTSReader] Stop requested, breaking at batch {batch_idx+1}")
                    break

                # Skip very short batches
                if len(batch) < 10:
                    print(f"[TTSReader] Skipping short batch {batch_idx+1}")
                    continue

                # Check stop event again before queuing
                if self.stop_event.is_set():
                    print(f"[TTSReader] Stop requested before queuing batch {batch_idx+1}")
                    break

                # Get pre-generated audio file (will wait if still generating)
                audio_file = self._get_audio_for_batch(batch_idx)
                
                # Trigger generation of next batch to maintain buffer
                self._trigger_next_generation(batch_idx)

                # Queue batch via Voice Daemon (NORMAL priority)
                if self.voice_daemon:
                    try:
                        word_count = len(batch.split())
                        print(f"[TTSReader] Queuing batch {batch_idx+1}/{len(batches)} ({word_count} words, {len(batch)} chars)")
                        
                        if audio_file and os.path.exists(audio_file):
                            print(f"[TTSReader] Using pre-generated audio: {audio_file}")
                        
                        result = self.voice_daemon.speak_file_content(
                            text=batch[:2000],
                            paragraph_num=batch_idx + 1,
                            language=language if language != "auto" else "auto",
                            audio_file=audio_file
                        )

                        if not result.get('success'):
                            print(f"[TTSReader] ⚠️ Failed to queue batch {batch_idx+1}: {result.get('error')}")
                        else:
                            print(f"[TTSReader] ✅ Queued batch {batch_idx+1}, queue position: {result.get('position')}")
                    except Exception as e:
                        print(f"[TTSReader] ⚠️ Error queuing batch {batch_idx+1}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[TTSReader] ⚠️ Voice daemon not available")
                    break

                # Check stop event again after queuing
                if self.stop_event.is_set():
                    print(f"[TTSReader] Stop requested after queuing batch {batch_idx+1}")
                    break

                # Small yield to allow generation threads to work
                time.sleep(0.05)

            print(f"[TTSReader] Finished reading loop. Processed {self.current_paragraph} paragraphs")

        except Exception as e:
            print(f"[TTSReader] ⚠️ Error in reading thread: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.is_reading = False
            self.stop_event.clear()
            self._cleanup_audio_cache()
            print(f"[TTSReader] Reading thread ended")

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
