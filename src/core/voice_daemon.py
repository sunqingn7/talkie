"""
Voice Daemon - Centralized TTS queue management system.

Provides priority-based speech queue:
- HIGH priority: LLM responses (interrupts current, speaks immediately after)
- NORMAL priority: File paragraphs (queued sequentially)

The daemon runs in a separate thread and processes the queue continuously.
"""

import asyncio
import subprocess
import threading
import queue
import time
from typing import Any, Dict, Optional, Callable
from enum import IntEnum
from dataclasses import dataclass, field


class Priority(IntEnum):
    """Speech priority levels."""
    HIGH = 1      # LLM responses - interrupt and speak immediately
    NORMAL = 2    # File reading - queue sequentially
    LOW = 3       # Background notifications


@dataclass
class SpeechRequest:
    """A request to speak text."""
    text: str
    priority: Priority
    language: str = "auto"
    speaker_id: Optional[str] = None
    speed: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        # For priority queue: lower priority number = higher priority
        if self.priority != other.priority:
            return self.priority < other.priority
        # Same priority: earlier timestamp first
        return self.timestamp < other.timestamp


class VoiceDaemon:
    """
    Centralized voice/TTS daemon with priority queue.
    
    Features:
    - Priority-based queue (HIGH for LLM, NORMAL for files)
    - Interrupt capability for high-priority messages
    - Sequential processing in daemon thread
    - Status tracking and control
    """
    
    def __init__(self, tts_tool=None):
        self.tts_tool = tts_tool
        self.speech_queue = queue.PriorityQueue()
        self.daemon_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.stop_event = threading.Event()
        self.current_speech_event = threading.Event()

        # Status tracking
        self.is_speaking = False
        self.current_text = ""
        self.queue_size = 0
        self.current_audio_process: Optional[subprocess.Popen] = None  # Track current playing audio
        self.current_priority: Optional[Priority] = None  # Track priority of current speech
        
        # Pause/Resume tracking for file reading
        self.paused_file_request: Optional[SpeechRequest] = None  # Stores interrupted file reading
        self.interrupted_by_high_priority = threading.Event()  # Signal that high priority interrupted
        
        self.stats = {
            "total_speeches": 0,
            "high_priority": 0,
            "normal_priority": 0,
            "interrupted": 0
        }
        
        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_queue_empty: Optional[Callable] = None
    
    def set_tts_tool(self, tts_tool):
        """Set the TTS tool reference."""
        self.tts_tool = tts_tool
    
    def start(self):
        """Start the voice daemon thread."""
        if self.is_running:
            print("[VoiceDaemon] Already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.daemon_thread = threading.Thread(
            target=self._daemon_loop,
            name="VoiceDaemon",
            daemon=True
        )
        self.daemon_thread.start()
        print("[VoiceDaemon] Started")
    
    def stop(self):
        """Stop the voice daemon completely."""
        if not self.is_running:
            return

        print("[VoiceDaemon] Stopping daemon...")
        # Set flag to stop daemon loop
        self._daemon_stop = True

        # Clear the queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break

        # Wait for thread to finish
        if self.daemon_thread and self.daemon_thread.is_alive():
            self.daemon_thread.join(timeout=3.0)

        self.is_running = False
        self.is_speaking = False
        # Clean up the stop flag
        if hasattr(self, '_daemon_stop'):
            delattr(self, '_daemon_stop')
        print("[VoiceDaemon] Stopped")
    
    def _daemon_loop(self):
        """Main daemon loop - continuously process speech queue."""
        # Create event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while True:
                try:
                    # Check if daemon should stop completely
                    if hasattr(self, '_daemon_stop'):
                        break

                    # Get next speech request (blocking with timeout)
                    request = self.speech_queue.get(timeout=0.5)
                    self.queue_size = self.speech_queue.qsize()

                    # Check if stop was requested (for current speech only)
                    if self.stop_event.is_set():
                        # Just reset and continue - queue was cleared
                        self.stop_event.clear()
                        print("[VoiceDaemon] Stop signal received, queue cleared, continuing...")
                        continue

                    # Process the speech request
                    self._process_speech(request, loop)

                except queue.Empty:
                    # No items in queue
                    self.queue_size = 0
                    if self.on_queue_empty:
                        try:
                            self.on_queue_empty()
                        except:
                            pass
                    continue
                except Exception as e:
                    print(f"[VoiceDaemon] Error in daemon loop: {e}")
                    time.sleep(0.1)

        finally:
            # Clean up event loop
            try:
                loop.close()
            except:
                pass
    
    def _process_speech(self, request: SpeechRequest, loop: asyncio.AbstractEventLoop):
        """Process a single speech request."""
        print(f"[VoiceDaemon] _process_speech called with text: {request.text[:50]}...")
        
        if not self.tts_tool:
            print("[VoiceDaemon] No TTS tool available")
            return
        
        print(f"[VoiceDaemon] TTS tool is available: {type(self.tts_tool)}")
        
        self.is_speaking = True
        self.current_text = request.text
        self.current_priority = request.priority  # Track priority for interruption logic
        self.stats["total_speeches"] += 1
        
        if request.priority == Priority.HIGH:
            self.stats["high_priority"] += 1
        else:
            self.stats["normal_priority"] += 1
        
        # Trigger callback
        if self.on_speech_start:
            try:
                self.on_speech_start(request)
            except:
                pass
        
        try:
            # Check if we should stop before speaking
            if self.stop_event.is_set():
                return
            
            # Show priority indicator
            priority_indicator = "⚡" if request.priority == Priority.HIGH else "📖"
            short_text = request.text[:60] + "..." if len(request.text) > 60 else request.text
            print(f"[VoiceDaemon] {priority_indicator} Speaking: {short_text}")
            
            # Execute TTS in the event loop
            print(f"[VoiceDaemon] Executing TTS for: {short_text}")
            
            try:
                # Since we're already in the daemon thread with the event loop,
                # use run_until_complete instead of run_coroutine_threadsafe
                print(f"[VoiceDaemon] Running TTS coroutine directly in event loop...")
                
                # Check stop before TTS
                if self.stop_event.is_set():
                    print(f"[VoiceDaemon] Stop requested before TTS, skipping")
                    return
                
                result = loop.run_until_complete(
                    self.tts_tool.execute(
                        text=request.text,
                        language=request.language,
                        speaker_id=request.speaker_id,
                        speed=request.speed
                    )
                )
                print(f"[VoiceDaemon] TTS result: success={result.get('success')}, error={result.get('error', 'None')}")
                
                # Check stop immediately after TTS completes
                if self.stop_event.is_set():
                    print(f"[VoiceDaemon] Stop requested after TTS, skipping wait")
                    return
                
                if result.get('success'):
                    # Get the audio process from TTS result (NON-BLOCKING)
                    audio_process = result.get('audio_process')
                    
                    if audio_process:
                        # Store in daemon for stop tracking (CRITICAL for immediate stop)
                        self.current_audio_process = audio_process
                        print(f"[VoiceDaemon] Stored audio process PID {audio_process.pid} in daemon")
                        
                        # Wait for actual audio playback to finish
                        print(f"[VoiceDaemon] Waiting for audio to finish (process PID {audio_process.pid})")
                        while audio_process.poll() is None:
                            if self.stop_event.is_set():
                                print(f"[VoiceDaemon] Stop requested, terminating audio process")
                                try:
                                    audio_process.terminate()
                                    try:
                                        audio_process.wait(timeout=1)
                                    except:
                                        audio_process.kill()
                                except Exception as e:
                                    print(f"[VoiceDaemon] Error terminating audio: {e}")
                                return
                            time.sleep(0.05)  # Check every 50ms
                        print(f"[VoiceDaemon] Audio process finished")
                    else:
                        # Fallback: old sleep method if no process available
                        estimated_duration = len(request.text) / 15  # ~15 chars per second
                        wait_time = min(estimated_duration + 0.5, 30)
                        print(f"[VoiceDaemon] No audio process, sleeping for {wait_time:.1f}s")
                        
                        sleep_start = time.time()
                        while time.time() - sleep_start < wait_time:
                            if self.stop_event.is_set():
                                print(f"[VoiceDaemon] Sleep interrupted by stop signal")
                                return
                            time.sleep(0.1)

                    print(f"[VoiceDaemon] Finished speaking: {short_text}")
                else:
                    print(f"[VoiceDaemon] TTS failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"[VoiceDaemon] TTS error during execution: {e}")
                import traceback
                traceback.print_exc()
        
        finally:
            self.is_speaking = False
            self.current_text = ""
            self.current_audio_process = None  # Clear the audio process reference
            self.current_priority = None  # Clear current priority
            
            # Trigger callback
            if self.on_speech_end:
                try:
                    self.on_speech_end(request)
                except:
                    pass
    
    def enqueue(self, text: str, priority: Priority = Priority.NORMAL,
                language: str = "auto", speaker_id: Optional[str] = None,
                speed: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add text to the speech queue.
        
        Args:
            text: Text to speak
            priority: Priority level (HIGH for LLM, NORMAL for files)
            language: Language code
            speaker_id: Speaker/voice ID
            speed: Speech speed
            metadata: Additional metadata
        
        Returns:
            Dict with queue position and estimated wait time
        """
        if not text or not text.strip():
            return {"success": False, "error": "No text provided"}
        
        if not self.is_running:
            return {"success": False, "error": "Voice daemon not running"}
        
        request = SpeechRequest(
            text=text.strip(),
            priority=priority,
            language=language,
            speaker_id=speaker_id,
            speed=speed,
            metadata=metadata or {}
        )
        
        # Handle high-priority requests
        if priority == Priority.HIGH:
            # Clear lower priority items from queue
            self._clear_lower_priority_items()
            
            # If currently speaking NORMAL priority (file reading), interrupt immediately
            if self.is_speaking and self.current_priority == Priority.NORMAL:
                print(f"[VoiceDaemon] ⚡ High priority speech interrupting current file reading")
                # Store the interrupted file request for later resumption
                # (Note: we don't know the exact position, but TTSReader will handle that)
                self.interrupted_by_high_priority.set()
                # Signal stop to interrupt current audio
                self.stop_event.set()
                print(f"[VoiceDaemon] ⚡ Interrupted file reading to play high priority speech")
            elif self.is_speaking:
                print(f"[VoiceDaemon] ⚡ High priority speech queued (will play after current)")
        
        # Add to queue
        self.speech_queue.put(request)
        self.queue_size = self.speech_queue.qsize()
        
        # Calculate queue position
        position = self._get_queue_position(priority)
        
        # Estimate wait time (rough estimate: 4 seconds per queued item)
        estimated_wait = position * 4
        
        priority_name = "HIGH" if priority == Priority.HIGH else "NORMAL"
        
        return {
            "success": True,
            "priority": priority_name,
            "queue_position": position,
            "estimated_wait_seconds": estimated_wait,
            "queue_size": self.queue_size,
            "is_speaking": self.is_speaking
        }
    
    def _clear_lower_priority_items(self):
        """Clear normal priority items from queue to make room for high priority."""
        # Get all items
        items = []
        while not self.speech_queue.empty():
            try:
                items.append(self.speech_queue.get_nowait())
            except queue.Empty:
                break
        
        # Keep only high priority items
        high_priority_items = [item for item in items if item.priority == Priority.HIGH]
        
        # Re-add high priority items
        for item in high_priority_items:
            self.speech_queue.put(item)
        
        # Normal priority items are dropped (they'll need to be requeued)
        dropped_count = len(items) - len(high_priority_items)
        if dropped_count > 0:
            print(f"[VoiceDaemon] Cleared {dropped_count} normal priority items for high priority speech")
            self.stats["interrupted"] += dropped_count
    
    def _get_queue_position(self, priority: Priority) -> int:
        """Get the position in queue for a given priority."""
        # This is approximate since we can't easily peek into PriorityQueue
        if priority == Priority.HIGH:
            # High priority goes first
            return 0 if not self.is_speaking else 1
        else:
            # Normal priority goes after all high priority items
            return self.queue_size
    
    def speak_immediately(self, text: str, language: str = "auto",
                         speaker_id: Optional[str] = None, speed: float = 1.0) -> Dict[str, Any]:
        """
        Speak text immediately (high priority).
        Convenience method for LLM responses.
        """
        return self.enqueue(
            text=text,
            priority=Priority.HIGH,
            language=language,
            speaker_id=speaker_id,
            speed=speed,
            metadata={"type": "llm_response"}
        )
    
    def speak_file_content(self, text: str, paragraph_num: int = 0,
                          language: str = "auto") -> Dict[str, Any]:
        """
        Queue file content for reading (normal priority).
        Convenience method for file reading.
        """
        return self.enqueue(
            text=text,
            priority=Priority.NORMAL,
            language=language,
            metadata={"type": "file_content", "paragraph": paragraph_num}
        )
    
    def stop_current(self) -> Dict[str, Any]:
        """Stop the current speech and clear the queue."""
        was_speaking = self.is_speaking
        queue_size_before = self.speech_queue.qsize()

        print(f"[VoiceDaemon] stop_current called. is_speaking={was_speaking}, queue_size={queue_size_before}")

        # Signal stop to interrupt ongoing speech/sleep
        self.stop_event.set()
        print(f"[VoiceDaemon] stop_event set")

        # CRITICAL: Directly stop the currently playing audio process
        # This is more reliable than delegating to TTS tool which might track wrong process
        if self.current_audio_process and self.current_audio_process.poll() is None:
            try:
                pid = self.current_audio_process.pid
                print(f"[VoiceDaemon] Directly terminating audio process PID {pid}")
                self.current_audio_process.terminate()
                try:
                    self.current_audio_process.wait(timeout=0.5)
                except:
                    self.current_audio_process.kill()
                    self.current_audio_process.wait(timeout=0.5)
                print(f"[VoiceDaemon] Audio process PID {pid} terminated")
            except Exception as e:
                print(f"[VoiceDaemon] Error terminating audio process: {e}")
            finally:
                self.current_audio_process = None
        
        # Also try TTS tool stop as backup
        if self.tts_tool and hasattr(self.tts_tool, 'stop_audio'):
            try:
                self.tts_tool.stop_audio()
            except Exception as e:
                print(f"[VoiceDaemon] Error in TTS tool stop_audio: {e}")

        # Clear queue
        cleared_count = 0
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break

        self.queue_size = 0
        print(f"[VoiceDaemon] Queue cleared: {cleared_count} items removed")

        # Reset stop event after a short delay to allow processing to stop
        # The daemon loop will clear it after processing the stop

        return {
            "success": True,
            "was_speaking": was_speaking,
            "cleared_from_queue": cleared_count,
            "message": f"Stopped speech. Cleared {cleared_count} items from queue."
        }
    

    def was_interrupted_by_high_priority(self) -> bool:
        """Check if file reading was interrupted by high priority speech."""
        return self.interrupted_by_high_priority.is_set()
    
    def clear_interruption_flag(self):
        """Clear the high priority interruption flag."""
        self.interrupted_by_high_priority.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get current daemon status."""
        return {
            "is_running": self.is_running,
            "is_speaking": self.is_speaking,
            "current_text": self.current_text[:100] + "..." if len(self.current_text) > 100 else self.current_text,
            "queue_size": self.queue_size,
            "stats": self.stats.copy()
        }
    
    def skip_current(self) -> Dict[str, Any]:
        """Skip the current speech and move to next in queue."""
        if not self.is_speaking:
            return {
                "success": False,
                "message": "Not currently speaking"
            }
        
        # Note: Actual interruption would require TTS tool support
        # For now, we just note the request
        return {
            "success": True,
            "message": "Skip requested (will take effect after current speech)"
        }


# Singleton instance for global access
_voice_daemon_instance: Optional[VoiceDaemon] = None


def get_voice_daemon(tts_tool=None) -> VoiceDaemon:
    """Get or create the global voice daemon instance."""
    global _voice_daemon_instance
    if _voice_daemon_instance is None:
        _voice_daemon_instance = VoiceDaemon(tts_tool)
    elif tts_tool is not None:
        _voice_daemon_instance.set_tts_tool(tts_tool)
    return _voice_daemon_instance


def reset_voice_daemon():
    """Reset the global voice daemon instance."""
    global _voice_daemon_instance
    if _voice_daemon_instance:
        _voice_daemon_instance.stop()
    _voice_daemon_instance = None
