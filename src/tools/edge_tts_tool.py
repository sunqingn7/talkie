"""
Edge TTS Tool - Microsoft Edge Text-to-Speech
Uses Microsoft's Edge TTS service for high-quality online voices.
"""

import asyncio
import os
import tempfile
import subprocess
from typing import Any, Dict, List, Optional
from pathlib import Path
import time

from tools import BaseTool


class EdgeTTSTool(BaseTool):
    """Tool to convert text to speech using Microsoft Edge TTS service.
    
    Features:
    - High-quality neural voices
    - Multiple languages
    - No local model download required
    - Requires internet connection
    """
    
    # Available voices (subset of popular ones)
    # Full list: https://github.com/rany2/edge-tts#list-voices
    AVAILABLE_VOICES = [
        # English (US)
        {"id": "en-US-AriaNeural", "name": "Aria", "gender": "Female", "locale": "en-US"},
        {"id": "en-US-GuyNeural", "name": "Guy", "gender": "Male", "locale": "en-US"},
        {"id": "en-US-JennyNeural", "name": "Jenny", "gender": "Female", "locale": "en-US"},
        # English (UK)
        {"id": "en-GB-SoniaNeural", "name": "Sonia", "gender": "Female", "locale": "en-GB"},
        {"id": "en-GB-RyanNeural", "name": "Ryan", "gender": "Male", "locale": "en-GB"},
        # Chinese (Simplified)
        {"id": "zh-CN-XiaoxiaoNeural", "name": "Xiaoxiao", "gender": "Female", "locale": "zh-CN"},
        {"id": "zh-CN-YunjianNeural", "name": "Yunjian", "gender": "Male", "locale": "zh-CN"},
        {"id": "zh-CN-XiaoyiNeural", "name": "Xiaoyi", "gender": "Female", "locale": "zh-CN"},
        # Chinese (Traditional)
        {"id": "zh-TW-HsiaoChenNeural", "name": "HsiaoChen", "gender": "Female", "locale": "zh-TW"},
        # Japanese
        {"id": "ja-JP-NanamiNeural", "name": "Nanami", "gender": "Female", "locale": "ja-JP"},
        {"id": "ja-JP-KeitaNeural", "name": "Keita", "gender": "Male", "locale": "ja-JP"},
        # Korean
        {"id": "ko-KR-SunHiNeural", "name": "SunHi", "gender": "Female", "locale": "ko-KR"},
        # Spanish
        {"id": "es-ES-ElviraNeural", "name": "Elvira", "gender": "Female", "locale": "es-ES"},
        {"id": "es-MX-DaliaNeural", "name": "Dalia", "gender": "Female", "locale": "es-MX"},
        # French
        {"id": "fr-FR-DeniseNeural", "name": "Denise", "gender": "Female", "locale": "fr-FR"},
        # German
        {"id": "de-DE-KatjaNeural", "name": "Katja", "gender": "Female", "locale": "de-DE"},
    ]
    
    def _get_description(self) -> str:
        return (
            "Convert text to speech using Microsoft Edge TTS service. "
            "High-quality neural voices with internet connectivity. "
            "Supports multiple languages including English, Chinese, Japanese, etc."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to speak aloud"
                },
                "voice": {
                    "type": "string",
                    "description": "Voice ID (e.g., 'en-US-AriaNeural', 'zh-CN-XiaoxiaoNeural')",
                    "default": "en-US-AriaNeural"
                },
                "speed": {
                    "type": "number",
                    "description": "Speech speed (0.5 = slow, 1.0 = normal, 1.5 = fast)",
                    "default": 1.0
                }
            },
            "required": ["text"]
        }
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.tts_config = config.get("tts", {})
        self.temp_dir = tempfile.mkdtemp(prefix="talkie_edge_tts_")
        self.current_voice = self.tts_config.get('edge_voice', 'en-US-AriaNeural')
        self.current_audio_process = None
        self.is_playing = False
        self.current_audio_type = None  # "chat" or "file" or None
    
    def set_audio_type(self, audio_type: str):
        """Set the type of audio that will be played (chat or file)."""
        self.current_audio_type = audio_type
        print(f" [AUDIO DEBUG] Edge TTS: Set audio_type to {audio_type}")
    
    def get_available_voices(self) -> List[Dict]:
        """Get list of available voices."""
        return self.AVAILABLE_VOICES
    
    def get_current_voice(self) -> str:
        """Get current voice ID."""
        return self.current_voice
    
    def set_voice(self, voice_id: str) -> bool:
        """Set the voice to use."""
        # Validate voice exists
        if any(v["id"] == voice_id for v in self.AVAILABLE_VOICES):
            self.current_voice = voice_id
            # Update config
            self.tts_config['edge_voice'] = voice_id
            return True
        return False
    
    def _get_voice_for_language(self, language: str) -> str:
        """Get appropriate voice for language."""
        language_map = {
            "en": "en-US-AriaNeural",
            "en-us": "en-US-AriaNeural",
            "en-gb": "en-GB-SoniaNeural",
            "zh": "zh-CN-XiaoxiaoNeural",
            "zh-cn": "zh-CN-XiaoxiaoNeural",
            "zh-tw": "zh-TW-HsiaoChenNeural",
            "ja": "ja-JP-NanamiNeural",
            "ko": "ko-KR-SunHiNeural",
            "es": "es-ES-ElviraNeural",
            "fr": "fr-FR-DeniseNeural",
            "de": "de-DE-KatjaNeural",
        }
        return language_map.get(language.lower(), self.current_voice)
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration in seconds."""
        import subprocess
        
        try:
            # Use ffprobe to get duration
            result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", 
                "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1:n=1",
                audio_path
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        # Estimate based on file size
        try:
            import os
            file_size = os.path.getsize(audio_path)
            # Rough estimate: ~10KB per second for 128kbps audio
            return max(file_size / 10240.0, 10.0)  # Minimum 10 seconds
        except:
            return 30.0  # Fallback
    
    def _play_audio(self, audio_path: str) -> Optional[subprocess.Popen]:
        """Play audio file - NON-BLOCKING.
        
        Returns:
            subprocess.Popen: The audio player process (can be terminated to stop)
            None: If no audio player found
        """
        import subprocess
        
        print(f" [Edge TTS] Starting playback (NON-BLOCKING)")

        # Try players - ffplay first (more reliable)
        players = [
            ("ffplay -autoexit -nodisp -loglevel quiet", "FFmpeg"),
            ("paplay", "PulseAudio"),
        ]

        for player_cmd, player_name in players:
            player_executable = player_cmd.split()[0]
            try:
                result = subprocess.run(
                    f"which {player_executable}",
                    shell=True,
                    capture_output=True,
                    check=False
                )
                if result.returncode == 0:
                    cmd = f"{player_cmd} \"{audio_path}\""
                    print(f" [AUDIO DEBUG] Edge TTS: Starting {player_name} (NON-BLOCKING)")
                    process = subprocess.Popen(
                        cmd, 
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    print(f" [AUDIO DEBUG] Edge TTS: Started {player_name} with PID {process.pid}")
                    self.current_audio_process = process
                    self.is_playing = True
                    return process
            except Exception as e:
                print(f" [AUDIO DEBUG] Edge TTS: {player_name} failed: {e}")
                continue

        # Fallback to aplay
        try:
            print(f" [AUDIO DEBUG] Edge TTS: Trying aplay as fallback (NON-BLOCKING)")
            process = subprocess.Popen(
                f"aplay \"{audio_path}\"",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f" [AUDIO DEBUG] Edge TTS: Started aplay with PID {process.pid}")
            self.current_audio_process = process
            self.is_playing = True
            return process
        except Exception as e:
            print(f" [AUDIO DEBUG] Edge TTS: aplay failed: {e}")

        print(f" [AUDIO DEBUG] Edge TTS: No audio player found!")
        return None
    
    def stop_audio(self, reason: str = "general") -> bool:
        """Stop the current audio playback immediately.
        
        Args:
            reason: Why we're stopping - "chat" stops chat audio, "file" skips stopping for file reading
        """
        print(f" [AUDIO DEBUG] Edge TTS stop_audio called. reason={reason}, audio_type={self.current_audio_type}, current_process: {self.current_audio_process}")
        
        # If we're asked to stop for chat but current audio is file reading, don't stop
        if reason == "chat" and self.current_audio_type == "file":
            print(f" [AUDIO DEBUG] Edge TTS: Skipping stop - current audio is file reading")
            return False
        
        if self.current_audio_process and self.current_audio_process.poll() is None:
            try:
                print(f" [AUDIO DEBUG] Edge TTS: Stopping audio process {self.current_audio_process.pid}")
                self.current_audio_process.terminate()
                try:
                    self.current_audio_process.wait(timeout=1)
                except:
                    self.current_audio_process.kill()
                self.is_playing = False
                self.current_audio_process = None
                self.current_audio_type = None
                print(f" [AUDIO DEBUG] Edge TTS: Audio stopped")
                return True
            except Exception as e:
                print(f" [AUDIO DEBUG] Edge TTS: Error stopping audio: {e}")
        
        self.is_playing = False
        self.current_audio_type = None
        return False
        self.is_playing = False
        return False
    
    def wait_for_audio(self, timeout: Optional[float] = None) -> bool:
        """Wait for current audio to finish playing.
        
        Args:
            timeout: Max time to wait in seconds (None = wait indefinitely)
            
        Returns:
            True if audio finished, False if timeout
        """
        if not self.current_audio_process:
            return True
            
        start_time = time.time()
        while self.current_audio_process.poll() is None:
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.05)  # Check every 50ms
        
        self.is_playing = False
        self.current_audio_process = None
        return True
    
    async def execute(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> Dict[str, Any]:
        """Convert text to speech using Edge TTS."""
        if not text or not text.strip():
            return {"success": False, "error": "No text provided"}
        
        text = text.strip()
        
        # Use provided voice or current voice
        voice_id = voice or self.current_voice
        
        try:
            import edge_tts
            import time
            
            # Create output file
            output_file = os.path.join(self.temp_dir, f"edge_tts_{id(text)}.mp3")
            
            # Calculate rate
            rate = "+0%" if speed == 1.0 else f"{int((speed - 1) * 100):+d}%"
            
            # Add short prefix to prevent Edge TTS from cutting off beginning of audio
            # This is a known Edge TTS issue
            prefix = "Hello. "
            text = prefix + text
            
            # Create communicate instance
            communicate = edge_tts.Communicate(text, voice_id, rate=rate)
            
            # Save to file
            start_time = time.time()
            await communicate.save(output_file)
            gen_time = time.time() - start_time
            print(f"   [Edge TTS] Generated in {gen_time:.2f}s")
            
            # Convert MP3 to WAV for playback compatibility
            wav_file = output_file.replace('.mp3', '.wav')
            audio_process = None
            try:
                conv_start = time.time()
                subprocess.run([
                    "ffmpeg", "-y", "-i", output_file,
                    "-ar", "24000", "-ac", "1", "-sample_fmt", "s16",
                    wav_file
                ], check=False, capture_output=True, timeout=30)
                conv_time = time.time() - conv_start
                print(f"   [AUDIO DEBUG] Edge TTS: Converted to WAV in {conv_time:.2f}s")

                # Try playing MP3 directly first (avoids conversion issues)
                time.sleep(0.3)
                audio_process = self._play_audio(output_file)
            except Exception as e:
                print(f"   [AUDIO DEBUG] Edge TTS: Error converting/playing: {e}")
                # Try playing MP3 directly
                audio_process = self._play_audio(output_file)

            return {
                "success": True,
                "text": text,
                "voice": voice_id,
                "output_file": output_file,
                "audio_process": audio_process,
                "audio_file": wav_file if os.path.exists(wav_file) else output_file
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "edge-tts not installed. Run: pip install edge-tts"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Edge TTS failed: {str(e)}"
            }
