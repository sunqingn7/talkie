"""
Speech-to-Text Tool - Captures and transcribes audio using whisper.cpp.
"""

import asyncio
import json
import wave
import tempfile
import os
from typing import Any, Dict
import requests

from . import BaseTool

# Try to import pyaudio, but make it optional
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠️  PyAudio not available. Voice input will use file upload instead.")


class STTTool(BaseTool):
    """Tool to capture and transcribe speech from microphone using whisper.cpp."""
    
    def _get_description(self) -> str:
        if PYAUDIO_AVAILABLE:
            return (
                "Listen to user's voice input and convert it to text. "
                "Uses whisper.cpp server for offline speech recognition. "
                "Records audio from microphone until silence is detected or timeout."
            )
        else:
            return (
                "Transcribe audio file to text using whisper.cpp. "
                "Provide the path to a WAV audio file to transcribe."
            )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        if PYAUDIO_AVAILABLE:
            return {
                "type": "object",
                "properties": {
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum recording time in seconds",
                        "default": 10
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (e.g., 'en', 'auto')",
                        "default": "en"
                    }
                }
            }
        else:
            return {
                "type": "object",
                "properties": {
                    "audio_file": {
                        "type": "string",
                        "description": "Path to WAV audio file to transcribe"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (e.g., 'en', 'auto')",
                        "default": "en"
                    }
                },
                "required": ["audio_file"]
            }
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.audio_config = config["audio"]
        self.whisper_config = config["stt"]["whisper"]
        self.server_url = self.whisper_config.get("server_url", "http://localhost:8081")
        
    def _check_server(self) -> bool:
        """Check if whisper.cpp server is running."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Record audio and transcribe to text using whisper.cpp."""
        if not self._check_server():
            return {
                "success": False,
                "error": f"whisper.cpp server not running at {self.server_url}. Please start it first.",
                "text": None
            }
        
        try:
            if PYAUDIO_AVAILABLE:
                # Use microphone recording
                timeout = kwargs.get('timeout', 10)
                language = kwargs.get('language', 'en')
                wav_file = await self._record_audio(timeout)
            else:
                # Use provided audio file
                audio_file = kwargs.get('audio_file')
                language = kwargs.get('language', 'en')
                if not audio_file or not os.path.exists(audio_file):
                    return {
                        "success": False,
                        "error": f"Audio file not found: {audio_file}",
                        "text": None
                    }
                wav_file = audio_file
            
            # Send to whisper.cpp server
            text = await self._transcribe_audio(wav_file, language)
            
            # Cleanup (only if we recorded it)
            if PYAUDIO_AVAILABLE and wav_file:
                os.unlink(wav_file)
            
            return {
                "success": True,
                "text": text,
                "language": language
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": None
            }
    
    async def _record_audio(self, timeout: int) -> str:
        """Record audio from microphone to WAV file."""
        if not PYAUDIO_AVAILABLE:
            raise Exception("PyAudio not available. Cannot record from microphone.")
        
        # Suppress ALSA warnings by redirecting stderr
        import sys
        import os
        from io import StringIO
        
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
        try:
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.audio_config["channels"],
                rate=self.audio_config["sample_rate"],
                input=True,
                frames_per_buffer=self.audio_config["chunk_size"]
            )
            
            print("🎤 Listening... (speak now)")
            
            frames = []
            
            import time
            start_time = time.time()
            
            while True:
                data = stream.read(self.audio_config["chunk_size"], exception_on_overflow=False)
                frames.append(data)
                
                # Check timeout
                if time.time() - start_time > timeout:
                    break
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                wav_file = tmpfile.name
            
            wf = wave.open(wav_file, 'wb')
            wf.setnchannels(self.audio_config["channels"])
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.audio_config["sample_rate"])
            wf.writeframes(b''.join(frames))
            wf.close()
            
            print("✅ Recording complete")
            
            return wav_file
        finally:
            # Restore stderr
            sys.stderr.close()
            sys.stderr = old_stderr
    
    async def _transcribe_audio(self, wav_file: str, language: str) -> str:
        """Send audio to whisper.cpp server for transcription."""
        with open(wav_file, 'rb') as f:
            files = {'file': ('audio.wav', f, 'audio/wav')}
            data = {'language': language}
            
            response = requests.post(
                f"{self.server_url}/inference",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code != 200:
            raise Exception(f"Transcription failed: {response.text}")
        
        result = response.json()
        text = result.get("text", "").strip()
        
        print(f"📝 Transcribed: '{text}'")
        
        return text if text else "(no speech detected)"
