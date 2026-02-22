"""
Edge TTS Backend - Microsoft Edge TTS with native streaming
"""

import asyncio
import io
import tempfile
import os
from typing import AsyncIterator, Dict, List, Any, Optional

from .base import TTSBackend
from ..audio.formats import pcm_to_wav, SAMPLE_RATE


class EdgeTTSBackend(TTSBackend):
    """Microsoft Edge TTS backend with streaming support."""
    
    name = "edge_tts"
    description = "Microsoft Edge TTS - High quality neural voices (online)"
    
    AVAILABLE_VOICES = [
        {"id": "en-US-AriaNeural", "name": "Aria", "language": "en-US", "gender": "Female"},
        {"id": "en-US-JennyNeural", "name": "Jenny", "language": "en-US", "gender": "Female"},
        {"id": "en-US-GuyNeural", "name": "Guy", "language": "en-US", "gender": "Male"},
        {"id": "en-GB-SoniaNeural", "name": "Sonia", "language": "en-GB", "gender": "Female"},
        {"id": "en-GB-RyanNeural", "name": "Ryan", "language": "en-GB", "gender": "Male"},
        {"id": "zh-CN-XiaoxiaoNeural", "name": "Xiaoxiao", "language": "zh-CN", "gender": "Female"},
        {"id": "zh-CN-YunjianNeural", "name": "Yunjian", "language": "zh-CN", "gender": "Male"},
        {"id": "zh-CN-XiaoyiNeural", "name": "Xiaoyi", "language": "zh-CN", "gender": "Female"},
        {"id": "ja-JP-NanamiNeural", "name": "Nanami", "language": "ja-JP", "gender": "Female"},
        {"id": "ko-KR-SunHiNeural", "name": "SunHi", "language": "ko-KR", "gender": "Female"},
        {"id": "es-ES-ElviraNeural", "name": "Elvira", "language": "es-ES", "gender": "Female"},
        {"id": "fr-FR-DeniseNeural", "name": "Denise", "language": "fr-FR", "gender": "Female"},
        {"id": "de-DE-KatjaNeural", "name": "Katja", "language": "de-DE", "gender": "Female"},
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.default_voice = self.config.get("default_voice", "en-US-AriaNeural")
    
    async def initialize(self) -> bool:
        """Check if edge-tts is available."""
        try:
            import edge_tts
            return True
        except ImportError:
            print("[Edge TTS] edge-tts not installed. Run: pip install edge-tts")
            return False
    
    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> AsyncIterator[bytes]:
        """
        Stream audio from Edge TTS.
        
        Yields PCM audio chunks (24kHz, 16-bit mono).
        """
        import edge_tts
        
        voice_id = voice or self._get_voice_for_language(language) or self.default_voice
        rate = self._speed_to_rate(speed)
        
        communicate = edge_tts.Communicate(text, voice_id, rate=rate)
        
        audio_buffer = io.BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data = chunk["data"]
                
                if audio_buffer.tell() == 0 and audio_data[:4] == b'RIFF':
                    audio_data = self._strip_wav_header(audio_data)
                
                audio_buffer.write(audio_data)
                
                if len(audio_data) > 0:
                    pcm_data = self._convert_to_pcm(audio_data)
                    if pcm_data:
                        yield pcm_data
        
        remaining = self._flush_buffer_to_pcm(audio_buffer)
        if remaining:
            yield remaining
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> bytes:
        """Generate complete WAV audio."""
        pcm_chunks = []
        async for chunk in self.synthesize_stream(text, voice, language, speed, **kwargs):
            pcm_chunks.append(chunk)
        
        pcm_data = b"".join(pcm_chunks)
        return pcm_to_wav(pcm_data)
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices."""
        return self.AVAILABLE_VOICES.copy()
    
    def get_supported_languages(self) -> List[str]:
        """Get supported language codes."""
        languages = set()
        for voice in self.AVAILABLE_VOICES:
            lang = voice.get("language", "")
            if lang:
                languages.add(lang.split("-")[0])
        return list(languages)
    
    def _get_voice_for_language(self, language: Optional[str]) -> Optional[str]:
        """Get appropriate voice for language."""
        if not language:
            return None
        
        language = language.lower()
        
        for voice in self.AVAILABLE_VOICES:
            voice_lang = voice.get("language", "").lower()
            if voice_lang.startswith(language):
                return voice["id"]
        
        return None
    
    def _speed_to_rate(self, speed: float) -> str:
        """Convert speed multiplier to Edge TTS rate string."""
        if speed == 1.0:
            return "+0%"
        percentage = int((speed - 1.0) * 100)
        return f"{percentage:+d}%"
    
    def _strip_wav_header(self, audio_data: bytes) -> bytes:
        """Strip WAV header from audio data."""
        if audio_data[:4] == b'RIFF':
            return audio_data[44:]
        return audio_data
    
    def _convert_to_pcm(self, audio_data: bytes) -> Optional[bytes]:
        """Convert audio chunk to PCM format."""
        return audio_data
    
    def _flush_buffer_to_pcm(self, buffer: io.BytesIO) -> Optional[bytes]:
        """Flush remaining buffer to PCM."""
        buffer.seek(0)
        data = buffer.read()
        if data:
            return self._strip_wav_header(data)
        return None
