"""
Abstract base class for TTS backends
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Any, Optional, Union


class TTSBackend(ABC):
    """Abstract base class for TTS backend implementations."""
    
    name: str = "base"
    description: str = "Base TTS backend"
    
    @abstractmethod
    async def synthesize_stream(
        self, 
        text: str, 
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> AsyncIterator[bytes]:
        """
        Synthesize text to audio and yield chunks.
        
        Args:
            text: Text to synthesize
            voice: Voice/speaker ID
            language: Language code
            speed: Speech speed multiplier
            **kwargs: Backend-specific parameters
            
        Yields:
            Audio chunks (PCM format, 24kHz, 16-bit mono)
        """
        pass
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0,
        **kwargs
    ) -> bytes:
        """
        Synthesize text to complete audio.
        
        Args:
            text: Text to synthesize
            voice: Voice/speaker ID
            language: Language code
            speed: Speech speed multiplier
            **kwargs: Backend-specific parameters
            
        Returns:
            Complete audio data (WAV format)
        """
        pass
    
    @abstractmethod
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available voices.
        
        Returns:
            List of voice info dicts with 'id', 'name', 'language', etc.
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of language codes (e.g., ['en', 'zh', 'ja'])
        """
        pass
    
    async def initialize(self) -> bool:
        """
        Initialize the backend (load models, etc.).
        
        Returns:
            True if initialization successful
        """
        return True
    
    async def shutdown(self) -> None:
        """Clean up resources."""
        pass
    
    def get_default_voice(self, language: str = "en") -> Optional[str]:
        """Get default voice for a language."""
        voices = self.get_available_voices()
        for voice in voices:
            if voice.get("language", "").startswith(language):
                return voice.get("id")
        return voices[0].get("id") if voices else None
