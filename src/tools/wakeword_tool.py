"""
Wake word detection tool for Talkie Voice Assistant.
Detects phrases like "Hey Talkie" to activate the assistant.
"""

import asyncio
import re
from typing import Any, Dict
from tools import BaseTool


class WakeWordTool(BaseTool):
    """
    Wake word detection tool.
    Monitors audio input for wake phrases like "Hey Talkie".
    Can work with audio files, microphone (when available), or text input.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.wake_phrases = config.get("wake_word", {}).get("phrases", ["hey talkie", "ok talkie", "talkie"])
        self.simulation_mode = config.get("wake_word", {}).get("simulation_mode", True)
        self.audio_buffer = []
        
    def _get_description(self) -> str:
        return """
        Wake word detection system. Listens for activation phrases like 'Hey Talkie'.
        
        Usage modes:
        1. Text mode: Pass text to check if it contains wake phrases
        2. Audio file mode: Process audio file to detect wake word
        3. Microphone mode: (requires PyAudio) Continuously listen for wake word
        
        Returns: {'detected': bool, 'confidence': float, 'phrase': str}
        """
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["check_text", "process_audio", "listen_continuous"],
                    "description": "Detection mode to use"
                },
                "text": {
                    "type": "string",
                    "description": "Text to check for wake phrases (mode: check_text)"
                },
                "audio_file": {
                    "type": "string",
                    "description": "Path to audio file (mode: process_audio)"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Listening timeout in seconds (mode: listen_continuous)",
                    "default": 30
                }
            },
            "required": ["mode"]
        }
    
    async def execute(self, mode: str, text: str = None, audio_file: str = None, 
                     timeout: int = 30) -> Dict[str, Any]:
        """Execute wake word detection based on mode."""
        
        if mode == "check_text":
            return await self._check_text(text or "")
        
        elif mode == "process_audio":
            return await self._process_audio_file(audio_file)
        
        elif mode == "listen_continuous":
            return await self._listen_continuous(timeout)
        
        else:
            return {"error": f"Unknown mode: {mode}"}
    
    async def _check_text(self, text: str) -> Dict[str, Any]:
        """Check if text contains any wake phrase."""
        text_lower = text.lower().strip()
        
        for phrase in self.wake_phrases:
            # Check for exact match or phrase at start
            if text_lower.startswith(phrase) or phrase in text_lower:
                # Calculate confidence based on match quality
                confidence = 1.0 if text_lower.startswith(phrase) else 0.8
                
                return {
                    "detected": True,
                    "confidence": confidence,
                    "phrase": phrase,
                    "full_text": text,
                    "remaining_text": text_lower.replace(phrase, "").strip()
                }
        
        return {
            "detected": False,
            "confidence": 0.0,
            "phrase": None,
            "full_text": text
        }
    
    async def _process_audio_file(self, audio_file: str) -> Dict[str, Any]:
        """Process audio file to detect wake word."""
        if not audio_file:
            return {"error": "No audio file provided"}
        
        try:
            # Simulate STT processing (would use actual STT in production)
            # For now, return a simulated result
            return {
                "detected": False,
                "confidence": 0.0,
                "note": "Audio file processing requires STT integration",
                "file": audio_file,
                "suggestion": "Use text mode for testing or install PyAudio for microphone support"
            }
        except Exception as e:
            return {"error": f"Failed to process audio: {str(e)}"}
    
    async def _listen_continuous(self, timeout: int) -> Dict[str, Any]:
        """Continuously listen for wake word (requires PyAudio)."""
        try:
            # Try to import PyAudio
            import pyaudio
            import wave
            
            # This would implement actual microphone listening
            return {
                "detected": False,
                "confidence": 0.0,
                "note": "Microphone listening mode",
                "timeout": timeout,
                "status": "Not fully implemented - requires PyAudio configuration"
            }
            
        except ImportError:
            return {
                "detected": False,
                "confidence": 0.0,
                "error": "PyAudio not installed",
                "installation": "Run: sudo apt-get install portaudio19-dev && pip install pyaudio",
                "alternative": "Use mode='check_text' for testing without microphone"
            }
    
    def add_wake_phrase(self, phrase: str):
        """Add a new wake phrase."""
        phrase = phrase.lower().strip()
        if phrase not in self.wake_phrases:
            self.wake_phrases.append(phrase)
            return True
        return False
    
    def remove_wake_phrase(self, phrase: str):
        """Remove a wake phrase."""
        phrase = phrase.lower().strip()
        if phrase in self.wake_phrases:
            self.wake_phrases.remove(phrase)
            return True
        return False
    
    def get_wake_phrases(self) -> list:
        """Get list of configured wake phrases."""
        return self.wake_phrases.copy()


class VoiceActivityTool(BaseTool):
    """
    Voice Activity Detection (VAD) tool.
    Detects when someone is speaking.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.energy_threshold = config.get("vad", {}).get("energy_threshold", 300)
        
    def _get_description(self) -> str:
        return """
        Voice Activity Detection (VAD) tool.
        Detects when speech is present in audio input.
        Useful for knowing when to start recording user commands.
        
        Returns: {'active': bool, 'energy_level': float}
        """
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["check_file", "listen_continuous"],
                    "description": "Detection mode"
                },
                "audio_file": {
                    "type": "string",
                    "description": "Path to audio file to analyze"
                },
                "duration": {
                    "type": "integer",
                    "description": "Duration to listen in seconds",
                    "default": 5
                }
            },
            "required": ["mode"]
        }
    
    async def execute(self, mode: str, audio_file: str = None, duration: int = 5) -> Dict[str, Any]:
        """Execute VAD based on mode."""
        
        if mode == "check_file":
            return await self._analyze_file(audio_file)
        
        elif mode == "listen_continuous":
            return await self._listen(duration)
        
        return {"error": f"Unknown mode: {mode}"}
    
    async def _analyze_file(self, audio_file: str) -> Dict[str, Any]:
        """Analyze audio file for voice activity."""
        if not audio_file:
            return {"error": "No audio file provided"}
        
        # Placeholder for actual VAD implementation
        return {
            "active": True,
            "energy_level": 0.0,
            "note": "Audio analysis not fully implemented",
            "file": audio_file
        }
    
    async def _listen(self, duration: int) -> Dict[str, Any]:
        """Listen for voice activity."""
        try:
            import pyaudio
            
            return {
                "active": False,
                "energy_level": 0.0,
                "note": "Microphone VAD requires PyAudio",
                "installation": "Run: sudo apt-get install portaudio19-dev && pip install pyaudio"
            }
            
        except ImportError:
            return {
                "active": False,
                "energy_level": 0.0,
                "error": "PyAudio not installed",
                "installation": "Run: sudo apt-get install portaudio19-dev && pip install pyaudio"
            }
