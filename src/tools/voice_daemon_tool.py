"""
Voice Daemon Control Tool - Control the voice daemon queue and status.
"""

from typing import Any, Dict
from . import BaseTool


class VoiceDaemonStatusTool(BaseTool):
    """Tool to get voice daemon status and queue information."""
    
    def __init__(self, config: dict, voice_daemon=None):
        super().__init__(config)
        self.voice_daemon = voice_daemon
    
    def set_voice_daemon(self, voice_daemon):
        """Set the voice daemon reference."""
        self.voice_daemon = voice_daemon
    
    def _get_description(self) -> str:
        return (
            "Get the current status of the Voice Daemon including queue size, "
            "current speech, and statistics."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    async def execute(self) -> Dict[str, Any]:
        """Get voice daemon status."""
        if not self.voice_daemon:
            return {
                "success": False,
                "error": "Voice daemon not available"
            }
        
        return {
            "success": True,
            "status": self.voice_daemon.get_status()
        }


class VoiceDaemonStopTool(BaseTool):
    """Tool to stop current speech and clear the voice daemon queue."""
    
    def __init__(self, config: dict, voice_daemon=None):
        super().__init__(config)
        self.voice_daemon = voice_daemon
    
    def set_voice_daemon(self, voice_daemon):
        """Set the voice daemon reference."""
        self.voice_daemon = voice_daemon
    
    def _get_description(self) -> str:
        return (
            "Stop the current speech and clear the voice daemon queue. "
            "This interrupts any ongoing speech immediately."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    async def execute(self) -> Dict[str, Any]:
        """Stop current speech and clear queue."""
        if not self.voice_daemon:
            return {
                "success": False,
                "error": "Voice daemon not available"
            }
        
        return self.voice_daemon.stop_current()


class VoiceDaemonSkipTool(BaseTool):
    """Tool to skip the current speech in the voice daemon."""
    
    def __init__(self, config: dict, voice_daemon=None):
        super().__init__(config)
        self.voice_daemon = voice_daemon
    
    def set_voice_daemon(self, voice_daemon):
        """Set the voice daemon reference."""
        self.voice_daemon = voice_daemon
    
    def _get_description(self) -> str:
        return (
            "Skip the current speech and move to the next item in the queue. "
            "Useful when you want to skip a long paragraph during file reading."
        )
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    async def execute(self) -> Dict[str, Any]:
        """Skip current speech."""
        if not self.voice_daemon:
            return {
                "success": False,
                "error": "Voice daemon not available"
            }
        
        return self.voice_daemon.skip_current()
