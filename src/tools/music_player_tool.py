"""
Music player tool for playing audio from online sources.
Uses mpv --no-video to play audio streams.
Requires: mpv (sudo apt install mpv)
"""

import json
import os
import random
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from tools import BaseTool


def check_mpv_available() -> bool:
    """Check if mpv is installed on the system."""
    return shutil.which("mpv") is not None


def get_music_history_path() -> str:
    """Get the path to the music history file."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)
    return str(config_dir / "music_history.json")


def load_music_history() -> List[dict]:
    """Load music play history from file."""
    history_path = get_music_history_path()
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_music_history(history: List[dict]):
    """Save music play history to file."""
    history_path = get_music_history_path()
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


class MusicPlayerTool(BaseTool):
    """
    Play music from online sources (URLs).
    Supports direct audio URLs, YouTube links, and other streaming sources.
    Uses mpv --no-video for audio playback.
    Maintains history of played music.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.current_process: Optional[subprocess.Popen] = None
        self.is_playing = False
        self.mpv_available = check_mpv_available()
        self._history_cache: Optional[List[dict]] = None
        if not self.mpv_available:
            print("[MusicPlayerTool] WARNING: mpv not found. Music playback will not work.")
            print("[MusicPlayerTool] Install with: sudo apt install mpv")
    
    @property
    def history(self) -> List[dict]:
        """Lazy load history."""
        if self._history_cache is None:
            self._history_cache = load_music_history()
        return self._history_cache
    
    @history.setter
    def history(self, value: List[dict]):
        """Save history when updated."""
        self._history_cache = value
        save_music_history(value)
        
    def _get_description(self) -> str:
        return """
        Play music from a URL. Supports direct audio links, YouTube, and other streaming sources.
        
        Examples:
        - "Play https://www.example.com/song.mp3"
        - "Play music from https://www.youtube.com/watch?v=..."
        - "Play https://soundcloud.com/..."
        
        Returns: Playback status and details.
        """
    
    def _get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["play", "stop", "pause", "resume", "status", "list"],
                    "description": "Action to perform: play, stop, pause, resume, status, or list"
                },
                "url": {
                    "type": "string",
                    "description": "URL of the audio source to play (optional for play - will pick random from history if not provided)"
                }
            },
            "required": ["action"]
        }
    
    async def execute(self, action: str, url: str = None) -> Dict[str, Any]:
        """Execute music player action."""
        
        if action == "play":
            return await self._play(url)
        elif action == "stop":
            return await self._stop()
        elif action == "pause":
            return await self._pause()
        elif action == "resume":
            return await self._resume()
        elif action == "status":
            return await self._status()
        elif action == "list":
            return await self._list_history()
        else:
            return {"error": f"Unknown action: {action}"}
    
    async def _play(self, url: str = None) -> Dict[str, Any]:
        """Play audio from URL."""
        if not self.mpv_available:
            return {"error": "mpv is not installed. Please install with: sudo apt install mpv"}
        
        # If no URL provided, pick random from history
        if not url:
            if not self.history:
                return {"error": "No music in history. Please provide a URL to play."}
            url = random.choice(self.history)["url"]
        
        # Stop any currently playing audio
        if self.is_playing:
            self._stop_process()
        
        try:
            # Use mpv --no-video to play audio only
            self.current_process = subprocess.Popen(
                ["mpv", "--no-video", "--quiet", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.is_playing = True
            
            # Add to history
            self._add_to_history(url)
            
            # Start a thread to monitor when playback ends
            monitor_thread = threading.Thread(
                target=self._monitor_playback,
                daemon=True
            )
            monitor_thread.start()
            
            return {
                "success": True,
                "action": "play",
                "url": url,
                "status": "playing",
                "message": f"Playing audio from: {url}"
            }
            
        except FileNotFoundError:
            return {"error": "mpv not found. Please install mpv: sudo apt install mpv"}
        except Exception as e:
            self.is_playing = False
            return {"error": f"Failed to play audio: {str(e)}"}
    
    def _add_to_history(self, url: str):
        """Add a URL to the play history."""
        # Check if URL already exists (move to front if so)
        for item in self.history:
            if item["url"] == url:
                self.history.remove(item)
                break
        
        # Add to beginning of history
        self.history.insert(0, {
            "url": url,
            "played_at": datetime.now().isoformat()
        })
        
        # Keep only last 100 entries
        if len(self.history) > 100:
            self.history = self.history[:100]
    
    def _monitor_playback(self):
        """Monitor playback and update status when done."""
        if self.current_process:
            self.current_process.wait()
            self.is_playing = False
            self.current_process = None
    
    def _stop_process(self):
        """Stop the current mpv process."""
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=2)
            except:
                try:
                    self.current_process.kill()
                except:
                    pass
            self.current_process = None
        self.is_playing = False
    
    async def _stop(self) -> Dict[str, Any]:
        """Stop playback."""
        if not self.is_playing:
            return {
                "success": True,
                "action": "stop",
                "status": "stopped",
                "message": "No audio playing"
            }
        
        self._stop_process()
        
        return {
            "success": True,
            "action": "stop",
            "status": "stopped",
            "message": "Stopped playback"
        }
    
    async def _pause(self) -> Dict[str, Any]:
        """Pause playback (sends pause command to mpv)."""
        if not self.is_playing or not self.current_process:
            return {"error": "No audio playing"}
        
        try:
            # Send pause command via IPC (mpv doesn't have simple pause via signals)
            # For simplicity, we'll just stop for now
            return await self._stop()
        except Exception as e:
            return {"error": f"Failed to pause: {str(e)}"}
    
    async def _resume(self) -> Dict[str, Any]:
        """Resume playback - not supported in this simple implementation."""
        return {"error": "Resume not supported. Please use 'play' to start a new track."}
    
    async def _status(self) -> Dict[str, Any]:
        """Get playback status."""
        return {
            "success": True,
            "status": "playing" if self.is_playing else "stopped",
            "is_playing": self.is_playing
        }
    
    async def _list_history(self) -> Dict[str, Any]:
        """List music play history."""
        return {
            "success": True,
            "history": self.history,
            "count": len(self.history)
        }
