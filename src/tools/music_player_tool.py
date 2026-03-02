"""
Music player tool for playing audio from online sources.
Uses mpv --no-video to play audio streams (local mode) or streams to web browser (web mode).
Requires: mpv (sudo apt install mpv) for local mode
"""

import asyncio
import json
import os
import random
import shutil
import subprocess
import threading
import time
import uuid
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
            with open(history_path, "r") as f:
                return json.load(f)
        except:
            return []
    return []


def save_music_history(history: List[dict]):
    """Save music play history to file."""
    history_path = get_music_history_path()
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


class MusicPlayerTool(BaseTool):
    """
    Play music from online sources (URLs).
    Supports direct audio URLs, YouTube links, and other streaming sources.
    Can play locally via mpv or stream to web browser.
    Maintains history of played music.
    """

    MAX_CONCURRENT_STREAMS = 3
    STREAM_CACHE_EXPIRY = 600  # 10 minutes

    def __init__(self, config: dict, web_interface=None):
        super().__init__(config)
        self.web_interface = web_interface
        self.current_process: Optional[subprocess.Popen] = None
        self.is_playing = False
        self.current_url = None
        self.mpv_available = check_mpv_available()
        self._history_cache: Optional[List[dict]] = None

        # Web streaming support
        self.music_output = config.get("music", {}).get("output", "local")
        self.stream_cache: Dict[str, Dict] = {}  # token -> {url, expires, title}
        self.stream_token = None
        self._cleanup_thread = None
        self._start_cleanup_thread()

        if not self.mpv_available and self.music_output == "local":
            print(
                "[MusicPlayerTool] WARNING: mpv not found. Local music playback will not work."
            )
            print("[MusicPlayerTool] Install with: sudo apt install mpv")

    def set_web_interface(self, web_interface):
        """Set web interface for audio streaming."""
        self.web_interface = web_interface
        print(f"[MusicPlayerTool] Web interface set: {web_interface is not None}")

    def _start_cleanup_thread(self):
        """Start background thread to cleanup expired streams."""
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        print("[MusicPlayerTool] Stream cleanup thread started")

    def _cleanup_loop(self):
        """Run every 60 seconds to cleanup expired streams."""
        while True:
            time.sleep(60)
            self._cleanup_expired_streams()

    def _cleanup_expired_streams(self):
        """Remove expired stream tokens."""
        now = time.time()
        expired = [t for t, data in self.stream_cache.items() if now > data["expires"]]
        for token in expired:
            print(f"[MusicPlayerTool] Cleaning up expired stream: {token}")
            del self.stream_cache[token]

    def _can_start_stream(self) -> bool:
        """Check if we can start a new stream."""
        active_streams = sum(
            1 for data in self.stream_cache.values() if time.time() < data["expires"]
        )
        return active_streams < self.MAX_CONCURRENT_STREAMS

    def _cleanup_oldest_stream(self):
        """Remove the oldest stream to make room for a new one."""
        if not self.stream_cache:
            return

        oldest_token = min(
            self.stream_cache.keys(),
            key=lambda t: self.stream_cache[t].get("expires", 0),
        )
        print(f"[MusicPlayerTool] Cleaning up oldest stream: {oldest_token}")
        del self.stream_cache[oldest_token]

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
                    "description": "Action to perform: play, stop, pause, resume, status, or list",
                },
                "url": {
                    "type": "string",
                    "description": "URL of the audio source to play (optional for play - will pick random from history if not provided)",
                },
            },
            "required": ["action"],
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

    def _is_direct_audio_url(self, url: str) -> bool:
        """Check if URL is a direct audio file."""
        audio_extensions = [".mp3", ".m4a", ".wav", ".webm", ".ogg", ".aac", ".flac"]
        return any(url.lower().endswith(ext) for ext in audio_extensions)

    async def _play(self, url: str = None) -> Dict[str, Any]:
        """Play audio from URL."""
        # If no URL provided, pick random from history
        if not url:
            if not self.history:
                return {"error": "No music in history. Please provide a URL to play."}
            url = random.choice(self.history)["url"]

        # Route to web or local playback based on config
        if self.music_output == "web" and self.web_interface:
            if self._is_direct_audio_url(url):
                return await self._play_web_direct(url)
            else:
                return await self._play_web_stream(url)
        else:
            return await self._play_local(url)

    async def _play_local(self, url: str) -> Dict[str, Any]:
        """Play audio locally using mpv."""
        if not self.mpv_available:
            return {
                "error": "mpv is not installed. Please install with: sudo apt install mpv"
            }

        # Stop any currently playing audio
        if self.is_playing:
            self._stop_process()

        try:
            # Use mpv --no-video to play audio only
            self.current_process = subprocess.Popen(
                ["mpv", "--no-video", "--quiet", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.is_playing = True
            self.current_url = url

            # Add to history
            self._add_to_history(url)

            # Start a thread to monitor when playback ends
            monitor_thread = threading.Thread(
                target=self._monitor_playback, daemon=True
            )
            monitor_thread.start()

            return {
                "success": True,
                "action": "play",
                "url": url,
                "status": "playing",
                "mode": "local",
                "message": f"Playing audio from: {url}",
            }

        except FileNotFoundError:
            return {"error": "mpv not found. Please install mpv: sudo apt install mpv"}
        except Exception as e:
            self.is_playing = False
            return {"error": f"Failed to play audio: {str(e)}"}

    async def _play_web_direct(self, url: str) -> Dict[str, Any]:
        """Send direct audio URL to frontend for browser playback."""
        if not self.web_interface:
            return {"error": "Web interface not available"}

        self.current_url = url
        self.is_playing = True
        self._add_to_history(url)

        try:
            await self.web_interface.manager.broadcast(
                {
                    "type": "music_control",
                    "action": "play",
                    "url": url,
                    "mode": "direct",
                }
            )
        except Exception as e:
            print(f"[MusicPlayerTool] Error broadcasting music_control: {e}")
            return {"error": f"Failed to send to web interface: {str(e)}"}

        return {
            "success": True,
            "action": "play",
            "url": url,
            "status": "playing",
            "mode": "direct",
        }

    async def _play_web_stream(self, url: str) -> Dict[str, Any]:
        """Start HTTP streaming for complex URLs (YouTube, etc.)."""
        if not self.web_interface:
            return {"error": "Web interface not available"}

        # Check if we can start a new stream
        if not self._can_start_stream():
            print("[MusicPlayerTool] Max streams reached, cleaning up oldest")
            self._cleanup_oldest_stream()

        # Generate unique stream token
        self.stream_token = str(uuid.uuid4())
        self.current_url = url
        self.is_playing = True
        self._add_to_history(url)

        # Start background extraction task
        asyncio.create_task(self._extract_and_cache_url(url))

        # Tell frontend to play stream
        stream_url = f"/api/music/stream?token={self.stream_token}"

        try:
            await self.web_interface.manager.broadcast(
                {
                    "type": "music_control",
                    "action": "play",
                    "url": stream_url,
                    "mode": "stream",
                }
            )
        except Exception as e:
            print(f"[MusicPlayerTool] Error broadcasting music_control: {e}")
            return {"error": f"Failed to send to web interface: {str(e)}"}

        return {
            "success": True,
            "action": "play",
            "url": url,
            "status": "playing",
            "mode": "stream",
            "stream_token": self.stream_token,
        }

    async def _extract_and_cache_url(self, url: str):
        """Extract direct URL via yt-dlp and cache it."""
        try:
            result = subprocess.run(
                ["yt-dlp", "-J", "-f", "bestaudio", url],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                print(f"[MusicPlayerTool] yt-dlp failed: {result.stderr}")
                return

            info = json.loads(result.stdout)
            formats = info.get("formats", [])

            # Find best audio-only format
            audio_formats = [
                f
                for f in formats
                if f.get("vcodec") == "none" and f.get("acodec") != "none"
            ]

            if not audio_formats:
                print("[MusicPlayerTool] No audio formats found")
                return

            best = max(audio_formats, key=lambda x: x.get("abr", 0))
            direct_url = best.get("url")

            if not direct_url:
                print("[MusicPlayerTool] No direct URL found")
                return

            # Cache with expiry
            self.stream_cache[self.stream_token] = {
                "url": direct_url,
                "expires": time.time() + self.STREAM_CACHE_EXPIRY,
                "title": info.get("title", "Unknown"),
            }

            print(
                f"[MusicPlayerTool] Cached stream {self.stream_token}: {info.get('title', 'Unknown')}"
            )

        except subprocess.TimeoutExpired:
            print("[MusicPlayerTool] yt-dlp timeout")
        except json.JSONDecodeError:
            print("[MusicPlayerTool] Failed to parse yt-dlp output")
        except Exception as e:
            print(f"[MusicPlayerTool] Extraction failed: {e}")

    def _add_to_history(self, url: str):
        """Add a URL to the play history."""
        # Check if URL already exists (move to front if so)
        for item in self.history:
            if item["url"] == url:
                self.history.remove(item)
                break

        # Add to beginning of history
        self.history.insert(0, {"url": url, "played_at": datetime.now().isoformat()})

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
        if self.music_output == "web" and self.web_interface and self.is_playing:
            return await self._stop_web()
        else:
            return await self._stop_local()

    async def _stop_local(self) -> Dict[str, Any]:
        """Stop local playback."""
        if not self.is_playing:
            return {
                "success": True,
                "action": "stop",
                "status": "stopped",
                "message": "No audio playing",
            }

        self._stop_process()

        return {
            "success": True,
            "action": "stop",
            "status": "stopped",
            "message": "Stopped playback",
        }

    async def _stop_web(self) -> Dict[str, Any]:
        """Stop web playback."""
        if not self.web_interface:
            return {"error": "Web interface not available"}

        try:
            await self.web_interface.manager.broadcast(
                {"type": "music_control", "action": "stop"}
            )
        except Exception as e:
            print(f"[MusicPlayerTool] Error broadcasting stop: {e}")

        self.is_playing = False
        self.current_url = None

        # Cleanup stream cache if applicable
        if self.stream_token and self.stream_token in self.stream_cache:
            del self.stream_cache[self.stream_token]
            self.stream_token = None

        return {
            "success": True,
            "action": "stop",
            "status": "stopped",
            "message": "Stopped playback",
        }

    async def _pause(self) -> Dict[str, Any]:
        """Pause playback."""
        if not self.is_playing:
            return {"error": "No audio playing"}

        if self.music_output == "web" and self.web_interface:
            # For web mode, send pause command to frontend
            try:
                await self.web_interface.manager.broadcast(
                    {"type": "music_control", "action": "pause"}
                )
            except Exception as e:
                print(f"[MusicPlayerTool] Error broadcasting pause: {e}")

            return {"success": True, "action": "pause", "status": "paused"}
        else:
            # For local mode, just stop (mpv pause is complex)
            return await self._stop()

    async def _resume(self) -> Dict[str, Any]:
        """Resume playback."""
        if self.music_output == "web" and self.web_interface:
            # For web mode, send resume command to frontend
            try:
                await self.web_interface.manager.broadcast(
                    {"type": "music_control", "action": "resume"}
                )
            except Exception as e:
                print(f"[MusicPlayerTool] Error broadcasting resume: {e}")

            return {"success": True, "action": "resume", "status": "playing"}
        else:
            return {
                "error": "Resume not supported in local mode. Please use 'play' to start a new track."
            }

    async def _status(self) -> Dict[str, Any]:
        """Get playback status."""
        return {
            "success": True,
            "status": "playing" if self.is_playing else "stopped",
            "is_playing": self.is_playing,
            "mode": self.music_output,
            "current_url": self.current_url,
        }

    async def _list_history(self) -> Dict[str, Any]:
        """List music play history."""
        return {"success": True, "history": self.history, "count": len(self.history)}
