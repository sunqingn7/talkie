"""
TTS Server Client - Manages connection to streaming TTS server
"""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Optional, Dict, Any
from pathlib import Path


class TTSServerClient:
    """Client for the streaming TTS server with auto-start capability."""
    
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 8083
    PROCESS_NAME = "tts_server"
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, auto_start: bool = True):
        self.host = host
        self.port = port
        self.auto_start = auto_start
        self._server_process: Optional[subprocess.Popen] = None
        self._ws = None
        self._connected = False
    
    @property
    def server_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws"
    
    def is_server_running(self) -> bool:
        """Check if TTS server is already running."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def start_server(self, wait: bool = True, timeout: float = 10.0) -> bool:
        """
        Start the TTS server.
        
        Args:
            wait: Wait for server to be ready
            timeout: Max seconds to wait for server startup
        
        Returns:
            True if server started successfully
        """
        if self.is_server_running():
            print(f"[TTS Server Client] Server already running at {self.server_url}")
            return True
        
        # Find TTSServer directory - try multiple approaches
        server_script = None
        server_dir = None
        
        # Approach 1: Relative to this file (src/tools -> TTSServer)
        try:
            dir1 = Path(__file__).resolve().parent.parent.parent / "TTSServer"
            script1 = dir1 / "server.py"
            if script1.exists():
                server_dir = dir1
                server_script = script1
        except Exception:
            pass
        
        # Approach 2: From current working directory
        if server_script is None:
            try:
                dir2 = Path.cwd() / "TTSServer"
                script2 = dir2 / "server.py"
                if script2.exists():
                    server_dir = dir2
                    server_script = script2
            except Exception:
                pass
        
        # Approach 3: Look for talkie project root
        if server_script is None:
            try:
                current = Path(__file__).resolve()
                for parent in current.parents:
                    candidate = parent / "TTSServer" / "server.py"
                    if candidate.exists():
                        server_dir = parent / "TTSServer"
                        server_script = candidate
                        break
            except Exception:
                pass
        
        if server_script is None or not server_script.exists():
            print(f"[TTS Server Client] Server script not found. Checked:")
            print(f"  - {Path(__file__).resolve().parent.parent.parent / 'TTSServer' / 'server.py'}")
            print(f"  - {Path.cwd() / 'TTSServer' / 'server.py'}")
            return False
        
        print(f"[TTS Server Client] Starting TTS server at {self.server_url}...")
        
        try:
            # Don't pipe stdout/stderr so we can see TTSServer logs
            self._server_process = subprocess.Popen(
                [sys.executable, str(server_script), "--host", self.host, "--port", str(self.port)],
                cwd=str(server_dir),
                start_new_session=True
            )
            
            if wait:
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if self.is_server_running():
                        print(f"[TTS Server Client] Server started successfully (PID: {self._server_process.pid})")
                        return True
                    time.sleep(0.5)
                
                # Server process started but not ready yet - it's loading in background
                print(f"[TTS Server Client] Server process started (PID: {self._server_process.pid}), loading model in background...")
                return True  # Return True because process IS running
            
            return True
            
        except Exception as e:
            print(f"[TTS Server Client] Failed to start server: {e}")
            return False
    
    def stop_server(self) -> bool:
        """Stop the TTS server if we started it."""
        if self._server_process is None:
            return True
        
        try:
            if self._server_process.poll() is None:
                print(f"[TTS Server Client] Stopping TTS server (PID: {self._server_process.pid})...")
                
                try:
                    os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
                except Exception:
                    self._server_process.terminate()
                
                try:
                    self._server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(self._server_process.pid), signal.SIGKILL)
                    except Exception:
                        self._server_process.kill()
                
                print("[TTS Server Client] Server stopped")
            
            self._server_process = None
            return True
            
        except Exception as e:
            print(f"[TTS Server Client] Error stopping server: {e}")
            return False
    
    async def get_available_backends(self) -> Optional[list]:
        """Get list of available TTS backends from server."""
        import aiohttp
        
        if not self.is_server_running():
            if self.auto_start:
                if not self.start_server():
                    return None
            else:
                return None
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/api/backends", timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("backends", [])
        except Exception as e:
            print(f"[TTS Server Client] Error getting backends: {e}")
        
        return None
    
    async def get_voices(self, backend: str = "edge_tts") -> Optional[list]:
        """Get list of available voices for a backend."""
        import aiohttp
        
        if not self.is_server_running():
            if self.auto_start:
                if not self.start_server():
                    return None
            else:
                return None
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/api/voices/{backend}", timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("voices", [])
        except Exception as e:
            print(f"[TTS Server Client] Error getting voices: {e}")
        
        return None
    
    async def synthesize(
        self,
        text: str,
        backend: str = "edge_tts",
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0
    ) -> Optional[bytes]:
        """
        Synthesize text using the TTS server (non-streaming).
        
        Returns complete WAV audio data.
        """
        import aiohttp
        
        if not self.is_server_running():
            if self.auto_start:
                if not self.start_server():
                    return None
            else:
                return None
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.server_url}/api/tts",
                    json={
                        "text": text,
                        "backend": backend,
                        "voice": voice,
                        "language": language,
                        "speed": speed
                    },
                    timeout=60
                ) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    else:
                        error = await resp.text()
                        print(f"[TTS Server Client] Synthesis failed: {error}")
                        return None
                        
        except asyncio.TimeoutError:
            print("[TTS Server Client] Synthesis timed out")
            return None
        except Exception as e:
            print(f"[TTS Server Client] Synthesis error: {e}")
            return None
    
    async def synthesize_stream(
        self,
        text: str,
        backend: str = "edge_tts",
        voice: Optional[str] = None,
        language: Optional[str] = None,
        speed: float = 1.0
    ):
        """
        Stream synthesis from TTS server via WebSocket.
        
        Yields audio chunks (WAV header first, then PCM chunks).
        """
        import websockets
        
        if not self.is_server_running():
            if self.auto_start:
                if not self.start_server():
                    return
            else:
                return
        
        try:
            async with websockets.connect(self.ws_url, timeout=10) as ws:
                await ws.send(json.dumps({
                    "type": "synthesize",
                    "text": text,
                    "backend": backend,
                    "voice": voice,
                    "language": language,
                    "speed": speed
                }))
                
                while True:
                    try:
                        msg = await ws.recv()
                        
                        if isinstance(msg, bytes):
                            yield msg
                        else:
                            data = json.loads(msg)
                            if data.get("type") == "done":
                                break
                            elif data.get("type") == "error":
                                print(f"[TTS Server Client] Server error: {data.get('message')}")
                                break
                            elif data.get("type") == "start":
                                pass
                    except websockets.exceptions.ConnectionClosed:
                        break
                        
        except Exception as e:
            print(f"[TTS Server Client] WebSocket error: {e}")


_tts_server_client: Optional[TTSServerClient] = None


def get_tts_server_client(host: str = "localhost", port: int = 8083, auto_start: bool = True) -> TTSServerClient:
    """Get or create the global TTS server client."""
    global _tts_server_client
    
    if _tts_server_client is None:
        _tts_server_client = TTSServerClient(host=host, port=port, auto_start=auto_start)
    
    return _tts_server_client
