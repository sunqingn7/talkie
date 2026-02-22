"""
TTS Server - FastAPI server with WebSocket streaming support
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Optional, Any

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

from tts_backends import TTSBackend, EdgeTTSBackend, QwenTTSBackend
from audio.formats import create_wav_header


class SynthesizeRequest(BaseModel):
    text: str
    backend: str = "edge_tts"
    voice: Optional[str] = None
    language: Optional[str] = None
    speed: float = 1.0


class TTSServer:
    """TTS Server managing multiple backends."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.backends: Dict[str, TTSBackend] = {}
        self.active_connections: list[WebSocket] = []
        self._init_backends()
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = __file__.replace("server.py", "config.yaml")
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"[TTS Server] Config not found at {config_path}, using defaults")
            return {}
    
    def _init_backends(self):
        """Initialize TTS backends."""
        backends_config = self.config.get("backends", {})
        
        if backends_config.get("edge_tts", {}).get("enabled", True):
            self.backends["edge_tts"] = EdgeTTSBackend(backends_config.get("edge_tts", {}))
            print("[TTS Server] Edge TTS backend registered")
        
        if backends_config.get("qwen_tts", {}).get("enabled", False):
            self.backends["qwen_tts"] = QwenTTSBackend(backends_config.get("qwen_tts", {}))
            print("[TTS Server] Qwen TTS backend registered")
    
    async def initialize(self):
        """Initialize all backends."""
        for name, backend in self.backends.items():
            try:
                success = await backend.initialize()
                if success:
                    print(f"[TTS Server] {name} initialized successfully", flush=True)
                else:
                    print(f"[TTS Server] {name} initialization failed", flush=True)
            except Exception as e:
                print(f"[TTS Server] Error initializing {name}: {e}", flush=True)
    
    async def shutdown(self):
        """Shutdown all backends."""
        for name, backend in self.backends.items():
            try:
                await backend.shutdown()
                print(f"[TTS Server] {name} shutdown")
            except Exception as e:
                print(f"[TTS Server] Error shutting down {name}: {e}")
    
    def get_backend(self, name: str) -> Optional[TTSBackend]:
        """Get a TTS backend by name."""
        return self.backends.get(name)
    
    def get_available_backends(self) -> list[dict]:
        """Get list of available backends."""
        return [
            {
                "id": name,
                "name": backend.name,
                "description": backend.description,
                "languages": backend.get_supported_languages(),
            }
            for name, backend in self.backends.items()
        ]
    
    async def connect_websocket(self, websocket: WebSocket):
        """Accept and register WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[TTS Server] WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect_websocket(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[TTS Server] WebSocket disconnected. Total: {len(self.active_connections)}")


tts_server = TTSServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    await tts_server.initialize()
    yield
    await tts_server.shutdown()


app = FastAPI(
    title="TTS Server",
    description="Streaming Text-to-Speech Server with multiple backends",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint with server info."""
    return {
        "name": "TTS Server",
        "version": "1.0.0",
        "backends": tts_server.get_available_backends()
    }


@app.get("/api/backends")
async def list_backends():
    """List available TTS backends."""
    return {"backends": tts_server.get_available_backends()}


@app.get("/api/voices/{backend_name}")
async def list_voices(backend_name: str):
    """List available voices for a backend."""
    backend = tts_server.get_backend(backend_name)
    if not backend:
        raise HTTPException(status_code=404, detail=f"Backend '{backend_name}' not found")
    
    return {"voices": backend.get_available_voices()}


@app.get("/api/cache/{backend_name}")
async def get_cache_stats(backend_name: str):
    """Get cache statistics for a backend."""
    backend = tts_server.get_backend(backend_name)
    if not backend:
        raise HTTPException(status_code=404, detail=f"Backend '{backend_name}' not found")
    
    if hasattr(backend, 'get_cache_stats'):
        return {"cache": backend.get_cache_stats()}
    return {"cache": {"enabled": False}}


@app.delete("/api/cache/{backend_name}")
async def clear_cache(backend_name: str):
    """Clear cache for a backend."""
    backend = tts_server.get_backend(backend_name)
    if not backend:
        raise HTTPException(status_code=404, detail=f"Backend '{backend_name}' not found")
    
    if hasattr(backend, 'clear_cache'):
        backend.clear_cache()
        return {"success": True, "message": f"Cache cleared for {backend_name}"}
    return {"success": False, "message": "Backend does not support caching"}


@app.post("/api/tts")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize text to speech (non-streaming).
    Returns complete WAV audio file.
    """
    backend = tts_server.get_backend(request.backend)
    if not backend:
        raise HTTPException(status_code=404, detail=f"Backend '{request.backend}' not found")
    
    try:
        start_time = time.time()
        wav_data = await backend.synthesize(
            text=request.text,
            voice=request.voice,
            language=request.language,
            speed=request.speed
        )
        duration = time.time() - start_time
        
        audio_duration = (len(wav_data) - 44) / (24000 * 2)
        
        return Response(
            content=wav_data,
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": f"{audio_duration:.2f}",
                "X-Generation-Time": f"{duration:.2f}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS.
    
    Protocol:
    - Client sends: {"type": "synthesize", "text": "...", "backend": "edge_tts", ...}
    - Server sends: Binary audio chunks (PCM 24kHz 16-bit mono)
    - Server sends: {"type": "done", "duration_ms": ...}
    """
    await tts_server.connect_websocket(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue
            
            msg_type = message.get("type")
            
            if msg_type == "synthesize":
                await handle_synthesize(websocket, message)
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })
    
    except WebSocketDisconnect:
        tts_server.disconnect_websocket(websocket)
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        tts_server.disconnect_websocket(websocket)


async def handle_synthesize(websocket: WebSocket, message: dict):
    """Handle synthesis request over WebSocket."""
    text = message.get("text", "")
    backend_name = message.get("backend", "edge_tts")
    voice = message.get("voice")
    language = message.get("language")
    speed = message.get("speed", 1.0)
    
    if not text:
        await websocket.send_json({"type": "error", "message": "No text provided"})
        return
    
    backend = tts_server.get_backend(backend_name)
    if not backend:
        await websocket.send_json({
            "type": "error",
            "message": f"Backend '{backend_name}' not found"
        })
        return
    
    try:
        await websocket.send_json({
            "type": "start",
            "backend": backend_name,
            "text_length": len(text)
        })
        
        wav_header_sent = False
        total_bytes = 0
        start_time = time.time()
        
        async for chunk in backend.synthesize_stream(
            text=text,
            voice=voice,
            language=language,
            speed=speed
        ):
            if not wav_header_sent:
                wav_header = create_wav_header(data_size=None)
                await websocket.send_bytes(wav_header)
                wav_header_sent = True
            
            await websocket.send_bytes(chunk)
            total_bytes += len(chunk)
        
        duration_ms = (time.time() - start_time) * 1000
        audio_duration = total_bytes / (24000 * 2)
        
        await websocket.send_json({
            "type": "done",
            "bytes_sent": total_bytes,
            "duration_ms": round(duration_ms, 2),
            "audio_duration_s": round(audio_duration, 2)
        })
        
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })


def run_server(host: str = "0.0.0.0", port: int = 8083):
    """Run the TTS server."""
    print(f"""
🎙️  TTS Server - Streaming Text-to-Speech
==========================================
Server starting on http://{host}:{port}

WebSocket: ws://{host}:{port}/ws
REST API:  http://{host}:{port}/api/

Available endpoints:
  GET  /                    - Server info
  GET  /api/backends        - List TTS backends
  GET  /api/voices/{backend} - List voices
  POST /api/tts             - Synthesize (returns WAV)
  WS   /ws                  - Stream synthesis

Press Ctrl+C to stop
""")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TTS Streaming Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8083, help="Port to bind to")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)
