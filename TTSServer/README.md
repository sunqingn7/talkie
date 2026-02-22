# TTS Server

A standalone streaming Text-to-Speech server supporting multiple TTS backends.

## Features

- **WebSocket Streaming**: Real-time audio streaming for low latency
- **REST API**: Simple HTTP endpoints for non-streaming synthesis
- **Multiple Backends**: Support for Edge TTS, Qwen TTS, and more
- **Standard Format**: PCM 24kHz 16-bit mono audio

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python server.py
```

Server runs on `http://localhost:8083` by default.

## API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info |
| `/api/backends` | GET | List available TTS backends |
| `/api/voices/{backend}` | GET | List voices for a backend |
| `/api/tts` | POST | Synthesize text (returns WAV) |

### WebSocket

Connect to `ws://localhost:8083/ws`

**Protocol:**

1. Send synthesis request:
```json
{
  "type": "synthesize",
  "text": "Hello, world!",
  "backend": "edge_tts",
  "voice": "en-US-AriaNeural",
  "speed": 1.0
}
```

2. Receive stream:
- First: WAV header (44 bytes)
- Then: Binary PCM chunks
- Finally: `{"type": "done", "duration_ms": ...}`

## Backends

### Edge TTS (Online)
- Microsoft Edge neural voices
- Requires internet connection
- Low latency, high quality
- Voices: Aria, Jenny, Xiaoxiao, etc.

### Qwen TTS (Offline)
- Local Qwen3-TTS models
- CustomVoice and VoiceDesign modes
- Requires GPU for good performance
- Voices: Vivian, Serena, Ryan, etc.

## Configuration

Edit `config.yaml` to configure:

```yaml
server:
  host: "0.0.0.0"
  port: 8083

backends:
  edge_tts:
    enabled: true
    default_voice: "en-US-AriaNeural"
  
  qwen_tts:
    enabled: true
    model_type: "custom_voice_0.6b"
    default_speaker: "Vivian"
    device: "cuda:0"
```

## Integration with Talkie

Add to `config/settings.yaml`:

```yaml
tts:
  streaming_server: "ws://localhost:8083"
  use_streaming: true
```

## Example Usage

### Python Client

```python
import asyncio
import websockets
import json

async def stream_tts(text: str):
    async with websockets.connect("ws://localhost:8083/ws") as ws:
        # Send request
        await ws.send(json.dumps({
            "type": "synthesize",
            "text": text,
            "backend": "edge_tts"
        }))
        
        # Receive audio
        audio_data = bytearray()
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                audio_data.extend(msg)
            else:
                data = json.loads(msg)
                if data.get("type") == "done":
                    break
        
        # Save to file
        with open("output.wav", "wb") as f:
            f.write(audio_data)

asyncio.run(stream_tts("Hello, world!"))
```

### curl (REST API)

```bash
# Synthesize text
curl -X POST http://localhost:8083/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "backend": "edge_tts"}' \
  --output output.wav

# List backends
curl http://localhost:8083/api/backends

# List voices
curl http://localhost:8083/api/voices/edge_tts
```
