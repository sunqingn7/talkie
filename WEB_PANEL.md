# Talkie Web Control Panel

A modern web-based interface for the Talkie Voice Assistant, featuring real-time chat, system controls, and settings management.

## Features

### 🎨 Modern UI
- Dark theme with gradient accents
- Responsive design for desktop and mobile
- Smooth animations and transitions
- Intuitive sidebar navigation

### 💬 Real-time Chat
- WebSocket-based instant messaging
- Message history with timestamps
- Quick action buttons for common queries
- Auto-scroll to latest messages
- Support for rich text formatting (links, newlines)

### 🎛️ Control Panel
- **Model Management**: Switch between TTS models
  - English (Tacotron2)
  - Chinese (Baker)
  - XTTS-v2 (17 languages, requires license)
- **System Status**: Monitor MCP server, LLM client, and tools
- **Quick Tools**: Test TTS, STT, and check weather
- **Available Tools**: Visual grid of all 13 available tools

### ⚙️ Settings
- **Audio**: Toggle voice output and input
- **Chat**: Show/hide tool call details, enable auto-scroll
- **Connection**: WebSocket status and reconnect option

### 🎤 Voice Support
- Browser-based voice output using Web Speech API
- Voice input recording interface (UI ready, STT integration pending)
- Visual voice recording animation

## Installation

The web panel is included in the main Talkie project. First, install the additional dependencies:

```bash
cd /home/qing/Project/talkie
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Start the Web Server

```bash
cd /home/qing/Project/talkie
source venv/bin/activate
python web_server.py
```

Or run directly:

```bash
python src/web/server.py
```

The server will start on **http://0.0.0.0:8082**

### Access the Interface

Open your web browser and navigate to:

```
http://localhost:8082
```

## Architecture

### Backend (FastAPI)
- **File**: `src/web/server.py`
- WebSocket endpoint for real-time communication
- REST API for system status and model management
- Integrates with existing Talkie MCP server and LLM client
- Handles conversation history and message routing

### Frontend (Vanilla JS)
- **File**: `src/web/static/js/app.js`
- WebSocket client for server communication
- Dynamic UI updates
- Voice recording using Web Audio API
- Responsive design utilities

### Templates & Styling
- **HTML**: `src/web/templates/index.html`
- **CSS**: `src/web/static/css/style.css`
- Modern dark theme with CSS variables
- Mobile-responsive layout
- CSS animations for smooth interactions

## API Endpoints

### WebSocket
- `ws://localhost:8082/ws` - Real-time chat and control

### REST API
- `GET /` - Main web interface
- `GET /api/status` - System status
- `GET /api/models` - Available TTS models
- `POST /api/switch-model` - Switch TTS model

## WebSocket Messages

### Client → Server
```json
{
  "type": "user_message",
  "content": "Hello!"
}
```

```json
{
  "type": "switch_model",
  "model_id": "tts_models/en/ljspeech/tacotron2-DDC"
}
```

```json
{
  "type": "clear_history"
}
```

### Server → Client
```json
{
  "type": "assistant_message",
  "content": "Hello! How can I help you?",
  "timestamp": "2026-02-12T10:30:00"
}
```

```json
{
  "type": "system_status",
  "mcp_server_ready": true,
  "llm_client_ready": true,
  "available_tools": ["listen", "speak", "weather", ...],
  "conversation_count": 5
}
```

```json
{
  "type": "thinking"
}
```

## File Structure

```
src/web/
├── server.py              # FastAPI web server
├── templates/
│   └── index.html         # Main web interface
└── static/
    ├── css/
    │   └── style.css      # Stylesheet
    └── js/
        └── app.js         # Frontend application
```

## Customization

### Change Port
Edit `src/web/server.py` and modify the `run_web_server()` call:

```python
run_web_server(host="0.0.0.0", port=8083)  # Change to desired port
```

### Add New Tools
Tools are automatically populated from the MCP server. To add a new tool:

1. Add it to `config/settings.yaml` in `mcp.tools_enabled`
2. Create the tool in `src/tools/`
3. Restart the web server

### Customize Theme
Edit CSS variables in `src/web/static/css/style.css`:

```css
:root {
    --primary-color: #6366f1;  /* Change primary color */
    --bg-dark: #0f172a;         /* Change background */
    /* ... */
}
```

## Troubleshooting

### WebSocket Connection Failed
- Check if the web server is running
- Verify firewall settings allow port 8082
- Check browser console for errors

### Voice Output Not Working
- Ensure browser supports Web Speech API (Chrome, Edge, Safari)
- Check if voice output is enabled in settings
- Verify system volume is not muted

### Model Switching Fails
- Ensure `COQUI_TOS_AGREED=1` environment variable is set for XTTS-v2
- Check server logs for error messages
- Verify the model ID is correct

## Browser Compatibility

- ✅ Chrome 60+
- ✅ Firefox 60+
- ✅ Safari 14+
- ✅ Edge 79+

Note: Voice features require modern browsers with Web Speech API support.

## Future Enhancements

- [ ] Full STT integration via WebSocket audio streaming
- [ ] User authentication and session management
- [ ] Conversation history persistence
- [ ] Customizable themes
- [ ] Mobile app wrapper
- [ ] Multi-language UI support
