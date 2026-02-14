# Talkie Voice Assistant

🗣️ An intelligent voice-powered AI assistant with web control panel

## ✨ Features

- **Voice Interface**: Speech-to-text (whisper.cpp) + Text-to-Speech
- **Multiple TTS Engines**: Edge TTS (online, high quality), Coqui XTTS, pyttsx3
- **16+ Voices**: English, Chinese, Japanese, Korean, Spanish, French, German, and more
- **Web Control Panel**: Real-time control via browser interface
- **Weather**: Automatic location detection from IP, works for any city worldwide
- **Web Search**: Integrated search via Tavily API / DuckDuckGo
- **13 Built-in Tools**: Weather, calculator, timer, file operations, commands, etc.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install core dependencies:

```bash
pip install edge-tts tavily-python fastapi uvicorn websockets
pip install fast-whisper  # Or use whisper.cpp
pip install coqui-tts  # Optional, for local TTS
```

### 2. Configure

Copy the example config and update it:

```bash
cp config/settings.example.yaml config/settings.yaml

# Edit config/settings.yaml to add:
# - Tavily API key (get from https://tavily.com, free tier)
# - Paths to whisper.cpp and llama-server
# - Model paths
# - Custom settings
```

### 3. Run

**Web Interface (Recommended)**:
```bash
python web_server.py
# Open http://localhost:8082
```

**Command Line**:
```bash
python src/main.py
```

## 🎨 Web Control Panel

Access at **http://localhost:8082**

- **Real-time Chat**: Interface for conversation with the assistant
- **TTS Engine Control**: Switch between Edge TTS, Coqui, and pyttsx3
- **Voice Selection**: 16+ voices for Edge TTS, XTTS personas for Coqui
- **LLM Management**: Switch between available models
- **System Status**: Monitor all tools and services

## 🌍 Recent Updates

### Edge TTS Integration
- **Default TTS Engine**: Microsoft Edge TTS (online, no download required)
- **16+ Voices**: Multilingual support with high quality
- **Fast & Efficient**: No model loading time, minimal latency
- **Fallback System**: Coqui TTS → pyttsx3 if unavailable

### Weather Tool Upgrade
- **API**: Open-Meteo (free, no API key)
- **Features**:
  - Weather for any city worldwide
  - IP-based location auto-detection (ipinfo.io)
  - Temperature, humidity, wind speed, conditions
- **Smart Query Handling**: Auto-detects location if not provided

### System Improvements
- **Dynamic Audio Timeout**: Long responses play completely without cutoff
- **Config Persistence**: All settings saved automatically
- **Inline Event Handlers**: Reliable dropdown controls
- **Enhanced Documentation**: Improved coverage and usage examples

## 📁 Configuration

### Key Settings (`config/settings.yaml`)

```yaml
tts:
  engine: edge_tts  # Default: edge_tts, coqui, pyttsx3
  edge_voice: en-US-AriaNeural

weather:
  api_key: null  # Optional: OpenWeatherMap API key
  auto_detect_location: true  # Auto-detect user location

web_search:
  tavily_api_key: null  # Required for web search
  use_duckduckgo_fallback: true
```

### TTS Engines

| Engine | Type | Description |
|--------|------|-------------|
| **edge_tts** | Online | Edge TTS - default, 16+ voices |
| **coqui** | Local | XTTS-v2 - requires ~1.5GB download |
| **pyttsx3** | Local | Basic fallback TTS |

## 🛠️ Available Tools

1. ✅ `listen` - Speech-to-text (whisper.cpp)
2. ✅ `speak` - Text-to-speech (Edge TTS / Coqui / pyttsx3)
3. ✅ `weather` - Weather with IP auto-detection
4. ✅ `execute_command` - Run shell commands
5. ✅ `read_file` - Read files
6. ✅ `write_file` - Write files
7. ✅ `list_directory` - List directories
8. ✅ `wake_word` - Wake phrase detection
9. ✅ `voice_activity` - Voice activity detection
10. ✅ `timer` - Timer functionality
11. ✅ `calculator` - Math calculations
12. ✅ `web_search` - Web search
13. ✅ `datetime` - Current date/time

## 📝 Usage Examples

### Voice Commands

```
"What's the weather?" → Auto-detects your location
"What's the weather in Tokyo?" → Gets Tokyo weather
"What date is it today?" → System date/time
"Search for AI news" → Web search
"Set a 5 minute timer"
"Calculate 15 times 23"
```

### Web Interface

1. Start web server: `python web_server.py`
2. Open browser: `http://localhost:8082`
3. Use **Control Panel** to:
   - Switch TTS engines
   - Select voices/personas
   - Test voices
   - Monitor system status

## 🔧 Setup Requirements

### Optional (Recommended for best experience)

- **whisper.cpp** - Local speech-to-text:
  ```bash
  # Clone and build whisper.cpp
  git clone https://github.com/ggerganov/whisper.cpp
  cd whisper.cpp
  make
  ```

- **llama.cpp** - LLM inference:
  ```bash
  # Clone and build llama.cpp
  git clone https://github.com/ggerganov/llama.cpp
  cd llama.cpp
  make
  ```

- **Coqui TTS** - Local high-quality TTS (optional):
  ```bash
  pip install TTS
  export COQUI_TOS_AGREED=1
  ```

### Required

- Python 3.10+
- FastAPI, uvicorn, websockets
- edge-tts (for default TTS)
- tavily-python (for web search)

## 📊 Project Structure

```
talkie/
├── config/
│   ├── models.yaml        # LLM model configurations
│   ├── settings.yaml      # Main configuration (not in git)
│   └── settings.example.yaml  # Example configuration template
├── src/
│   ├── core/             # Core LLM, model management
│   ├── mcp_integration/  # MCP server and tool registration
│   ├── tools/            # All available tools
│   │   ├── edge_tts_tool.py     # Edge TTS (NEW)
│   │   ├── tts_tool.py          # TTS manager
│   ├── web_search_tool.py
│   └── weather_tool.py     # Weather with IP detection
│   └── web/              # Web interface
│       ├── server.py
│       ├── templates/
│       │   └── index.html
│       └── static/
│           ├── css/
│           │   └── style.css
│           └── js/
│               └── app.js
└── requirements.txt
```

## 🎯 Use Cases

- **Personal Voice Assistant**: Hands-free information access
- **Smart Home Control**: Connect to IoT devices via commands
- **Accessibility**: Voice interface for accessibility needs
- **Education**: Interactive learning assistant
- **Demo Platform**: Show off AI capabilities

## 🔐 Security & Privacy

- All sensitive data is in `config/settings.yaml` (excluded from git)
- Use provided `config/settings.example.yaml` as template
- No telemetry or data collection
- Runs locally on your machine

## 📄 License

This project is open source and available under an appropriate license.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- More TTS engines and voices
- Additional tools and features
- Mobile responsiveness improvements
- User interface enhancements
- Integration with other services

## 📝 Notes

- **Edge TTS** requires internet connection (online service)
- **Coqui TTS** can work offline but requires ~1.5GB download
- **Weather** uses ipinfo.io for location detection (free, no signup needed)
- **Web Search** requires Tavily API key (free tier: 1000 calls/month)

---

**Built with ❤️ using Python, FastAPI, and edge-tts**
