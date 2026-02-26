# Talkie Voice Assistant

🗣️ An intelligent voice-powered AI assistant with web control panel

## ✨ Features

- **Voice Interface**: Speech-to-text (whisper.cpp) + Text-to-Speech
- **Multiple TTS Engines**: Qwen TTS (default), Edge TTS, Coqui XTTS, pyttsx3
- **16+ Voices**: English, Chinese, Japanese, Korean, Spanish, French, German, and more
- **Web Control Panel**: Real-time control via browser interface
- **File Reading**: Read uploaded files or webpages aloud with pause/resume
- **Smart Interruption**: Chat messages pause file reading, auto-resume after response
- **Multi-LLM Support**: Switch between vllm, llama.cpp, ollama, and cloud providers
- **Weather**: Automatic location detection from IP, works for any city worldwide
- **Web Search**: Integrated search via Tavily API / DuckDuckGo
- **14+ Built-in Tools**: Weather, calculator, timer, file operations, commands, etc.

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

### File Reading with Pause/Resume (Latest)
- **Chunk-by-Chunk Reading**: Progressive loading with buffer-based streaming
- **Smart Interruption**: Chat messages automatically pause file reading
- **Auto-Resume**: File reading resumes automatically after chat response
- **Position Persistence**: Resume from exact position across sessions
- **URL Support**: Read webpages directly by providing URL

### Multi-LLM Orchestrator
- **Provider Support**: vllm, llama.cpp, ollama, Google, Anthropic, xAI
- **Auto-Detection**: Automatic model detection for vllm and llama.cpp
- **Agent System**: Specialized agents (coder, reasoner, searcher, etc.)
- **Fallback System**: Automatic fallback to backup provider on failure
- **Dynamic Switching**: Switch providers via web interface or commands

### Qwen TTS Integration
- **Default TTS Engine**: Qwen/Qwen3-TTS-12Hz (high quality, fast)
- **Multilingual**: Excellent Chinese and English support
- **Auto Language Detection**: Automatically detects input language
- **Voice Design**: Custom voice characteristics support

### Weather Tool Upgrade
- **API**: Open-Meteo (free, no API key)
- **Features**:
  - Weather for any city worldwide
  - IP-based location auto-detection (ipinfo.io)
  - Temperature, humidity, wind speed, conditions
- **Smart Query Handling**: Auto-detects location if not provided

### System Improvements
- **Voice Daemon**: Centralized TTS queue with priority management
- **Priority-Based Speech**: HIGH (chat) vs NORMAL (file reading) priorities
- **Dynamic Audio Timeout**: Long responses play completely without cutoff
- **Sentence-Based Chunking**: Intelligent text splitting for natural pauses
- **Config Persistence**: All settings saved automatically
- **Enhanced Web Interface**: Real-time status, model switching, file upload

## 📁 Configuration

### Key Settings (`config/settings.yaml`)

```yaml
tts:
  engine: qwen_tts  # Options: qwen_tts, edge_tts, coqui, pyttsx3
  voice_output: web  # Output to web interface (or 'local' for system audio)

llm:
  default_provider: llamacpp  # Options: vllm, llamacpp, ollama, google, etc.
  auto_detect_models: true  # Auto-detect available models

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
| **qwen_tts** | Local (GPU) | Qwen3-TTS - default, high quality, fast |
| **edge_tts** | Online | Edge TTS - 16+ voices, no install |
| **coqui** | Local | XTTS-v2 - requires ~1.5GB download |
| **pyttsx3** | Local | Basic fallback TTS |

### LLM Providers

| Provider | Type | Description |
|----------|------|-------------|
| **vllm** | Local (GPU) | High-throughput inference server |
| **llamacpp** | Local (CPU/GPU) | llama.cpp backend, auto-detect models |
| **ollama** | Local | Ollama runtime support |
| **google** | Cloud | Google Gemini API |
| **anthropic** | Cloud | Claude API |
| **xAI** | Cloud | Grok API |

## 🛠️ Available Tools

1. ✅ `listen` - Speech-to-text (whisper.cpp)
2. ✅ `speak` - Text-to-speech (Qwen TTS / Edge TTS / Coqui / pyttsx3)
3. ✅ `read_file_chunk` - Read files/webpages aloud with pause/resume
4. ✅ `pause_reading` - Pause current file reading
5. ✅ `resume_reading` - Resume paused file reading
6. ✅ `stop_reading` - Stop file reading completely
7. ✅ `weather` - Weather with IP auto-detection
8. ✅ `execute_command` - Run shell commands
9. ✅ `write_file` - Write files
10. ✅ `list_directory` - List directories
11. ✅ `wake_word` - Wake phrase detection
12. ✅ `voice_activity` - Voice activity detection
13. ✅ `timer` - Timer functionality
14. ✅ `calculator` - Math calculations
15. ✅ `web_search` - Web search
16. ✅ `datetime` - Current date/time

## 📝 Usage Examples

### Voice Commands

```
"What's the weather?" → Auto-detects location
"What's the weather in Tokyo?" → Tokyo weather
"Read this file" → Reads uploaded file aloud
"Pause reading" / "Resume reading" → Control file reading
"Search for AI news" → Web search
"Set a 5 minute timer"
"Calculate 15 times 23"
```

### File Reading

- **Upload a file** via web interface or mention it in chat
- **Read webpage**: Just provide a URL (e.g., "Read https://example.com")
- **Chat during reading**: File reading pauses automatically, resumes after response
- **Position saved**: Resume from exact position across sessions

### Web Interface

1. Start web server: `python web_server.py`
2. Open browser: `http://localhost:8082`
3. Use **Control Panel** to:
   - Switch TTS engines (Qwen, Edge, Coqui)
   - Switch LLM providers (vllm, llama.cpp, ollama, cloud)
   - Upload files for reading
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
│   ├── core/             # Core LLM, model management, voice daemon
│   │   ├── llm_providers/  # Multi-provider LLM support
│   │   ├── voice_daemon.py # Priority-based TTS queue
│   │   └── reading_position_manager.py  # Position persistence
│   ├── mcp_integration/  # MCP server and tool registration
│   ├── tools/            # All available tools
│   │   ├── qwen_tts_tool.py     # Qwen TTS (default)
│   │   ├── edge_tts_tool.py     # Edge TTS
│   │   ├── tts_tool.py          # TTS manager
│   │   ├── file_reading_tool.py # File reading with pause/resume
│   │   └── web_fetch_tool.py    # Webpage fetching
│   ├── utils/            # Utilities
│   │   └── file_stream_reader.py # Streaming file reader
│   └── web/              # Web interface
│       ├── web_server.py
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

- **Qwen TTS** requires GPU for best performance (CUDA recommended)
- **Edge TTS** requires internet connection (online service)
- **Coqui TTS** can work offline but requires ~1.5GB download
- **vllm/llama.cpp** require compatible GGUF or HF models
- **Weather** uses ipinfo.io for location detection (free, no signup needed)
- **Web Search** requires Tavily API key (free tier: 1000 calls/month)
- **File Reading** works with any text file or webpage URL

---

**Built with ❤️ using Python, FastAPI, Qwen TTS, and Multi-LLM Orchestrator**

**Latest**: Commit `13b9dc0` - Pause/resume file reading when chat arrives
