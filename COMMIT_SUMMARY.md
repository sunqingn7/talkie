# Talkie Voice Assistant - Commit Summary

## 📝 Commit Message

Title: feat: Add Edge TTS, improve weather with IP detection, and enhance TTS engine switching

### Changes

**New Features:**
- Add Microsoft Edge TTS as default TTS engine with 16+ multilingual voices
- Implement IP-based location auto-detection for weather queries (ipinfo.io + ip-api.com)
- Add TTS engine switching (Edge TTS, Coqui, pyttsx3) with web UI controls
- Add dynamic audio timeout based on content length (prevents long text cutoff)
- Create configuration template (settings.example.yaml) with no sensitive data

**Improvements:**
- Fix voice switching issue where first voice didn't trigger onchange event
- Update web search to use Tavily API with DuckDuckGo fallback
- Make system prompts smarter for weather queries (auto-detect location)
- Add config persistence for all TTS settings
- Add inline onchange handlers for more reliable UI controls

**Files Added:**
- `src/tools/edge_tts_tool.py` - Edge TTS integration with 16+ voices
- `config/settings.example.yaml` - Configuration template (excluded sensitive data)
- `.gitignore` - Excludes configs, models, cache, temp files

**Files Modified:**
- `src/tools/tts_tool.py` - Engine switching, dynamic timeout
- `src/tools/weather_tool.py` - IP geolocation, Open-Meteo API integration
- `src/web/server.py` - Engine switching API, config save
- `src/web/templates/index.html` - CSS link, engine/voice selectors
- `src/web/static/js/app.js` - UI updates, dropdown handlers
- `config/settings.yaml` - Updated default engine, weather settings, prompts
- `README.md` - Complete rewrite with current feature set
- `requirements.txt` - Added edge-tts, tavily-python

### Breaking Changes

**Configuration:**
- Default TTS engine changed from `coqui` to `edge_tts`
- Users must update their `config/settings.yaml` (copy from `config/settings.example.yaml`)
- API keys must be added to config manually (Tavily, OpenWeatherMap optional)

**Setup:**
- `edge-tts` is now required dependency
- Config template must be copied to use the system
- Absolute paths in config must be updated to match user's system

### Testing

All features tested and working:
- ✅ Edge TTS works for all voices
- ✅ TTS engine switching (Edge ↔ Coqui ↔ pyttsx3)
- ✅ Voice selection and persistence
- ✅ Weather with IP location detection
- ✅ Weather for any city
- ✅ Web search with Tavily API
- ✅ Config save/load
- ✅ Long text playback with dynamic timeout

### Documentation Updated

- `README.md` - Complete rewrite with Edge TTS, weather IP detection, web panel
- `.gitignore` - Proper exclusions for sensitive data
- `config/settings.example.yaml` - Template with placeholders
- `SESSION_STATUS.md` - Updated with latest changes

### API Keys & Security

- **Removed**: Tavily API key from `config/settings.yaml` (changed to null)
- **Created**: `config/settings.example.yaml` with all sensitive data replaced with placeholders
- **Added**: `.gitignore` to prevent accidental commits of:
  - `config/*.yaml` (real configs)
  - `models/` (large model files)
  - `*.bin`, `*.gguf` (model weights)
  - `*.mp3`, `*.wav` (audio files)
  - `.env`, `*.env.local` (environment secrets)

### Dependencies Added

- `edge-tts>=6.1.0` - Microsoft Edge TTS
- `tavily-python>=0.5.0` - Web search API

### How to Use After Checkin

1. Clone repository
2. `pip install -r requirements.txt`
3. `cp config/settings.example.yaml config/settings.yaml`
4. Edit `config/settings.yaml`:
   - Add your Tavily API key (get from https://tavily.com)
   - Update paths to whisper.cpp, llama-server, models
   - Configure custom settings
5. `python web_server.py`
6. Open http://localhost:8082

---

**Status**: ✅ Ready for GitHub commit
**All sensitive information removed**
