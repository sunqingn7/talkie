# TTS Engine Switching Improvements

## Changes Made:

### 1. Removed Redundant Edge TTS Voice Section (HTML)
- Deleted the separate "Edge TTS Voice Selector" section from `index.html`
- The Voice Persona dropdown now handles both Edge TTS voices and XTTS speakers

### 2. Updated JavaScript (`app.js`)
- Modified `updateEngineUI()` to:
  - Show/hide XTTS Model section only (removed edge-voice-section references)
  - Always show Speaker section for all engines
  - Update section labels dynamically based on engine type
  
- Modified `updateTTSSpeakers()` to:
  - Use Edge TTS voices when engine is "edge_tts"
  - Use XTTS speakers when engine is "coqui"
  - Display locale info for Edge TTS voices (e.g., "Aria (Female) - en-US")
  - Handle speaker selection properly for both engines
  
- Updated message handlers:
  - `tts_engine_switched`: Updates UI and refreshes models
  - `edge_voice_switched`: Updates shared speaker dropdown
  
- Removed event listener for deleted edge-voice-select element

### 3. How It Works Now:
1. User selects TTS Engine from dropdown (Edge TTS / Coqui / pyttsx3)
2. System switches TTS engine on the backend
3. UI updates to show appropriate sections:
   - Edge TTS: Shows Voice dropdown with locale info, hides XTTS Model section
   - Coqui: Shows XTTS Model section and Voice Persona dropdown
4. Voice/Persona dropdown is repopulated with appropriate options
5. Selecting a voice triggers auto-demo

### Testing:
1. Start web server: `python web_server.py`
2. Open http://localhost:8082
3. Go to Control Panel
4. Switch TTS Engine - Voice dropdown should update immediately
5. Select a voice - Should hear demo audio
