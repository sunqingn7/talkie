# LLM Model Management Feature

## Overview
Added the ability to switch between different LLM models directly from the web control panel. The system automatically stops the current llama-server and starts a new one with the selected model.

## Features

### 🔧 Model Configuration (`config/models.yaml`)
Pre-configured models with optimal parameters:

1. **GPT-OSS 20B (MXFP4)** - 12GB
   - Good general-purpose model
   - Optimized for chat and coding
   - Context size: 32K
   - All layers on GPU

2. **Qwen3 Coder Next (MXFP4)** - 41GB
   - Excellent for programming tasks
   - Large MoE architecture
   - Context size: 32K
   - NUMA optimization for large models

3. **GLM 4.7 Flash (Q4)** - ~3GB
   - Fast and efficient
   - Large context: 128K
   - Good for daily tasks
   - Can handle parallel requests

### 🎛️ Web Interface Updates

#### Control Panel
- **LLM Model Selector**: Lists all available models with download status
- **Server Status Indicator**: Shows if llama-server is running
- **Restart Button**: Restart the current server
- **Model Details**: Shows parameters, size, quantization, and description

#### Model Selection Behavior
- ✅ **Downloaded models**: Can be selected (click to switch)
- ❌ **Not downloaded**: Shown as disabled with "Not Downloaded" badge
- 🔄 **Active model**: Highlighted with "Active" badge
- ⚠️ **Confirmation dialog**: Asks before switching (warns about restart)

### 🖥️ Backend Components

#### Model Manager (`src/core/model_manager.py`)
```python
# Key methods:
- scan_available_models()      # Scan ~/.cache/llama.cpp/ for GGUF files
- get_running_model()          # Check if llama-server is running
- stop_server()                # Gracefully stop llama-server
- start_server(model_id)       # Start llama-server with model
- switch_model(model_id)       # Complete switch process
- restart_server()             # Restart with current model
```

#### Server Parameters (Optimized)
```yaml
ctx_size: 32768          # Context window
threads: 8               # CPU threads
threads_batch: 8         # Batch processing threads
n_gpu_layers: -1         # All layers on GPU (-1 = auto)
batch_size: 2048         # Batch size
parallel: 1              # Parallel slots
flash_attn: true         # Flash attention for speed
cont_batching: true      # Continuous batching
embeddings: true         # Enable embeddings API
mlock: false            # Don't lock memory
no_mmap: false          # Allow memory mapping
```

### 🌐 API Endpoints

#### REST API
- `GET /api/llm-status` - Get LLM server status
- `POST /api/llm-switch` - Switch to different model
- `POST /api/llm-restart` - Restart current server

#### WebSocket Messages
```javascript
// Client → Server
{type: "get_llm_status"}
{type: "switch_llm_model", model_id: "gpt-oss-20b-mxfp4"}
{type: "restart_llm_server"}

// Server → Client
{type: "llm_status", running: true, current_model: "..."}
{type: "llm_switching", message: "Switching to model..."}
{type: "llm_model_switched", success: true, message: "..."}
{type: "llm_server_restarted", success: true, message: "..."}
```

## Files Modified/Created

### New Files
- `config/models.yaml` - Model configurations with parameters
- `src/core/model_manager.py` - LLM server lifecycle management
- `WEB_PANEL.md` - Web panel documentation

### Updated Files
- `src/web/server.py` - Added LLM management endpoints
- `src/web/templates/index.html` - Added LLM model selector UI
- `src/web/static/js/app.js` - Added model switching logic
- `requirements.txt` - Added psutil dependency

## Usage

### Via Web Interface
1. Open http://localhost:8082
2. Go to "Control Panel"
3. View available LLM models
4. Click on a downloaded model to switch
5. Confirm the restart
6. Wait for server to be ready

### Via API
```bash
# Get status
curl http://localhost:8082/api/llm-status

# Switch model
curl -X POST http://localhost:8082/api/llm-switch \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gpt-oss-20b-mxfp4"}'

# Restart server
curl -X POST http://localhost:8082/api/llm-restart
```

## How Model Switching Works

1. **User selects model** from web interface
2. **Confirmation dialog** warns about server restart
3. **Backend stops** current llama-server process
   - Sends SIGTERM first (graceful)
   - Waits up to 5 seconds
   - Sends SIGKILL if needed (force)
4. **Backend starts** new llama-server with selected model
   - Builds command with optimal parameters
   - Starts process with new process group
   - Waits for server to be ready (health check)
5. **LLM client reconnects** to new server
6. **UI updates** to show new active model

## Safety Features

- ✅ Graceful shutdown with fallback to force kill
- ✅ Health check before confirming server ready
- ✅ Process verification (checks PID exists)
- ✅ Error handling with user feedback
- ✅ Confirmation dialogs for destructive actions
- ✅ Read-only mode for undownloaded models

## Performance Optimizations

Parameters tuned for RTX 3090 Ti (24GB VRAM):
- All GPU layers offloaded (`-ngl -1`)
- Flash attention enabled for speed
- Continuous batching for throughput
- Optimal batch sizes per model
- NUMA support for large models

For systems with less VRAM, edit `config/models.yaml` and adjust:
- `n_gpu_layers`: Reduce to fit in VRAM
- `batch_size`: Lower for memory savings
- `ctx_size`: Reduce context window

## Future Enhancements

- [ ] Download models directly from web interface
- [ ] Auto-detect optimal parameters based on hardware
- [ ] Model quantization options
- [ ] VRAM usage monitoring
- [ ] Multi-GPU support configuration
- [ ] Model comparison/benchmarking
- [ ] Custom model parameters per user
