#!/bin/bash

# Start Talkie Voice Assistant Servers
# This script starts both whisper.cpp and llama.cpp servers with correct paths

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Talkie Voice Assistant Servers...${NC}"
echo "=========================================="

# Configuration
WHISPER_BIN="/home/qing/Project/whisper.cpp/build/bin/whisper-server"
LLAMA_BIN="/home/qing/Project/llama.cpp/build/bin/llama-server"
WHISPER_MODEL="/home/qing/Project/talkie/models/ggml-base.en.bin"
LLAMA_CACHE_DIR="/home/qing/.cache/llama.cpp"

# Check if binaries exist
if [ ! -f "$WHISPER_BIN" ]; then
    echo -e "${RED}❌ whisper-server not found at: $WHISPER_BIN${NC}"
    exit 1
fi

if [ ! -f "$LLAMA_BIN" ]; then
    echo -e "${RED}❌ llama-server not found at: $LLAMA_BIN${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found whisper-server${NC}"
echo -e "${GREEN}✓ Found llama-server${NC}"

# Check if whisper model exists
if [ ! -f "$WHISPER_MODEL" ]; then
    echo -e "${YELLOW}⚠️  Whisper model not found. Downloading...${NC}"
    mkdir -p models
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$WHISPER_MODEL" \
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
    else
        curl -L -o "$WHISPER_MODEL" \
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
    fi
    echo -e "${GREEN}✓ Whisper model downloaded${NC}"
else
    echo -e "${GREEN}✓ Whisper model found${NC}"
fi

# Find available Llama models in cache
echo ""
echo -e "${BLUE}📦 Available Llama models in cache:${NC}"
echo "=========================================="

# List available models
mapfile -t models < <(find "$LLAMA_CACHE_DIR" -name "*.gguf" -type f 2>/dev/null | sort)

if [ ${#models[@]} -eq 0 ]; then
    echo -e "${RED}❌ No models found in $LLAMA_CACHE_DIR${NC}"
    echo "Please download a model first:"
    echo "  llama-cli --hf-repo bartowski/Llama-3.2-3B-Instruct-GGUF --model Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    exit 1
fi

# Display models with numbers
for i in "${!models[@]}"; do
    model_name=$(basename "${models[$i]}")
    model_size=$(du -h "${models[$i]}" | cut -f1)
    if [[ "$model_name" == *"gpt-oss-20b"* ]]; then
        echo "  [$((i+1))] $model_name ($model_size) ⭐ RECOMMENDED"
    else
        echo "  [$((i+1))] $model_name ($model_size)"
    fi
done

echo ""
echo -n "Select a model (1-${#models[@]}, or press Enter for gpt-oss-20b): "
read -r selection

# Default to gpt-oss-20b if no input
if [ -z "$selection" ]; then
    # Find gpt-oss-20b index
    for i in "${!models[@]}"; do
        if [[ "$(basename "${models[$i]}")" == *"gpt-oss-20b"* ]]; then
            selection=$((i+1))
            echo "Auto-selected: gpt-oss-20b"
            break
        fi
    done
fi

# Validate selection
if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt "${#models[@]}" ]; then
    echo -e "${RED}❌ Invalid selection${NC}"
    exit 1
fi

LLAMA_MODEL="${models[$((selection-1))]}"
MODEL_NAME=$(basename "$LLAMA_MODEL")

echo ""
echo -e "${GREEN}✓ Selected: $MODEL_NAME${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}Starting servers...${NC}"
echo "=========================================="
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down servers...${NC}"
    if [ -n "$WHISPER_PID" ]; then
        kill $WHISPER_PID 2>/dev/null || true
        echo "   ✓ whisper.cpp stopped"
    fi
    if [ -n "$LLAMA_PID" ]; then
        kill $LLAMA_PID 2>/dev/null || true
        echo "   ✓ llama.cpp stopped"
    fi
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup INT TERM

# Start whisper.cpp server in background
echo -e "${GREEN}[1/2] Starting whisper.cpp server on port 8081...${NC}"
"$WHISPER_BIN" -m "$WHISPER_MODEL" --port 8081 &
WHISPER_PID=$!

# Wait a moment for whisper to start
sleep 2

# Check if whisper is running
if ! kill -0 $WHISPER_PID 2>/dev/null; then
    echo -e "${RED}❌ Failed to start whisper.cpp server${NC}"
    exit 1
fi

echo -e "${GREEN}   ✓ whisper.cpp running (PID: $WHISPER_PID)${NC}"
echo ""

# Start llama.cpp server in background
echo -e "${GREEN}[2/2] Starting llama.cpp server on port 8080...${NC}"
echo -e "${YELLOW}   Loading $MODEL_NAME... (this may take a moment)${NC}"
"$LLAMA_BIN" -m "$LLAMA_MODEL" -c 4096 --port 8080 &
LLAMA_PID=$!

# Wait a moment for llama to start
sleep 5

# Check if llama is running
if ! kill -0 $LLAMA_PID 2>/dev/null; then
    echo -e "${RED}❌ Failed to start llama.cpp server${NC}"
    cleanup
    exit 1
fi

echo -e "${GREEN}   ✓ llama.cpp running (PID: $LLAMA_PID)${NC}"
echo ""

echo "=========================================="
echo -e "${GREEN}✅ Both servers are running!${NC}"
echo "=========================================="
echo ""
echo "📝 Server Status:"
echo "   whisper.cpp: http://localhost:8081 (PID: $WHISPER_PID)"
echo "   llama.cpp:   http://localhost:8080 (PID: $LLAMA_PID)"
echo "   Model:       $MODEL_NAME"
echo ""
echo "💡 To use the assistant:"
echo "   1. Open a new terminal"
echo "   2. cd /home/qing/Project/talkie"
echo "   3. source venv/bin/activate"
echo "   4. python src/main.py"
echo ""
echo -e "${YELLOW}⚠️  Press Ctrl+C to stop both servers${NC}"
echo ""

# Wait for user to press Ctrl+C
wait
