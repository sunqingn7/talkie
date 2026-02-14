#!/bin/bash

# Quick Start - Uses gpt-oss-20b model automatically
# This script starts both servers with the recommended model

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Quick Starting Talkie with gpt-oss-20b...${NC}"
echo "=========================================="

WHISPER_BIN="/home/qing/Project/whisper.cpp/build/bin/whisper-server"
LLAMA_BIN="/home/qing/Project/llama.cpp/build/bin/llama-server"
WHISPER_MODEL="/home/qing/Project/talkie/models/ggml-base.en.bin"
LLAMA_MODEL="/home/qing/.cache/llama.cpp/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf"

# Check binaries and models
if [ ! -f "$WHISPER_BIN" ]; then
    echo -e "${RED}❌ whisper-server not found${NC}"
    exit 1
fi

if [ ! -f "$LLAMA_BIN" ]; then
    echo -e "${RED}❌ llama-server not found${NC}"
    exit 1
fi

if [ ! -f "$WHISPER_MODEL" ]; then
    echo -e "${YELLOW}⚠️  Downloading whisper model...${NC}"
    mkdir -p models
    curl -L -o "$WHISPER_MODEL" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
fi

if [ ! -f "$LLAMA_MODEL" ]; then
    echo -e "${RED}❌ gpt-oss-20b model not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ whisper.cpp ready${NC}"
echo -e "${GREEN}✓ llama.cpp ready${NC}"
echo -e "${GREEN}✓ Model: gpt-oss-20b (12GB)${NC}"

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping servers...${NC}"
    [ -n "$WHISPER_PID" ] && kill $WHISPER_PID 2>/dev/null || true
    [ -n "$LLAMA_PID" ] && kill $LLAMA_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Servers stopped${NC}"
    exit 0
}

trap cleanup INT TERM

echo ""
echo "=========================================="
echo -e "${GREEN}Starting servers...${NC}"
echo "=========================================="
echo ""

# Start whisper.cpp
echo -e "${GREEN}[1/2] Starting whisper.cpp on port 8081...${NC}"
"$WHISPER_BIN" -m "$WHISPER_MODEL" --port 8081 &
WHISPER_PID=$!
sleep 2

if ! kill -0 $WHISPER_PID 2>/dev/null; then
    echo -e "${RED}❌ whisper.cpp failed to start${NC}"
    exit 1
fi
echo -e "${GREEN}   ✓ Running (PID: $WHISPER_PID)${NC}"
echo ""

# Start llama.cpp
echo -e "${GREEN}[2/2] Starting llama.cpp with gpt-oss-20b...${NC}"
echo -e "${YELLOW}   Loading 12GB model (this may take 30-60s)...${NC}"
"$LLAMA_BIN" -m "$LLAMA_MODEL" -c 4096 --port 8080 &
LLAMA_PID=$!
sleep 5

if ! kill -0 $LLAMA_PID 2>/dev/null; then
    echo -e "${RED}❌ llama.cpp failed to start${NC}"
    cleanup
    exit 1
fi
echo -e "${GREEN}   ✓ Running (PID: $LLAMA_PID)${NC}"
echo ""

echo "=========================================="
echo -e "${GREEN}✅ Both servers running!${NC}"
echo "=========================================="
echo ""
echo "📝 Status:"
echo "   whisper.cpp: http://localhost:8081"
echo "   llama.cpp:   http://localhost:8080"
echo "   Model:       gpt-oss-20b"
echo ""
echo "💡 Next step:"
echo "   Open a new terminal and run:"
echo "     cd /home/qing/Project/talkie"
echo "     source venv/bin/activate"
echo "     python src/main.py"
echo ""
echo -e "${YELLOW}⚠️  Press Ctrl+C to stop servers${NC}"
echo ""

wait
