#!/bin/bash

# Talkie Voice Assistant Setup Script
# MCP-First Design with whisper.cpp

set -e

echo "🚀 Setting up Talkie Voice Assistant..."
echo "=========================================="

# Check Python version
echo "📦 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✓ Virtual environment created"
else
    echo "   ✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create models directory
echo "📦 Setting up models directory..."
mkdir -p models

# Check whisper.cpp binary
echo ""
echo "📦 Checking whisper.cpp..."
WHISPER_BIN="/home/qing/Project/whisper.cpp/build/bin/whisper-server"
if [ -f "$WHISPER_BIN" ]; then
    echo "   ✓ whisper.cpp found at: $WHISPER_BIN"
else
    echo "   ❌ whisper.cpp not found at: $WHISPER_BIN"
    echo "   Please build whisper.cpp first:"
    echo "     cd /home/qing/Project/whisper.cpp"
    echo "     mkdir -p build && cd build"
    echo "     cmake .. && make"
    exit 1
fi

# Check llama.cpp binary
echo ""
echo "📦 Checking llama.cpp..."
LLAMA_BIN="/home/qing/Project/llama.cpp/build/bin/llama-server"
if [ -f "$LLAMA_BIN" ]; then
    echo "   ✓ llama.cpp found at: $LLAMA_BIN"
else
    echo "   ❌ llama.cpp not found at: $LLAMA_BIN"
    echo "   Please build llama.cpp first:"
    echo "     cd /home/qing/Project/llama.cpp"
    echo "     mkdir -p build && cd build"
    echo "     cmake .. && make"
    exit 1
fi

# Download whisper.cpp model if not exists
WHISPER_MODEL="models/ggml-base.en.bin"
if [ ! -f "$WHISPER_MODEL" ]; then
    echo ""
    echo "📦 Downloading whisper.cpp model..."
    echo "   This may take a few minutes..."
    
    # Download using wget or curl
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$WHISPER_MODEL" \
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
    elif command -v curl &> /dev/null; then
        curl -L -o "$WHISPER_MODEL" \
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
    else
        echo "   ⚠️  Please manually download whisper model from:"
        echo "      https://huggingface.co/ggerganov/whisper.cpp"
        echo "      Save to: $WHISPER_MODEL"
        exit 1
    fi
    
    echo "   ✓ Downloaded whisper.cpp model"
else
    echo "   ✓ whisper.cpp model already exists"
fi

# Check for Llama models in cache
echo ""
echo "📦 Checking for Llama models..."
LLAMA_CACHE_DIR="/home/qing/.cache/llama.cpp"
if [ -d "$LLAMA_CACHE_DIR" ]; then
    model_count=$(find "$LLAMA_CACHE_DIR" -name "*.gguf" -type f 2>/dev/null | wc -l)
    if [ "$model_count" -gt 0 ]; then
        echo "   ✓ Found $model_count model(s) in cache"
        echo ""
        echo "   Available models:"
        find "$LLAMA_CACHE_DIR" -name "*.gguf" -type f -exec basename {} \; | while read -r model; do
            size=$(du -h "$LLAMA_CACHE_DIR/$model" | cut -f1)
            echo "     • $model ($size)"
        done
    else
        echo "   ⚠️  No models found in cache"
        echo "   You can download models using:"
        echo "     llama-cli --hf-repo <repo> --model <model>"
    fi
else
    echo "   ⚠️  Cache directory not found: $LLAMA_CACHE_DIR"
fi

# Setup complete
echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Start the servers:"
echo "      ./start_servers.sh"
echo "      (You'll be prompted to select a model)"
echo ""
echo "   2. In a new terminal, run the assistant:"
echo "      source venv/bin/activate"
echo "      python src/main.py"
echo ""
