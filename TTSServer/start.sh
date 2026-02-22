#!/bin/bash

# TTS Server startup script

cd "$(dirname "$0")"

# Check for virtual environment
if [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run server
python server.py "$@"
