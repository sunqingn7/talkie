#!/usr/bin/env python3
"""
Talkie Web Control Panel Launcher
Run this script to start the web interface
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from web.server import run_web_server

if __name__ == "__main__":
    run_web_server()
