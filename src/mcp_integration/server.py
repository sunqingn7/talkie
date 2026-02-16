"""
MCP Server for Talkie Voice Assistant
All functionality is exposed as tools that the LLM can call.
"""

import json
import asyncio
from typing import Any, Dict, List, Optional
from mcp.server import Server
from mcp.types import TextContent, Tool
import yaml

# Import voice daemon
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.voice_daemon import get_voice_daemon, VoiceDaemon


class TalkieMCPServer:
    """MCP Server that exposes all assistant capabilities as tools."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.server = Server(self.config["mcp"]["server_name"])
        self.tools = {}
        self.voice_daemon: Optional[VoiceDaemon] = None
        self._register_tools()
        
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _register_tools(self):
        """Register all available tools."""
        enabled = self.config["mcp"]["tools_enabled"]
        
        if "listen" in enabled:
            from tools.stt_tool import STTTool
            self.tools["listen"] = STTTool(self.config)
            
        # Always register speak tool for web interface use, even if not in enabled list
        # The LLM won't see it in list_tools, but web interface can still call it
        from tools.tts_tool import TTSTool
        self.tools["speak"] = TTSTool(self.config)
        
        # Initialize Voice Daemon with TTS tool
        self.voice_daemon = get_voice_daemon(self.tools["speak"])
        
        # Register TTS reader tools for reading files with paragraph-by-paragraph TTS
        # These now use the voice daemon internally
        from tools.tts_reader_tool import TTSReaderTool, StopReadingTool
        self.tools["read_file_aloud"] = TTSReaderTool(self.config)
        self.tools["read_file_aloud"].set_voice_daemon(self.voice_daemon)
        
        self.tools["stop_reading"] = StopReadingTool(self.config)
        self.tools["stop_reading"].set_reader_tool(self.tools["read_file_aloud"])
        
        # Register Voice Daemon control tools
        from tools.voice_daemon_tool import VoiceDaemonStatusTool, VoiceDaemonStopTool, VoiceDaemonSkipTool
        self.tools["voice_daemon_status"] = VoiceDaemonStatusTool(self.config)
        self.tools["voice_daemon_status"].set_voice_daemon(self.voice_daemon)
        
        self.tools["voice_daemon_stop"] = VoiceDaemonStopTool(self.config)
        self.tools["voice_daemon_stop"].set_voice_daemon(self.voice_daemon)
        
        self.tools["voice_daemon_skip"] = VoiceDaemonSkipTool(self.config)
        self.tools["voice_daemon_skip"].set_voice_daemon(self.voice_daemon)
            
        if "weather" in enabled:
            from tools.weather_tool import WeatherTool
            self.tools["weather"] = WeatherTool(self.config)
            
        if "execute_command" in enabled:
            from tools.execute_tool import ExecuteTool
            self.tools["execute_command"] = ExecuteTool(self.config)
            
        if "read_file" in enabled:
            from tools.file_tool import ReadFileTool
            self.tools["read_file"] = ReadFileTool(self.config)
            
        if "write_file" in enabled:
            from tools.file_tool import WriteFileTool
            self.tools["write_file"] = WriteFileTool(self.config)
            
        if "list_directory" in enabled:
            from tools.file_tool import ListDirectoryTool
            self.tools["list_directory"] = ListDirectoryTool(self.config)
            
        if "wake_word" in enabled:
            from tools.wakeword_tool import WakeWordTool
            self.tools["wake_word"] = WakeWordTool(self.config)
            
        if "voice_activity" in enabled:
            from tools.wakeword_tool import VoiceActivityTool
            self.tools["voice_activity"] = VoiceActivityTool(self.config)
            
        if "timer" in enabled:
            from tools.utility_tool import TimerTool
            self.tools["timer"] = TimerTool(self.config)
            
        if "calculator" in enabled:
            from tools.utility_tool import CalculatorTool
            self.tools["calculator"] = CalculatorTool(self.config)
            
        if "web_search" in enabled:
            from tools.web_search_tool import WebSearchTool
            self.tools["web_search"] = WebSearchTool(self.config)
            
        if "web_news" in enabled:
            from tools.web_search_tool import WebNewsTool
            self.tools["web_news"] = WebNewsTool(self.config)
            
        if "datetime" in enabled:
            from tools.datetime_tool import DateTimeTool
            self.tools["datetime"] = DateTimeTool(self.config)
        
        # Register Session Memory tools
        # These allow the LLM to query conversation history and context
        from tools.memory_tools import (
            SearchSessionMemoryTool, GetRecentAttachmentsTool,
            GetAttachmentContentTool, GetSessionContextTool, GetLastUserRequestTool
        )
        from core.session_memory import get_session_memory
        
        self.session_memory = get_session_memory()
        
        self.tools["search_session_memory"] = SearchSessionMemoryTool(self.config)
        self.tools["search_session_memory"].set_session_memory(self.session_memory)
        
        self.tools["get_recent_attachments"] = GetRecentAttachmentsTool(self.config)
        self.tools["get_recent_attachments"].set_session_memory(self.session_memory)
        
        self.tools["get_attachment_content"] = GetAttachmentContentTool(self.config)
        self.tools["get_attachment_content"].set_session_memory(self.session_memory)
        
        self.tools["get_session_context"] = GetSessionContextTool(self.config)
        self.tools["get_session_context"].set_session_memory(self.session_memory)
        
        self.tools["get_last_user_request"] = GetLastUserRequestTool(self.config)
        self.tools["get_last_user_request"].set_session_memory(self.session_memory)
    
    def list_tools(self) -> List[Tool]:
        """Return list of available tools for MCP protocol."""
        tools = []
        for name, tool in self.tools.items():
            tools.append(Tool(
                name=name,
                description=tool.description,
                inputSchema=tool.input_schema
            ))
        return tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Execute a tool with given arguments."""
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        
        tool = self.tools[name]
        result = await tool.execute(**arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    async def initialize(self):
        """Initialize all tools (for web interface compatibility)."""
        # Initialize voice daemon first
        if self.voice_daemon:
            self.voice_daemon.start()
            print("✅ Voice daemon started")
        
        # Initialize all tools
        for name, tool in self.tools.items():
            if hasattr(tool, 'initialize'):
                try:
                    await tool.initialize()
                except Exception as e:
                    print(f"Warning: Failed to initialize tool {name}: {e}")
    
    async def run(self):
        """Start the MCP server."""
        @self.server.list_tools()
        async def handle_list_tools():
            return self.list_tools()
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            return await self.call_tool(name, arguments)
        
        await self.server.run_stdio_async()
