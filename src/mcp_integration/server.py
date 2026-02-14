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


class TalkieMCPServer:
    """MCP Server that exposes all assistant capabilities as tools."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.server = Server(self.config["mcp"]["server_name"])
        self.tools = {}
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
