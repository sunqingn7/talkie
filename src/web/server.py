"""
Talkie Web Control Panel
FastAPI-based web interface for the voice assistant
"""

import asyncio
import json
import sys
import os
from typing import Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcp_integration.server import TalkieMCPServer
from core.llm_client import LLMClient
from core.model_manager import get_model_manager
import yaml

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)


class WebTalkieInterface:
    """Web interface for Talkie voice assistant."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.mcp_server: Optional[TalkieMCPServer] = None
        self.llm_client: Optional[LLMClient] = None
        self.conversation_history: List[dict] = []
        self.manager = ConnectionManager()
        self.model_manager = get_model_manager("config/models.yaml")
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', config_path)
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def save_config(self):
        """Save current config to settings.yaml."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'settings.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            return True
        except Exception as e:
            print(f"⚠️  Failed to save config: {e}")
            return False
    
    async def initialize(self):
        """Initialize MCP server and LLM client."""
        print("🚀 Initializing Talkie Web Interface...")
        
        # Initialize MCP server - pass config_path, not config dict
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'settings.yaml')
        self.mcp_server = TalkieMCPServer(config_path)
        await self.mcp_server.initialize()
        
        # Initialize LLM client - pass config_path
        self.llm_client = LLMClient(config_path)
        
        print("✅ Web interface ready!")
        
    async def process_message(self, user_message: str) -> dict:
        """Process a user message and return response."""
        if not self.mcp_server or not self.llm_client:
            return {
                "type": "error",
                "content": "System not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Build context messages
            messages = self._build_context_messages(user_message)
            
            # Format tools for LLM - exclude 'speak' tool as web interface handles TTS
            tools_dict = {k: v for k, v in self.mcp_server.tools.items() if k != 'speak'}
            tools = self.llm_client.format_tools_for_llm(tools_dict)
            
            # Get LLM response
            response = self.llm_client.chat_completion(messages, tools=tools)
            
            if "error" in response and "choices" not in response:
                return {
                    "type": "error",
                    "content": f"LLM Error: {response['error']}",
                    "timestamp": datetime.now().isoformat()
                }
            
            message = response["choices"][0]["message"]
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            
            # Handle tool calls
            if tool_calls:
                return await self._handle_tool_calls(tool_calls, messages, user_message)
            elif content:
                # Update conversation history
                self._add_to_history(user_message, content)
                
                return {
                    "type": "assistant_message",
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "type": "assistant_message",
                    "content": "I didn't understand that. Could you please rephrase?",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "type": "error",
                "content": f"Error processing message: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _handle_tool_calls(self, tool_calls: list, messages: list, original_input: str) -> dict:
        """Execute tool calls and return final response."""
        tool_results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            
            try:
                result = await self.mcp_server.call_tool(tool_name, tool_args)
                result_text = result[0].text if result else "{}"
                
                tool_results.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result_text
                })
                
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result_text
                })
                
            except Exception as e:
                error_msg = json.dumps({"error": str(e)})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": error_msg
                })
        
        # Get final response after tool execution
        final_response = self.llm_client.chat_completion(messages)
        
        if "choices" in final_response:
            final_content = final_response["choices"][0]["message"].get("content", "")
            self._add_to_history(original_input, final_content, tool_results)
            
            return {
                "type": "assistant_message",
                "content": final_content,
                "tool_calls": tool_results,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "type": "error",
                "content": "Failed to get final response",
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_context_messages(self, user_input: str) -> list:
        """Build context messages with conversation history."""
        messages = []
        
        # Add system prompt
        system_prompt = self.config.get('llm', {}).get('system_prompt', '')
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add recent history (last 10 exchanges)
        recent_history = self.conversation_history[-20:] if len(self.conversation_history) <= 20 else self.conversation_history[-20:]
        messages.extend(recent_history)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _add_to_history(self, user_input: str, assistant_response: str, tool_results=None):
        """Add exchange to conversation history."""
        import time
        
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        
        assistant_msg = {
            "role": "assistant",
            "content": assistant_response,
            "timestamp": time.time()
        }
        if tool_results:
            assistant_msg["tool_results"] = tool_results
        
        self.conversation_history.append(assistant_msg)
        
        # Trim history if too long
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-40:]
    
    def get_system_status(self) -> dict:
        """Get current system status."""
        # Get actually running LLM model, not just config
        running_model = self.model_manager.get_running_model()
        if running_model and running_model.get("model_path"):
            # Extract model name from path
            model_path = running_model["model_path"]
            llm_model = os.path.basename(model_path) if "/" in model_path else model_path
            # Also try to get friendly name from model manager
            available_models = self.model_manager.scan_available_models()
            for model in available_models:
                if model.get("path") == model_path:
                    llm_model = model.get("name", llm_model)
                    break
        else:
            llm_model = "Not running"
        
        return {
            "type": "system_status",
            "mcp_server_ready": self.mcp_server is not None,
            "llm_client_ready": self.llm_client is not None,
            "available_tools": list(self.mcp_server.tools.keys()) if self.mcp_server else [],
            "conversation_count": len(self.conversation_history) // 2,
            "config": {
                "tts_engine": self.config.get('tts', {}).get('engine', 'unknown'),
                "tts_model": self.config.get('tts', {}).get('coqui_model', 'unknown'),
                "llm_model": llm_model,
                "stt_model": self.config.get('stt', {}).get('model', 'unknown'),
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def get_available_models(self) -> dict:
        """Get list of available TTS and LLM models."""
        # TTS Engines
        tts_engines = [
            {"id": "edge_tts", "name": "Edge TTS (Microsoft)", "type": "online", "description": "High quality, fast, requires internet"},
            {"id": "coqui", "name": "Coqui TTS (Local)", "type": "local", "description": "Offline, customizable voices, large models"},
            {"id": "pyttsx3", "name": "pyttsx3 (Fallback)", "type": "local", "description": "Basic offline TTS"},
        ]
        
        # Coqui TTS Models
        tts_models = [
            {"id": "tts_models/en/ljspeech/tacotron2-DDC", "name": "English (Tacotron2)", "language": "en", "size": "113MB"},
            {"id": "tts_models/zh-CN/baker/tacotron2-DDC-GST", "name": "Chinese (Baker)", "language": "zh", "size": "686MB"},
            {"id": "tts_models/multilingual/multi-dataset/xtts_v2", "name": "XTTS-v2 (17 languages)", "language": "multilingual", "size": "1.5GB", "requires_license": True},
        ]
        
        # Get LLM models from model manager
        llm_models = self.model_manager.scan_available_models()
        running_model = self.model_manager.get_running_model()
        
        # Get TTS info
        tts_speakers = []
        current_speaker = "default"
        current_engine = "unknown"
        edge_voices = []
        current_edge_voice = "default"
        
        if self.mcp_server and 'speak' in self.mcp_server.tools:
            tts_tool = self.mcp_server.tools['speak']
            current_engine = tts_tool.get_current_engine()
            
            # Get speakers/voices based on current engine
            if current_engine == "edge_tts":
                # For Edge TTS, get voices from Edge TTS tool
                if hasattr(tts_tool, 'edge_tts_tool') and tts_tool.edge_tts_tool:
                    edge_voices = tts_tool.edge_tts_tool.get_available_voices()
                    current_edge_voice = tts_tool.edge_tts_tool.get_current_voice()
                    # Also populate tts_speakers with Edge voices for compatibility
                    tts_speakers = [{"id": v["id"], "name": v["name"], "gender": v["gender"], "locale": v["locale"], "type": "edge"} for v in edge_voices]
                else:
                    # Edge TTS not initialized yet, return default voices from class
                    try:
                        from tools.edge_tts_tool import EdgeTTSTool
                        edge_voices = EdgeTTSTool.AVAILABLE_VOICES
                        current_edge_voice = self.config.get('tts', {}).get('edge_voice', 'en-US-AriaNeural')
                        tts_speakers = [{"id": v["id"], "name": v["name"], "gender": v["gender"], "locale": v["locale"], "type": "edge"} for v in edge_voices]
                    except Exception as e:
                        print(f"⚠️  Failed to load Edge TTS voices: {e}")
            else:
                # For Coqui/pyttsx3, get speakers
                tts_speakers = tts_tool.get_available_speakers()
                current_speaker = tts_tool.get_current_speaker()
        
        return {
            "type": "available_models",
            "tts_engines": tts_engines,
            "current_tts_engine": current_engine,
            "tts_models": tts_models,
            "current_tts_model": self.config.get('tts', {}).get('coqui_model', 'unknown'),
            "tts_speakers": tts_speakers,
            "current_tts_speaker": current_speaker,
            "edge_voices": edge_voices,
            "current_edge_voice": current_edge_voice,
            "llm_models": llm_models,
            "current_llm_model": running_model.get("model_path") if running_model else None,
            "llm_server_running": running_model is not None,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_llm_status(self) -> dict:
        """Get LLM server status."""
        running_model = self.model_manager.get_running_model()
        available_models = self.model_manager.scan_available_models()
        
        return {
            "type": "llm_status",
            "running": running_model is not None,
            "current_model": running_model.get("model_path") if running_model else None,
            "pid": running_model.get("pid") if running_model else None,
            "available_models": [m["id"] for m in available_models if m.get("exists")],
            "timestamp": datetime.now().isoformat()
        }
    
    async def switch_llm_model(self, model_id: str) -> dict:
        """Switch LLM model by restarting llama-server."""
        try:
            result = self.model_manager.switch_model(model_id)
            
            if result["success"]:
                # Reinitialize LLM client to connect to new server
                config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'settings.yaml')
                self.llm_client = LLMClient(config_path)
            
            return {
                "type": "llm_model_switched",
                "success": result["success"],
                "message": result["message"],
                "model_id": model_id if result["success"] else None,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "type": "error",
                "success": False,
                "message": f"Failed to switch LLM model: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def switch_tts_model(self, model_id: str) -> dict:
        """Switch TTS model."""
        try:
            # Update config
            self.config['tts']['coqui_model'] = model_id
            
            # Reinitialize TTS tool
            if self.mcp_server and 'speak' in self.mcp_server.tools:
                from tools.tts_tool import TTSTool
                self.mcp_server.tools['speak'] = TTSTool(self.config)
                await self.mcp_server.tools['speak'].initialize()
            
            return {
                "type": "model_switched",
                "success": True,
                "message": f"TTS model switched to {model_id}",
                "new_model": model_id,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "type": "error",
                "success": False,
                "message": f"Failed to switch model: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def switch_tts_engine(self, engine_id: str) -> dict:
        """Switch TTS engine (edge_tts, coqui, pyttsx3)."""
        try:
            if self.mcp_server and 'speak' in self.mcp_server.tools:
                tts_tool = self.mcp_server.tools['speak']
                result = tts_tool.switch_engine(engine_id)

                if result.get("success"):
                    actual_engine = result.get("engine", engine_id)

                    # Update config with the actual engine being used
                    self.config['tts']['engine'] = actual_engine

                    # Save config to file
                    self.save_config()

                    # Build message
                    message = f"TTS engine switched to {actual_engine}"
                    if result.get("warning"):
                        message += f" ({result['warning']})"

                    return {
                        "type": "tts_engine_switched",
                        "success": True,
                        "message": message,
                        "new_engine": actual_engine,
                        "requested_engine": engine_id,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "type": "error",
                        "success": False,
                        "message": f"Failed to switch TTS engine: {result.get('error', engine_id)}",
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                return {
                    "type": "error",
                    "success": False,
                    "message": "TTS tool not available",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "type": "error",
                "success": False,
                "message": f"Failed to switch TTS engine: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def switch_edge_voice(self, voice_id: str) -> dict:
        """Switch Edge TTS voice."""
        try:
            if self.mcp_server and 'speak' in self.mcp_server.tools:
                tts_tool = self.mcp_server.tools['speak']
                if hasattr(tts_tool, 'edge_tts_tool') and tts_tool.edge_tts_tool:
                    success = tts_tool.edge_tts_tool.set_voice(voice_id)
                    
                    if success:
                        # Update config
                        self.config['tts']['edge_voice'] = voice_id
                        
                        # Save config to file
                        self.save_config()
                        
                        return {
                            "type": "edge_voice_switched",
                            "success": True,
                            "message": f"Edge TTS voice switched to {voice_id}",
                            "new_voice": voice_id,
                            "timestamp": datetime.now().isoformat()
                        }
                
                return {
                    "type": "error",
                    "success": False,
                    "message": "Edge TTS not available",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "type": "error",
                    "success": False,
                    "message": "TTS tool not available",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "type": "error",
                "success": False,
                "message": f"Failed to switch Edge voice: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_tts_speakers(self) -> dict:
        """Get available TTS speakers/personas."""
        speakers = []
        current_speaker = "default"
        current_engine = "unknown"
        edge_voices = []
        current_edge_voice = "default"
        
        if self.mcp_server and 'speak' in self.mcp_server.tools:
            tts_tool = self.mcp_server.tools['speak']
            current_engine = tts_tool.get_current_engine()
            speakers = tts_tool.get_available_speakers()
            current_speaker = tts_tool.get_current_speaker()
            
            # Get Edge TTS info if that's the current engine
            if current_engine == "edge_tts":
                if hasattr(tts_tool, 'edge_tts_tool') and tts_tool.edge_tts_tool:
                    edge_voices = tts_tool.edge_tts_tool.get_available_voices()
                    current_edge_voice = tts_tool.edge_tts_tool.get_current_voice()
                else:
                    # Edge TTS not initialized yet
                    try:
                        from tools.edge_tts_tool import EdgeTTSTool
                        edge_voices = EdgeTTSTool.AVAILABLE_VOICES
                        current_edge_voice = self.config.get('tts', {}).get('edge_voice', 'en-US-AriaNeural')
                    except Exception as e:
                        print(f"⚠️  Failed to load Edge TTS voices: {e}")
        
        return {
            "type": "tts_speakers",
            "speakers": speakers,
            "current_speaker": current_speaker,
            "current_tts_engine": current_engine,
            "edge_voices": edge_voices,
            "current_edge_voice": current_edge_voice,
            "count": len(speakers),
            "timestamp": datetime.now().isoformat()
        }
    
    async def switch_tts_speaker(self, speaker_id: str) -> dict:
        """Switch TTS speaker/persona."""
        try:
            if self.mcp_server and 'speak' in self.mcp_server.tools:
                tts_tool = self.mcp_server.tools['speak']
                success = tts_tool.set_speaker(speaker_id)
                
                if success:
                    # Save config to file (for Coqui TTS, speaker is stored in config)
                    self.save_config()
                    
                    return {
                        "type": "speaker_switched",
                        "success": True,
                        "message": f"TTS speaker switched to {speaker_id}",
                        "new_speaker": speaker_id,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "type": "error",
                        "success": False,
                        "message": f"Failed to switch speaker: {speaker_id}",
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                return {
                    "type": "error",
                    "success": False,
                    "message": "TTS tool not available",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "type": "error",
                "success": False,
                "message": f"Failed to switch speaker: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def test_tts_speaker(self, speaker_id: str = None) -> dict:
        """Test TTS with current or specified speaker."""
        try:
            if self.mcp_server and 'speak' in self.mcp_server.tools:
                # Test message
                test_text = "Hello! This is a test of the text-to-speech system."
                
                # Use specified speaker or current
                result = await self.mcp_server.tools['speak'].execute(
                    text=test_text,
                    speaker_id=speaker_id
                )
                
                return {
                    "type": "tts_test_result",
                    "success": result.get('success', False),
                    "message": "TTS test completed" if result.get('success') else "TTS test failed",
                    "speaker": speaker_id or "default",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "type": "error",
                    "success": False,
                    "message": "TTS tool not available",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "type": "error",
                "success": False,
                "message": f"TTS test failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def demo_tts_speaker(self, speaker_id: str = None, demo_text: str = None) -> dict:
        """Demo TTS with custom text for the selected speaker."""
        try:
            if self.mcp_server and 'speak' in self.mcp_server.tools:
                # Use provided text or default greeting
                text = demo_text or f"Hello, this is the selected voice"
                
                # Use specified speaker
                result = await self.mcp_server.tools['speak'].execute(
                    text=text,
                    speaker_id=speaker_id
                )
                
                return {
                    "type": "tts_demo_result",
                    "success": result.get('success', False),
                    "message": "Voice demo played" if result.get('success') else "Voice demo failed",
                    "speaker": speaker_id or "default",
                    "text": text,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "type": "error",
                    "success": False,
                    "message": "TTS tool not available",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "type": "error",
                "success": False,
                "message": f"Voice demo failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def speak_assistant_response(self, text: str) -> dict:
        """Speak assistant response using TTS with current speaker."""
        try:
            if self.mcp_server and 'speak' in self.mcp_server.tools:
                # Get current speaker from config
                tts_tool = self.mcp_server.tools['speak']
                current_speaker = tts_tool.get_current_speaker()
                
                # Truncate long text for TTS
                max_chars = 500
                speak_text = text[:max_chars] if len(text) > max_chars else text
                
                result = await tts_tool.execute(
                    text=speak_text,
                    speaker_id=current_speaker
                )
                
                return {
                    "type": "tts_spoken",
                    "success": result.get('success', False),
                    "speaker": current_speaker,
                    "characters": len(speak_text),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "type": "error",
                    "success": False,
                    "message": "TTS tool not available",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "type": "error",
                "success": False,
                "message": f"TTS failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }


# Create global interface instance
web_interface = WebTalkieInterface()
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    # Startup
    await web_interface.initialize()
    yield
    # Shutdown
    pass


# Create FastAPI app
app = FastAPI(
    title="Talkie Web Control Panel",
    description="Web interface for Talkie Voice Assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Setup templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def get_status():
    """Get system status."""
    return web_interface.get_system_status()


@app.get("/api/models")
async def get_models():
    """Get available models."""
    return web_interface.get_available_models()


@app.post("/api/switch-model")
async def switch_model(data: dict):
    """Switch TTS model."""
    model_id = data.get("model_id")
    if not model_id:
        return {"type": "error", "message": "No model_id provided"}
    return await web_interface.switch_tts_model(model_id)


@app.get("/api/tts-speakers")
async def get_tts_speakers():
    """Get available TTS speakers."""
    return web_interface.get_tts_speakers()


@app.post("/api/tts-switch-speaker")
async def switch_tts_speaker(data: dict):
    """Switch TTS speaker."""
    speaker_id = data.get("speaker_id")
    if not speaker_id:
        return {"type": "error", "message": "No speaker_id provided"}
    return await web_interface.switch_tts_speaker(speaker_id)


@app.post("/api/tts-test")
async def test_tts(data: dict = None):
    """Test TTS with current or specified speaker."""
    speaker_id = data.get("speaker_id") if data else None
    return await web_interface.test_tts_speaker(speaker_id)


@app.post("/api/tts-switch-engine")
async def switch_tts_engine(data: dict):
    """Switch TTS engine (edge_tts, coqui, pyttsx3)."""
    engine_id = data.get("engine_id")
    if not engine_id:
        return {"type": "error", "message": "No engine_id provided"}
    return await web_interface.switch_tts_engine(engine_id)


@app.post("/api/tts-switch-edge-voice")
async def switch_edge_voice(data: dict):
    """Switch Edge TTS voice."""
    voice_id = data.get("voice_id")
    if not voice_id:
        return {"type": "error", "message": "No voice_id provided"}
    return await web_interface.switch_edge_voice(voice_id)


@app.get("/api/llm-status")
async def get_llm_status():
    """Get LLM server status."""
    return web_interface.get_llm_status()


@app.post("/api/llm-switch")
async def switch_llm_model(data: dict):
    """Switch LLM model."""
    model_id = data.get("model_id")
    if not model_id:
        return {"type": "error", "message": "No model_id provided"}
    return await web_interface.switch_llm_model(model_id)


@app.post("/api/llm-restart")
async def restart_llm_server():
    """Restart LLM server with current model."""
    result = web_interface.model_manager.restart_server()
    return {
        "type": "llm_server_restarted",
        "success": result["success"],
        "message": result["message"],
        "timestamp": datetime.now().isoformat()
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await manager.connect(websocket)
    
    # Send initial status
    await websocket.send_json(web_interface.get_system_status())
    await websocket.send_json(web_interface.get_available_models())
    await websocket.send_json(web_interface.get_tts_speakers())
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "user_message":
                content = data.get("content", "")
                
                # Send thinking indicator
                await websocket.send_json({
                    "type": "thinking",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Process message
                response = await web_interface.process_message(content)
                await websocket.send_json(response)
                
            elif message_type == "get_status":
                await websocket.send_json(web_interface.get_system_status())
                
            elif message_type == "get_models":
                await websocket.send_json(web_interface.get_available_models())
                
            elif message_type == "switch_model":
                model_id = data.get("model_id")
                if model_id:
                    result = await web_interface.switch_tts_model(model_id)
                    await websocket.send_json(result)
            
            elif message_type == "get_tts_speakers":
                await websocket.send_json(web_interface.get_tts_speakers())
            
            elif message_type == "switch_tts_speaker":
                speaker_id = data.get("speaker_id")
                if speaker_id:
                    result = await web_interface.switch_tts_speaker(speaker_id)
                    await websocket.send_json(result)
            
            elif message_type == "test_tts":
                speaker_id = data.get("speaker_id") if data else None
                result = await web_interface.test_tts_speaker(speaker_id)
                await websocket.send_json(result)
            
            elif message_type == "switch_tts_engine":
                engine_id = data.get("engine_id")
                if engine_id:
                    result = await web_interface.switch_tts_engine(engine_id)
                    await websocket.send_json(result)
                    # Send updated models info
                    await websocket.send_json(web_interface.get_available_models())
            
            elif message_type == "switch_edge_voice":
                voice_id = data.get("voice_id")
                if voice_id:
                    result = await web_interface.switch_edge_voice(voice_id)
                    await websocket.send_json(result)
            
            elif message_type == "demo_tts":
                speaker_id = data.get("speaker_id") if data else None
                demo_text = data.get("text", f"Hello, this is the selected voice")
                result = await web_interface.demo_tts_speaker(speaker_id, demo_text)
                await websocket.send_json(result)
            
            elif message_type == "speak_assistant_response":
                text = data.get("text", "")
                if text:
                    result = await web_interface.speak_assistant_response(text)
                    await websocket.send_json(result)
            
            elif message_type == "get_llm_status":
                await websocket.send_json(web_interface.get_llm_status())
            
            elif message_type == "switch_llm_model":
                model_id = data.get("model_id")
                if model_id:
                    # Send switching notification
                    await websocket.send_json({
                        "type": "llm_switching",
                        "message": f"Switching to model: {model_id}...",
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    result = await web_interface.switch_llm_model(model_id)
                    await websocket.send_json(result)
                    
                    # Send updated status
                    await websocket.send_json(web_interface.get_llm_status())
            
            elif message_type == "restart_llm_server":
                await websocket.send_json({
                    "type": "llm_restarting",
                    "message": "Restarting LLM server...",
                    "timestamp": datetime.now().isoformat()
                })
                
                result = web_interface.model_manager.restart_server()
                await websocket.send_json({
                    "type": "llm_server_restarted",
                    "success": result["success"],
                    "message": result["message"],
                    "timestamp": datetime.now().isoformat()
                })
                await websocket.send_json(web_interface.get_llm_status())
                    
            elif message_type == "clear_history":
                web_interface.conversation_history.clear()
                await websocket.send_json({
                    "type": "history_cleared",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


def run_web_server(host: str = "0.0.0.0", port: int = 8082):
    """Run the web server."""
    print(f"""
🌐 Talkie Web Control Panel
===========================
Server starting on http://{host}:{port}

Features:
- Real-time chat interface
- Voice/text input
- Model switching
- System monitoring
- Conversation history

Press Ctrl+C to stop
""")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_web_server()
