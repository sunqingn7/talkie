"""
Talkie Web Control Panel
FastAPI-based web interface for the voice assistant
"""

import asyncio
import json
import sys
import os
import tempfile
from typing import Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcp_integration.server import TalkieMCPServer
from core.llm_client import LLMClient
from core.model_manager import get_model_manager
from core.session_memory import get_session_memory, SessionMemory
from tools.file_attachment_tool import FileAttachmentTool
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
        self.file_attachment_tool = None
        self.pending_attachments: List[dict] = []  # Store uploaded files waiting to be attached
        
        # Initialize session memory for persistent conversation tracking
        self.session_memory: SessionMemory = get_session_memory()
        print(f"[SessionMemory] Initialized with session: {self.session_memory.session_id}")
        
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
        
        # Initialize MCP server - pass config_path and session_memory to ensure shared instance
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'settings.yaml')
        print(f"[Web Server] Passing session memory to MCP server: {self.session_memory.session_id}")
        self.mcp_server = TalkieMCPServer(config_path, session_memory=self.session_memory)
        await self.mcp_server.initialize()
        
        # Initialize LLM client - pass config_path
        self.llm_client = LLMClient(config_path)
        
        # Initialize file attachment tool
        upload_config = self.config.get('upload', {})
        upload_config['upload_dir'] = upload_config.get('upload_dir', os.path.join(tempfile.gettempdir(), 'talkie_uploads'))
        self.file_attachment_tool = FileAttachmentTool(upload_config)
        
        print("✅ Web interface ready!")
        
    async def upload_file(self, file_content: bytes, filename: str, transcribe: bool = True) -> dict:
        """Upload and process a file, returning file info for attachment."""
        print(f"📎 Processing upload: {filename} ({len(file_content)} bytes)")
        
        if not self.file_attachment_tool:
            print("   ❌ File attachment tool not initialized!")
            return {
                "success": False,
                "error": "File attachment tool not initialized. Please refresh the page.",
                "filename": filename
            }
        
        try:
            # Process the uploaded file
            result = await self.file_attachment_tool.process_upload(
                file_content, filename, transcribe
            )
            
            if result.get("success"):
                # Store as pending attachment
                attachment_info = {
                    "filename": filename,
                    "file_type": result["metadata"]["file_type"],
                    "content": result.get("content", ""),
                    "saved_path": result.get("saved_path"),
                    "size": result["metadata"]["size_bytes"],
                    "uploaded_at": datetime.now().isoformat()
                }
                self.pending_attachments.append(attachment_info)
                
                # Limit pending attachments to avoid memory issues
                if len(self.pending_attachments) > 10:
                    self.pending_attachments = self.pending_attachments[-10:]
                
                # Safely get content preview
                content = result.get("content") or ""
                content_preview = content[:500] + "..." if len(content) > 500 else content
                
                # Record attachment in session memory
                attachment_memory_id = self.session_memory.record_attachment(
                    filename=filename,
                    file_type=result["metadata"]["file_type"],
                    content=content,
                    file_path=result.get("saved_path"),
                    metadata={
                        "size_bytes": result["metadata"]["size_bytes"],
                        "pending_attachment_id": len(self.pending_attachments) - 1
                    }
                )
                print(f"   ✅ Upload processed: {filename} ({result['metadata']['file_type']}) [Memory ID: {attachment_memory_id}]")
                return {
                    "success": True,
                    "filename": filename,
                    "file_type": result["metadata"]["file_type"],
                    "content_preview": content_preview,
                    "attachment_id": len(self.pending_attachments) - 1
                }
            else:
                print(f"   ❌ Upload failed: {filename} - {result.get('error')}")
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "filename": filename
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Upload failed: {str(e)}",
                "filename": filename
            }
    
    async def process_message(self, user_message: str, attachment_ids: Optional[List[int]] = None) -> dict:
        """Process a user message and return response."""
        if not self.mcp_server or not self.llm_client:
            return {
                "type": "error",
                "content": "System not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Record user message in session memory
            self.session_memory.record_message(
                role="user",
                content=user_message,
                metadata={"attachment_ids": attachment_ids} if attachment_ids else {}
            )
            
            # Build context messages
            messages = self._build_context_messages(user_message)
            
            # Check if user is asking to read the file aloud
            user_wants_reading = any(keyword in user_message.lower() for keyword in
                                     ['read', 'read aloud', 'read this', 'read it', 'speak', 'narrate'])

            # Check if user is referencing a previously uploaded file without attaching it again
            file_reference_keywords = ['the file i uploaded', 'that file', 'the document', 'that document',
                                       'file i just uploaded', 'uploaded file', 'the pdf', 'the txt']
            user_references_past_file = any(keyword in user_message.lower() for keyword in file_reference_keywords)

            # If user references past file but didn't attach it, inject recent attachments info
            if user_references_past_file and not attachment_ids:
                recent_attachments = self.session_memory.get_recent_attachments(3)
                if recent_attachments:
                    # Add context about recent files to help LLM
                    files_info = "\n\n[SYSTEM NOTE: User may be referring to one of these recently uploaded files:\n"
                    for i, att in enumerate(recent_attachments, 1):
                        files_info += f"{i}. {att['filename']} ({att['file_type']}, uploaded at {att['datetime']})\n"
                    files_info += "Use get_attachment_content tool with the appropriate file to access its content.]"

                    # Append to user's message
                    last_message = messages[-1]
                    if last_message["role"] == "user":
                        last_message["content"] += files_info
                        print(f"[SessionMemory] Injected recent attachments info for reference: {[a['filename'] for a in recent_attachments]}")

            # Add attachment content to the last user message if attachments are specified
            if attachment_ids and self.pending_attachments:
                attachment_content = []
                for idx in attachment_ids:
                    if 0 <= idx < len(self.pending_attachments):
                        attachment = self.pending_attachments[idx]
                        
                        # Check if user is asking to read this file
                        is_reading_request = user_wants_reading and idx == attachment_ids[0]
                        
                        if is_reading_request:
                            # For reading requests, only show file info, not full content in chat
                            file_info = f"\n\n[File attached: {attachment['filename']} ({attachment['file_type']})]\n"
                            file_info += f"The file content is available for reading. User wants me to read it aloud."
                            # Store full content for the tool but don't display it
                            attachment_content.append(file_info)
                            # Add content in a way that's only visible to LLM, not in chat display
                            attachment['_full_content'] = attachment.get('content', '')
                        else:
                            # For analysis/summary requests, include content normally
                            file_info = f"\n\n--- Attached File: {attachment['filename']} ---\n"
                            file_info += f"Type: {attachment['file_type']}\n"
                            if attachment.get('content'):
                                file_info += f"Content:\n{attachment['content'][:8000]}"
                                if len(attachment['content']) > 8000:
                                    file_info += "\n[Content truncated for length]"
                            attachment_content.append(file_info)
                
                if attachment_content:
                    # Append attachment content to the last user message
                    last_message = messages[-1]
                    if last_message["role"] == "user":
                        last_message["content"] += "\n\n" + "\n".join(attachment_content)
                        
                        # If reading was requested, add content as a system message for LLM only
                        if user_wants_reading and self.pending_attachments:
                            for idx in attachment_ids:
                                if 0 <= idx < len(self.pending_attachments):
                                    attachment = self.pending_attachments[idx]
                                    if attachment.get('_full_content'):
                                        # Insert before last message (so LLM sees it but it's not in visible chat)
                                        messages.insert(-1, {
                                            "role": "system",
                                            "content": f"Full content of {attachment['filename']}:\n{attachment['_full_content'][:10000]}"
                                        })
            
            # Format tools for LLM - exclude 'speak' tool as web interface handles TTS
            tools_dict = {k: v for k, v in self.mcp_server.tools.items() if k != 'speak'}
            tools = self.llm_client.format_tools_for_llm(tools_dict)

            # Debug: Log available tools
            print(f"[Tools Available] {len(tools)} tools: {[t['function']['name'] for t in tools]}")

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
                # Check if user wanted reading but LLM only responded with text (no tool call)
                # This can happen if LLM generates acknowledgment but forgets to call the tool
                if user_wants_reading and attachment_ids and self.pending_attachments:
                    # Check if LLM response indicates reading intent
                    reading_acknowledged = any(phrase in content.lower() for phrase in 
                                               ['read this', 'reading', 'will read', 'start reading', 
                                                'narrate', 'speak aloud', 'read it'])
                    if reading_acknowledged:
                        # Auto-trigger reading with the first attachment
                        first_attachment = self.pending_attachments[attachment_ids[0]]
                        if first_attachment.get('_full_content') or first_attachment.get('content'):
                            # Start reading in background
                            reading_content = first_attachment.get('_full_content') or first_attachment.get('content')
                            if reading_content:
                                asyncio.create_task(self.read_file_aloud(
                                    content=reading_content,
                                    start_paragraph=1,
                                    language="auto"
                                ))
                            # Update the content to indicate reading has started
                            content += "\n\n📖 *Reading started... Say 'stop reading' to pause.*"
                
                # Update conversation history
                self._add_to_history(user_message, content)
                
                # Record assistant response in session memory
                self.session_memory.record_message(
                    role="assistant",
                    content=content,
                    metadata={"auto_reading_triggered": user_wants_reading and attachment_ids is not None}
                )
                
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

        print(f"[Tool Calls] Processing {len(tool_calls)} tool call(s):")
        for i, tc in enumerate(tool_calls, 1):
            print(f"  {i}. {tc['function']['name']}")

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            print(f"[Tool Call] Executing: {tool_name}({json.dumps(tool_args)[:100]}...)")

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
            
            # Check if reading-related tools were called or intent was expressed
            reading_intent_phrases = ['read this', 'reading', 'will read', 'start reading',
                                      'narrate', 'speak aloud', 'read it', 'read aloud']
            has_reading_intent = any(phrase in final_content.lower() for phrase in reading_intent_phrases)
            has_called_read_tool = any(tr["tool"] == "read_file_aloud" for tr in tool_results)
            has_fetched_attachment = any(tr["tool"] == "get_attachment_content" for tr in tool_results)

            # If LLM fetched attachment content but didn't call read_file_aloud, auto-trigger reading
            if has_fetched_attachment and not has_called_read_tool:
                print(f"[Auto Read] LLM fetched attachment but didn't call read_file_aloud. Auto-triggering...")
                # Get the content from the tool result
                attachment_tool_result = next((tr for tr in tool_results if tr["tool"] == "get_attachment_content"), None)
                if attachment_tool_result:
                    try:
                        result_data = json.loads(attachment_tool_result["result"])
                        if result_data.get("success") and result_data.get("content"):
                            content = result_data["content"]
                            filename = result_data.get("attachment", {}).get("filename", "the file")
                            print(f"[Auto Read] Auto-reading fetched file: {filename}")
                            asyncio.create_task(self.read_file_aloud(
                                content=content,
                                start_paragraph=1,
                                language="auto"
                            ))
                            final_content += f"\n\n📖 *Now reading {filename}... Say 'stop reading' to pause.*"
                            has_called_read_tool = True  # Mark as called for skip_voice
                    except json.JSONDecodeError as e:
                        print(f"[Auto Read] Failed to parse attachment result: {e}")

            # If LLM expressed reading intent but didn't call the tool, auto-trigger reading (fallback)
            elif has_reading_intent and not has_called_read_tool:
                print(f"[Auto Read] LLM said it will read but didn't call read_file_aloud. Auto-triggering...")
                # Try to get most recent attachment and read it
                recent_attachments = self.session_memory.get_recent_attachments(1)
                if recent_attachments:
                    attachment = recent_attachments[0]
                    content = self.session_memory.get_attachment_content(attachment['id'])
                    if content:
                        print(f"[Auto Read] Auto-reading file: {attachment['filename']}")
                        asyncio.create_task(self.read_file_aloud(
                            content=content,
                            start_paragraph=1,
                            language="auto"
                        ))
                        final_content += f"\n\n📖 *Now reading {attachment['filename']}... Say 'stop reading' to pause.*"
                        has_called_read_tool = True  # Mark as called for skip_voice

            # If LLM called the tool but didn't say anything about reading, add a note
            if has_called_read_tool and not has_reading_intent and not has_fetched_attachment:
                print(f"[Auto Read] Tool was called but LLM didn't acknowledge. Adding note...")
                final_content += "\n\n📖 *Reading the file now... Say 'stop reading' to pause.*"
            
            self._add_to_history(original_input, final_content, tool_results)
            
            # Record assistant response with tool results in session memory
            self.session_memory.record_message(
                role="assistant",
                content=final_content,
                metadata={
                    "tool_calls": [tr["tool"] for tr in tool_results],
                    "tools_used_count": len(tool_results)
                }
            )
            
            # Determine if we should skip auto-speaking the response
            # Skip when read_file_aloud was called because the file content will be spoken instead
            skip_voice = has_called_read_tool
            print(f"[Assistant Response] skip_voice={skip_voice}, has_called_read_tool={has_called_read_tool}")

            return {
                "type": "assistant_message",
                "content": final_content,
                "tool_calls": tool_results,
                "skip_voice": skip_voice,  # Tell frontend not to auto-speak this
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
        
        # Add system prompt with memory capabilities
        system_prompt = self.config.get('llm', {}).get('system_prompt', '')
        
        # Add memory capabilities information
        memory_info = """
MEMORY CAPABILITIES:
You have access to session memory tools that allow you to recall previous conversations and files:

1. search_session_memory - Search chat history by keywords (e.g., "weather", "file", "read")
2. get_recent_attachments - List recently uploaded files when user refers to "the file I uploaded"
3. get_attachment_content - Retrieve full content of a previously uploaded file
4. get_session_context - Get overview of session (topics discussed, files available)
5. get_last_user_request - Retrieve user's previous request when they say "let's redo" or "again"
6. read_file_aloud - ACTUALLY READS FILE CONTENT ALOUD using text-to-speech

CRITICAL INSTRUCTION - When user asks you to READ A FILE ALOUD:
You MUST call TWO tools in sequence:
1. First: get_attachment_content to get the file content
2. Second: read_file_aloud with the content parameter to actually start reading

DO NOT just acknowledge that you will read it - you MUST call read_file_aloud tool.
The read_file_aloud tool will queue the content for text-to-speech playback.

Example workflow:
User: "Read the file I uploaded"
You: [call get_attachment_content to get the file content]
[Tool returns file content]
You: [call read_file_aloud with content="file content here"]

Use these tools when:
- User asks to "redo", "do that again", "repeat" - use get_last_user_request
- User refers to "the file I uploaded", "that document" - use get_recent_attachments or get_attachment_content
- User says "as I mentioned before" or references previous topics - use search_session_memory
- User asks about conversation history - use get_session_context
- User asks you to read a file aloud - use get_attachment_content THEN read_file_aloud

Always proactively use these tools when the user references past interactions.
"""
        
        full_system_prompt = f"{system_prompt}\n\n{memory_info}" if system_prompt else memory_info
        
        messages.append({
            "role": "system",
            "content": full_system_prompt
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
        # Get actually running LLM model
        running_model = self.model_manager.get_running_model()
        
        if running_model and running_model.get("model_path"):
            # Use model_name if available, otherwise extract from path
            if running_model.get("model_name"):
                llm_model = running_model["model_name"]
            else:
                model_path = running_model["model_path"]
                llm_model = os.path.basename(model_path) if "/" in model_path else model_path
            is_running = True
        else:
            llm_model = "Not running"
            is_running = False
        
        # Get voice daemon status
        voice_daemon_status = {}
        if self.mcp_server and self.mcp_server.voice_daemon:
            voice_daemon_status = self.mcp_server.voice_daemon.get_status()
        
        return {
            "type": "system_status",
            "mcp_server_ready": self.mcp_server is not None,
            "llm_client_ready": self.llm_client is not None,
            "llm_running": is_running,
            "available_tools": list(self.mcp_server.tools.keys()) if self.mcp_server else [],
            "conversation_count": len(self.conversation_history) // 2,
            "config": {
                "tts_engine": self.config.get('tts', {}).get('engine', 'unknown'),
                "tts_model": self.config.get('tts', {}).get('coqui_model', 'unknown'),
                "llm_model": llm_model,
                "stt_model": self.config.get('stt', {}).get('model', 'unknown'),
            },
            "voice_daemon": voice_daemon_status,
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
    
    async def read_file_aloud(self, content: str = None, file_path: str = None, start_paragraph: int = 1, language: str = "auto") -> dict:
        """Read file content aloud with paragraph-by-paragraph TTS."""
        try:
            if self.mcp_server and 'read_file_aloud' in self.mcp_server.tools:
                reader_tool = self.mcp_server.tools['read_file_aloud']
                result = await reader_tool.execute(
                    content=content,
                    file_path=file_path,
                    start_paragraph=start_paragraph,
                    language=language
                )
                
                return {
                    "type": "reading_started",
                    "success": result.get('success', False),
                    "message": result.get('message', ''),
                    "preview": result.get('preview', ''),
                    "total_paragraphs": result.get('total_paragraphs', 0),
                    "instruction": result.get('instruction', ''),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "type": "error",
                    "success": False,
                    "message": "File reader tool not available",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "type": "error",
                "success": False,
                "message": f"Failed to start reading: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def stop_reading(self) -> dict:
        """Stop the current file reading session."""
        try:
            if self.mcp_server and 'stop_reading' in self.mcp_server.tools:
                stop_tool = self.mcp_server.tools['stop_reading']
                result = await stop_tool.execute()
                
                return {
                    "type": "reading_stopped",
                    "success": result.get('success', False),
                    "message": result.get('message', ''),
                    "paragraphs_read": result.get('paragraphs_read', 0),
                    "total_paragraphs": result.get('total_paragraphs', 0),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "type": "error",
                    "success": False,
                    "message": "Stop reading tool not available",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "type": "error",
                "success": False,
                "message": f"Failed to stop reading: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_reading_status(self) -> dict:
        """Get current reading status."""
        try:
            if self.mcp_server and 'read_file_aloud' in self.mcp_server.tools:
                reader_tool = self.mcp_server.tools['read_file_aloud']
                status = reader_tool.get_reading_status()
                
                return {
                    "type": "reading_status",
                    "is_reading": status.get('is_reading', False),
                    "current_paragraph": status.get('current_paragraph', 0),
                    "total_paragraphs": status.get('total_paragraphs', 0),
                    "progress_percent": status.get('progress_percent', 0),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "type": "reading_status",
                    "is_reading": False,
                    "message": "Reader tool not available",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "type": "error",
                "success": False,
                "message": f"Failed to get reading status: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_session_memory_status(self) -> dict:
        """Get session memory status and summary."""
        try:
            summary = self.session_memory.get_session_summary()
            recent_attachments = self.session_memory.get_recent_attachments(5)
            
            return {
                "type": "session_memory_status",
                "session_id": summary['session_id'],
                "started_at": summary['started_at'],
                "message_count": summary['message_count'],
                "attachment_count": summary['attachment_count'],
                "recent_topics": summary.get('recent_topics', []),
                "recent_attachments": [
                    {
                        "id": a['id'],
                        "filename": a['filename'],
                        "type": a['file_type'],
                        "uploaded_at": a['datetime']
                    }
                    for a in recent_attachments
                ],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "type": "error",
                "success": False,
                "message": f"Failed to get session memory status: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def search_session_memory(self, query: str, limit: int = 5) -> dict:
        """Search session memory for messages matching query."""
        try:
            results = self.session_memory.search_messages(query, limit=limit)
            
            return {
                "type": "session_memory_search",
                "success": True,
                "query": query,
                "result_count": len(results),
                "results": [
                    {
                        "role": r['role'],
                        "content": r['content'][:200] + "..." if len(r['content']) > 200 else r['content'],
                        "time": r['datetime'],
                        "has_attachment": bool(r.get('metadata', {}).get('attachment_ids'))
                    }
                    for r in results
                ],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "type": "error",
                "success": False,
                "message": f"Failed to search session memory: {str(e)}",
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


@app.post("/api/upload")
async def upload_file_endpoint(
    file: UploadFile = File(...),
    transcribe: bool = Form(True)
):
    """Upload a file for attachment."""
    import time
    start_time = time.time()
    
    try:
        # Read file content
        content = await file.read()
        filename = file.filename or "unknown_file"
        
        print(f"📤 Upload request received: {filename} ({len(content)} bytes)")
        
        # Check file size
        if not content:
            return JSONResponse(
                content={"success": False, "error": "Empty file", "filename": filename},
                status_code=400
            )
        
        result = await web_interface.upload_file(content, filename, transcribe)
        
        elapsed = time.time() - start_time
        print(f"📤 Upload completed in {elapsed:.2f}s: {filename}")
        
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        elapsed = time.time() - start_time
        print(f"❌ Upload error after {elapsed:.2f}s: {e}")
        traceback.print_exc()
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=500
        )


@app.get("/api/attachments")
async def get_attachments():
    """Get list of pending attachments."""
    return {
        "attachments": [
            {
                "id": i,
                "filename": att["filename"],
                "file_type": att["file_type"],
                "size": att["size"],
                "uploaded_at": att["uploaded_at"]
            }
            for i, att in enumerate(web_interface.pending_attachments)
        ]
    }


@app.delete("/api/attachments/{attachment_id}")
async def delete_attachment(attachment_id: int):
    """Delete a pending attachment."""
    try:
        if 0 <= attachment_id < len(web_interface.pending_attachments):
            web_interface.pending_attachments.pop(attachment_id)
            return {"success": True, "message": "Attachment removed"}
        else:
            return {"success": False, "error": "Invalid attachment ID"}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
                attachment_ids = data.get("attachment_ids", [])  # Get attachment IDs from frontend
                
                # Send thinking indicator
                await websocket.send_json({
                    "type": "thinking",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Process message with attachments
                response = await web_interface.process_message(content, attachment_ids)
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
            
            elif message_type == "read_file_aloud":
                content = data.get("content", "")
                file_path = data.get("file_path", "")
                start_paragraph = data.get("start_paragraph", 1)
                language = data.get("language", "auto")
                
                if content or file_path:
                    result = await web_interface.read_file_aloud(
                        content=content,
                        file_path=file_path,
                        start_paragraph=start_paragraph,
                        language=language
                    )
                    await websocket.send_json(result)
            
            elif message_type == "stop_reading":
                result = await web_interface.stop_reading()
                await websocket.send_json(result)
            
            elif message_type == "get_reading_status":
                await websocket.send_json(web_interface.get_reading_status())
            
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


# Session Memory API Endpoints
@app.get("/api/session-memory")
async def get_session_memory():
    """Get session memory status and summary."""
    return web_interface.get_session_memory_status()


@app.post("/api/session-memory/search")
async def search_session_memory_endpoint(data: dict):
    """Search session memory for messages."""
    query = data.get("query", "")
    limit = data.get("limit", 5)
    if not query:
        return {"type": "error", "message": "No query provided"}
    return await web_interface.search_session_memory(query, limit)


@app.get("/api/session-memory/attachments")
async def get_session_attachments():
    """Get recent attachments from session memory."""
    try:
        attachments = web_interface.session_memory.get_recent_attachments(10)
        return {
            "type": "session_attachments",
            "attachments": [
                {
                    "id": a['id'],
                    "filename": a['filename'],
                    "type": a['file_type'],
                    "uploaded_at": a['datetime'],
                    "preview": a.get('content_preview', '')[:100] + "..." if a.get('content_preview') else None
                }
                for a in attachments
            ]
        }
    except Exception as e:
        return {"type": "error", "message": str(e)}


@app.post("/api/session-memory/clear")
async def clear_session_memory():
    """Clear current session memory."""
    try:
        web_interface.session_memory.clear()
        return {
            "type": "session_memory_cleared",
            "message": "Session memory cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"type": "error", "message": str(e)}


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
