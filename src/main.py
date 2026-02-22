"""
Main orchestrator for the voice assistant.
Coordinates between STT, MCP tools, LLM, and TTS via Voice Daemon.
"""

import os
# Fix for PyTorch 2.6+ weights loading (must be set before importing torch)
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml

from core.llm_client import LLMClient
from mcp_integration.server import TalkieMCPServer
from core.voice_daemon import Priority


class VoiceAssistant:
    """Main voice assistant orchestrator using MCP-first design."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.llm_client = LLMClient(config_path)
        self.mcp_server = TalkieMCPServer(config_path)
        self.conversation_history = []
        self.session_start = datetime.now()
        self.wake_word_mode = False
        self.context_summary = None
        
        print("🚀 Talkie Voice Assistant initialized")
        print(f"   Available tools: {list(self.mcp_server.tools.keys())}")
        print(f"   Session started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    async def run(self):
        """Main assistant loop."""
        print("\n" + "="*60)
        print("🎙️  Talkie Voice Assistant - MCP-First Design")
        print("="*60)
        print("\n💡 Commands: 'help', 'quit', 'wake' (toggle wake word mode)\n")
        
        # Check if wake word tool is available
        if "wake_word" in self.mcp_server.tools:
            print("✅ Wake word detection available (try 'wake' command)")
        
        # Check if microphone is available
        try:
            import pyaudio
            print("✅ Microphone input available (try 'listen' command)\n")
            self.mic_available = True
        except ImportError:
            print("⚠️  Microphone not available\n")
            self.mic_available = False
        
        while True:
            try:
                # Get user input
                prompt = "🎤 You" if not self.wake_word_mode else "👂 Listening"
                user_input = input(f"{prompt}: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    await self._show_session_summary()
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'wake':
                    await self._toggle_wake_word_mode()
                    continue
                
                if user_input.lower() == 'context':
                    self._show_context()
                    continue
                
                if user_input.lower() == 'clear':
                    self._clear_context()
                    continue
                
                if user_input.lower() == 'listen':
                    if self.mic_available and "listen" in self.mcp_server.tools:
                        await self._record_and_process()
                    else:
                        print("❌ Microphone not available or listen tool not enabled")
                    continue
                
                # Check for wake word in wake mode
                if self.wake_word_mode and "wake_word" in self.mcp_server.tools:
                    wake_result = await self.mcp_server.tools["wake_word"].execute(
                        mode="check_text", 
                        text=user_input
                    )
                    
                    if not wake_result.get("detected"):
                        print("   💤 (Wake word not detected, still listening...)")
                        continue
                    else:
                        # Extract the command after wake word
                        remaining = wake_result.get("remaining_text", user_input)
                        if remaining:
                            user_input = remaining
                        print(f"   ✅ Wake word detected! Processing...")
                
                # Process the input
                await self._process_input(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
    
    async def _toggle_wake_word_mode(self):
        """Toggle wake word detection mode."""
        if "wake_word" not in self.mcp_server.tools:
            print("❌ Wake word tool not available")
            return
        
        self.wake_word_mode = not self.wake_word_mode
        
        if self.wake_word_mode:
            wake_phrases = self.mcp_server.tools["wake_word"].get_wake_phrases()
            print(f"\n👂 Wake word mode ENABLED")
            print(f"   Say one of these phrases to activate: {', '.join(wake_phrases)}")
            print(f"   Example: 'Hey Talkie, what's the weather?'")
            print(f"   Or type commands directly\n")
        else:
            print(f"\n📝 Wake word mode DISABLED")
            print(f"   Direct input mode active\n")
    
    def _show_context(self):
        """Display current conversation context."""
        print("\n" + "-"*60)
        print("📋 Conversation Context:")
        print("-"*60)
        print(f"Session start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total exchanges: {len(self.conversation_history) // 2}")
        print(f"Context window: Last {min(len(self.conversation_history), 10)} messages")
        
        if self.conversation_history:
            print("\nRecent messages:")
            for i, msg in enumerate(self.conversation_history[-6:], 1):
                role = "You" if msg['role'] == 'user' else "Assistant"
                content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                print(f"   {i}. {role}: {content}")
        
        if self.context_summary:
            print(f"\nSummary: {self.context_summary}")
        print("-"*60 + "\n")
    
    def _clear_context(self):
        """Clear conversation history and context."""
        self.conversation_history = []
        self.context_summary = None
        print("🗑️  Conversation context cleared\n")
    
    async def _show_session_summary(self):
        """Show session summary before exiting."""
        duration = datetime.now() - self.session_start
        exchanges = len(self.conversation_history) // 2
        
        print("\n" + "="*60)
        print("📊 Session Summary:")
        print("="*60)
        print(f"Duration: {duration.seconds // 60}m {duration.seconds % 60}s")
        print(f"Total exchanges: {exchanges}")
        print("="*60 + "\n")
    
    async def _process_input(self, user_input: str):
        """Process user input through LLM with tool support."""
        print("🤔 Thinking...")
        
        # Build enhanced context with system info
        context_messages = self._build_context_messages(user_input)
        
        # Format tools for LLM
        tools = self.llm_client.format_tools_for_llm(self.mcp_server.tools)
        
        # Get LLM response - run in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.llm_client.chat_completion(context_messages, tools=tools)
        )
        
        if "error" in response and "choices" not in response:
            print(f"❌ Error: {response['error']}")
            return
        
        # Extract message from response
        message = response["choices"][0]["message"]
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        
        # Handle tool calls
        if tool_calls:
            print(f"   [DEBUG] Processing {len(tool_calls)} tool call(s)")
            await self._handle_tool_calls(tool_calls, context_messages, user_input)
        elif content:
            # Direct response
            print(f"🤖 Assistant: {content}")
            
            # Auto-speak via Voice Daemon (HIGH priority - interrupts file reading)
            if self.mcp_server.voice_daemon:
                print("   [DEBUG] Direct response - queueing via Voice Daemon (HIGH priority)")
                self.mcp_server.voice_daemon.speak_immediately(text=content)
            elif "speak" in self.mcp_server.tools:
                # Fallback to direct TTS if voice daemon not available
                print("   [DEBUG] Voice daemon not available - using direct TTS")
                await self.mcp_server.tools["speak"].execute(text=content)
            
            # Update conversation history with metadata
            self._add_to_history(user_input, content)
    
    def _build_context_messages(self, user_input: str) -> list:
        """Build context messages with enhanced conversation history."""
        # Start with system context
        messages = []
        
        # Add conversation summary if we have a long history
        if len(self.conversation_history) > 20 and self.context_summary:
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.context_summary}"
            })
        
        # Add recent conversation history (last 10 exchanges)
        recent_history = self.conversation_history[-20:] if len(self.conversation_history) <= 20 else self.conversation_history[-20:]
        messages.extend(recent_history)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _add_to_history(self, user_input: str, assistant_response: str, tool_results = None):
        """Add exchange to conversation history with metadata."""
        import time
        
        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        
        # Add assistant message
        assistant_msg = {
            "role": "assistant",
            "content": assistant_response,
            "timestamp": time.time()
        }
        if tool_results:
            assistant_msg["tool_results"] = tool_results
        
        self.conversation_history.append(assistant_msg)
        
        # Trim history if too long (keep last 50 messages)
        if len(self.conversation_history) > 50:
            # Generate summary of oldest exchanges before removing
            if not self.context_summary:
                self._generate_context_summary()
            self.conversation_history = self.conversation_history[-40:]
    
    def _generate_context_summary(self):
        """Generate a summary of the conversation for context."""
        # Simple summary: note the topics discussed
        topics = set()
        for msg in self.conversation_history[:-20]:  # Look at older messages
            if msg["role"] == "user":
                # Extract key topics (simple approach)
                content = msg["content"].lower()
                if any(word in content for word in ["weather", "temperature", "rain"]):
                    topics.add("weather inquiries")
                if any(word in content for word in ["file", "read", "write", "folder"]):
                    topics.add("file operations")
                if any(word in content for word in ["command", "run", "execute"]):
                    topics.add("command execution")
        
        if topics:
            self.context_summary = "Previously discussed: " + ", ".join(topics)
    
    async def _handle_tool_calls(self, tool_calls: list, messages: list, original_input: str = ""):
        """Execute tool calls requested by LLM."""
        tool_results = []
        speak_called = False  # Track if LLM already called speak tool
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            
            print(f"🔧 Calling tool: {tool_name}({tool_args})")
            
            # Track if speak tool was called
            if tool_name == "speak":
                speak_called = True
                print(f"   [DEBUG] LLM called speak tool with args: {tool_args}")
            
            try:
                result = await self.mcp_server.call_tool(tool_name, tool_args)
                result_text = result[0].text if result else "{}"
                
                # Track tool results
                tool_results.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result_text
                })
                
                # Add tool result to messages
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
                
                print(f"✅ Tool result: {result_text[:100]}...")
                
            except Exception as e:
                print(f"❌ Tool error: {e}")
                error_msg = json.dumps({"error": str(e)})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": error_msg
                })
                tool_results.append({
                    "tool": tool_name,
                    "error": str(e)
                })
        
        # Get final response after tool execution - run in executor to avoid blocking event loop
        print("🤔 Processing tool results...")
        loop = asyncio.get_event_loop()
        final_response = await loop.run_in_executor(
            None,
            lambda: self.llm_client.chat_completion(messages)
        )
        
        if "choices" in final_response:
            final_content = final_response["choices"][0]["message"].get("content", "")
            if final_content:
                print(f"🤖 Assistant: {final_content}")
                
                # Only auto-speak if LLM didn't already call speak tool
                if not speak_called:
                    if self.mcp_server.voice_daemon:
                        # Use Voice Daemon (HIGH priority - interrupts file reading)
                        print("   [DEBUG] Auto-speaking via Voice Daemon (HIGH priority)")
                        self.mcp_server.voice_daemon.speak_immediately(text=final_content)
                    elif "speak" in self.mcp_server.tools:
                        # Fallback to direct TTS
                        print("   [DEBUG] Auto-speaking final response (LLM didn't call speak)")
                        await self.mcp_server.tools["speak"].execute(text=final_content)
                elif speak_called:
                    print("   [DEBUG] Skipping auto-speak (LLM already called speak tool)")
                
                # Update conversation history
                self._add_to_history(original_input, final_content, tool_results)
    
    async def _record_and_process(self):
        """Record audio from microphone and process it."""
        if "listen" not in self.mcp_server.tools:
            print("❌ Listen tool not available")
            return
        
        try:
            # Record and transcribe
            stt_result = await self.mcp_server.tools["listen"].execute(timeout=10)
            
            if stt_result.get("success"):
                text = stt_result.get("text", "")
                if text and text != "(no speech detected)":
                    print(f"📝 You said: '{text}'")
                    await self._process_input(text)
                else:
                    print("📝 (No speech detected)")
            else:
                print(f"❌ Speech recognition failed: {stt_result.get('error')}")
        
        except Exception as e:
            print(f"❌ Error recording audio: {e}")
    
    def _show_help(self):
        """Display help information."""
        print("\n" + "-"*60)
        print("📖 Available Commands:")
        print("-"*60)
        print("  help              - Show this help message")
        print("  quit/exit/bye     - Exit the assistant")
        print("  wake              - Toggle wake word detection mode")
        print("  context           - Show conversation context")
        print("  clear             - Clear conversation history")
        print("  listen            - Record voice input from microphone")
        print("\n🔧 Available Tools:")
        for name, tool in self.mcp_server.tools.items():
            desc = tool.description[:50] if len(tool.description) > 50 else tool.description
            print(f"  • {name}: {desc}")
        print("-"*60 + "\n")


async def main():
    """Main entry point."""
    assistant = VoiceAssistant()
    await assistant.run()


if __name__ == "__main__":
    asyncio.run(main())
