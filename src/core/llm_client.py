"""
LLM Client - Communicates with llama.cpp server via HTTP API.
"""

import json
import requests
from typing import Dict, List, Any, Optional, AsyncGenerator
import yaml


class LLMClient:
    """Client for llama.cpp HTTP server."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.llm_config = self.config["llm"]
        self.base_url = self.llm_config["base_url"]
        self.system_prompt = self.llm_config.get("system_prompt", "")
        
    def _get_completion_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Send chat completion request to llama.cpp server."""
        
        payload = {
            "model": self.llm_config["model"],
            "messages": messages,
            "max_tokens": self.llm_config.get("max_tokens", 512),
            "temperature": self.llm_config.get("temperature", 0.7),
            "stream": stream
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            response = requests.post(
                self._get_completion_url(),
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return {
                "error": f"Cannot connect to llama.cpp server at {self.base_url}",
                "choices": [{"message": {"content": "Error: LLM server not running"}}]
            }
        except Exception as e:
            return {
                "error": str(e),
                "choices": [{"message": {"content": f"Error: {str(e)}"}}]
            }
    
    def format_tools_for_llm(self, tools: Dict[str, Any]) -> List[Dict]:
        """Format tools in OpenAI-compatible format."""
        formatted_tools = []
        
        for name, tool in tools.items():
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })
        
        return formatted_tools
    
    def prepare_messages(self, user_input: str, conversation_history: List[Dict] = None) -> List[Dict]:
        """Prepare message list for LLM."""
        messages = []
        
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        return messages
