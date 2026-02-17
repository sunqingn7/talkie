import requests
from typing import Dict, List, Any, Optional
from .base import LLMProviderBase


class AnthropicProvider(LLMProviderBase):
    """Anthropic API provider (Claude) - supports OpenAI SDK compatibility."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self._init_client()
        
    def _init_client(self):
        try:
            from openai import AsyncOpenAI
            # Anthropic supports OpenAI SDK compatibility layer
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.anthropic.com/v1",
                max_retries=2,
                timeout=60.0
            )
        except ImportError:
            self.client = None
            
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        # Anthropic via OpenAI SDK
        if self.client and self.api_key:
            return await self._openai_sdk_completion(messages, tools, stream)
        else:
            return await self._direct_completion(messages, tools, stream)
    
    async def _openai_sdk_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        try:
            # Map model names for Anthropic
            model = self.model
            if not model.startswith("claude-"):
                model = f"claude-{model}"
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            
            response = await self.client.chat.completions.create(**payload)
            return response.model_dump()
            
        except Exception as e:
            return {
                "error": str(e),
                "choices": [{"message": {"content": f"Anthropic Error: {str(e)}"}}]
            }
    
    async def _direct_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Direct Anthropic API call."""
        try:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            
            # Build Anthropic-format messages
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    # Store system prompt for later
                    system = msg["content"]
                else:
                    anthropic_messages.append(msg)
            
            payload = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # Convert to OpenAI format
            content = data.get("content", [{}])[0].get("text", "")
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content
                    }
                }]
            }
            
        except Exception as e:
            return {
                "error": f"Anthropic API error: {str(e)}",
                "choices": [{"message": {"content": f"Error: {str(e)}"}}]
            }
    
    async def is_available(self) -> bool:
        if not self.api_key:
            return False
            
        try:
            # Simple API key validation
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1
            }
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            # 400 means key is valid but request failed (expected)
            return response.status_code in [200, 400]
        except Exception:
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": "claude-opus-4-6-20250514"},
            {"id": "claude-sonnet-4-20250514"},
            {"id": "claude-haiku-3-5-20250514"},
            {"id": "claude-3-opus-20240229"},
            {"id": "claude-3-sonnet-20240229"},
            {"id": "claude-3-haiku-20240307"},
        ]
