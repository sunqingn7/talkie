import requests
from typing import Dict, List, Any, Optional
from .base import LLMProviderBase


class OpenAIProvider(LLMProviderBase):
    """OpenAI API provider (GPT-4, GPT-4o, etc.)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self._init_client()
        
    def _init_client(self):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url or "https://api.openai.com/v1",
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
        if not self.client:
            return self._fallback_completion(messages, tools, stream)
            
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": stream
            }
            
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            
            if stream:
                response = await self.client.chat.completions.create(**payload)
                content = ""
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                return {"choices": [{"message": {"content": content}}]}
            else:
                response = await self.client.chat.completions.create(**payload)
                return response.model_dump()
                
        except Exception as e:
            return {
                "error": str(e),
                "choices": [{"message": {"content": f"OpenAI Error: {str(e)}"}}]
            }
    
    def _fallback_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Fallback using requests if openai SDK not available."""
        try:
            url = f"{self.base_url or 'https://api.openai.com/v1'}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            return {
                "error": f"OpenAI API error: {str(e)}",
                "choices": [{"message": {"content": f"Error: {str(e)}"}}]
            }
    
    async def is_available(self) -> bool:
        if not self.api_key:
            return False
            
        try:
            if self.client:
                # Try a simple models list call
                await self.client.models.list()
                return True
            else:
                # Fallback check
                url = f"{self.base_url or 'https://api.openai.com/v1'}/models"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.get(url, headers=headers, timeout=10)
                return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            if self.client:
                models = await self.client.models.list()
                return [{"id": m.id, "object": "model"} for m in models.data]
            else:
                url = f"{self.base_url or 'https://api.openai.com/v1'}/models"
                headers = {"Authorization": f"Bearer {self.api_key}"}
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return [{"id": m["id"]} for m in data.get("data", [])]
                return []
        except Exception:
            # Return default models
            return [
                {"id": "gpt-4o"},
                {"id": "gpt-4o-mini"},
                {"id": "gpt-4-turbo"},
                {"id": "gpt-3.5-turbo"},
            ]
