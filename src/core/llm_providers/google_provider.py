import requests
from typing import Dict, List, Any, Optional
from .base import LLMProviderBase


class GoogleProvider(LLMProviderBase):
    """Google Gemini API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self._init_client()
        
    def _init_client(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai
        except ImportError:
            self.client = None
            
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        if self.client:
            return await self._gemini_completion(messages, tools, stream)
        else:
            return await self._rest_completion(messages, tools, stream)
    
    async def _gemini_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        try:
            import google.generativeai as genai
            
            # Convert messages to Gemini format
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    # Gemini uses system instruction
                    model = genai.GenerativeModel(
                        self.model,
                        system_instruction=msg["content"]
                    )
                else:
                    contents.append(msg["content"])
            
            if not model:
                model = genai.GenerativeModel(self.model)
            
            # Start chat
            chat = model.start_chat(history=[])
            
            # Send message
            response = chat.send_message(
                contents[-1] if contents else "",
                stream=stream
            )
            
            # Convert to OpenAI format
            content = response.text if hasattr(response, "text") else str(response)
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
                "error": str(e),
                "choices": [{"message": {"content": f"Google Gemini Error: {str(e)}"}}]
            }
    
    async def _rest_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Direct REST API call."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            
            # Build prompt from messages
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
                else:
                    prompt += f"User: {msg['content']}\n"
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                }
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
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
                "error": f"Google API error: {str(e)}",
                "choices": [{"message": {"content": f"Error: {str(e)}"}}]
            }
    
    async def is_available(self) -> bool:
        if not self.api_key:
            return False
            
        try:
            # Test with a simple models list call
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [{"id": m["name"].split("/")[-1]} for m in data.get("models", [])]
        except Exception:
            pass
            
        return [
            {"id": "gemini-2.0-pro"},
            {"id": "gemini-2.0-flash"},
            {"id": "gemini-1.5-pro"},
            {"id": "gemini-1.5-flash"},
        ]
