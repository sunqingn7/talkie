import requests
from typing import Dict, List, Any, Optional
from .base import LLMProviderBase


class OllamaProvider(LLMProviderBase):
    """Ollama local LLM provider."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        try:
            url = f"{self.base_url}/api/chat"
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }
            
            if stream:
                # For streaming, return iterator
                response = requests.post(url, json=payload, stream=True, timeout=120)
                response.raise_for_status()
                
                full_content = ""
                for line in response.iter_lines():
                    if line:
                        data = line.decode('utf-8')
                        if data.startswith('data: '):
                            data = data[6:]
                        if data == '[DONE]':
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            if "message" in chunk and "content" in chunk["message"]:
                                full_content += chunk["message"]["content"]
                        except:
                            pass
                
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": full_content
                        }
                    }]
                }
            else:
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                data = response.json()
                
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": data.get("message", {}).get("content", "")
                        }
                    }]
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "error": f"Cannot connect to Ollama at {self.base_url}",
                "choices": [{"message": {"content": "Error: Ollama server not running"}}]
            }
        except Exception as e:
            return {
                "error": str(e),
                "choices": [{"message": {"content": f"Error: {str(e)}"}}]
            }
    
    async def is_available(self) -> bool:
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [{"id": m["name"]} for m in data.get("models", [])]
            return []
        except Exception:
            return []
    
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        return await self.chat_completion(messages, tools, stream=True)
