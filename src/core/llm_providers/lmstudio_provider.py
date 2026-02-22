import requests
import os
import yaml
from typing import Dict, List, Any, Optional
from .base import LLMProviderBase


def _get_talkie_timeout() -> int:
    """Get timeout from model_params.yaml, default to 120."""
    try:
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'model_params.yaml')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            global_params = data.get("global_params", {})
            if "talkie_timeout" in global_params:
                return int(global_params["talkie_timeout"])
            extra_params = data.get("extra_params", "")
            if "--talkie_timeout" in extra_params:
                import shlex
                args = shlex.split(extra_params)
                for i, arg in enumerate(args):
                    if arg == "--talkie_timeout" and i + 1 < len(args):
                        return int(args[i + 1])
    except:
        pass
    return 120


class LMStudioProvider(LLMProviderBase):
    """LM Studio local LLM provider (OpenAI-compatible API)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:1234/v1")
        
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        try:
            url = f"{self.base_url}/chat/completions"
            
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
                response = requests.post(url, json=payload, stream=True, timeout=_get_talkie_timeout())
                response.raise_for_status()
                
                full_content = ""
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                import json
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                if delta.get("content"):
                                    full_content += delta["content"]
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
                response = requests.post(url, json=payload, timeout=_get_talkie_timeout())
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.ConnectionError:
            return {
                "error": f"Cannot connect to LM Studio at {self.base_url}",
                "choices": [{"message": {"content": "Error: LM Studio server not running"}}]
            }
        except Exception as e:
            return {
                "error": str(e),
                "choices": [{"message": {"content": f"Error: {str(e)}"}}]
            }
    
    async def is_available(self) -> bool:
        try:
            url = f"{self.base_url}/models"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            # Try alternative endpoint
            try:
                url = f"{self.base_url.replace('/v1', '')}/models"
                response = requests.get(url, timeout=5)
                return response.status_code == 200
            except Exception:
                return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        try:
            url = f"{self.base_url}/models"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [{"id": m["id"]} for m in data.get("data", [])]
        except Exception:
            pass
        return []
