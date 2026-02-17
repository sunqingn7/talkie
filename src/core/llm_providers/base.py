from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator


class LLMProviderBase(ABC):
    """Abstract base class for all LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = config.get("provider", "unknown")
        self.model = config.get("model", "")
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "")
        self.system_prompt = config.get("system_prompt", "")
        self.max_tokens = config.get("max_tokens", 512)
        self.temperature = config.get("temperature", 0.7)
        
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Send chat completion request and return response."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available (server running, API key valid, etc.)"""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models for this provider."""
        pass
    
    def format_tools_for_llm(self, tools: Dict[str, Any]) -> List[Dict]:
        """Format tools in OpenAI-compatible format (default implementation)."""
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
    
    def prepare_messages(
        self, 
        user_input: str, 
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Prepare message list for LLM."""
        messages = []
        
        prompt = system_prompt or self.system_prompt
        if prompt:
            messages.append({
                "role": "system",
                "content": prompt
            })
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        return messages
    
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream completion yields tokens. Override in subclass if supported."""
        # Default: non-streaming, yield full response at end
        response = await self.chat_completion(messages, tools, stream=False)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        yield content


class ProviderCapability:
    """Flags indicating provider capabilities."""
    STREAMING = "streaming"
    TOOL_CALLING = "tool_calling"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"  # Alias for tool_calling
    EMBEDDINGS = "embeddings"
    
    @classmethod
    def get_defaults(cls) -> Dict[str, bool]:
        """Return default capability flags."""
        return {
            cls.STREAMING: True,
            cls.TOOL_CALLING: True,
            cls.VISION: False,
            cls.EMBEDDINGS: False,
        }
