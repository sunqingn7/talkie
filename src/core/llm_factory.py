from typing import Dict, Any, Optional
from .llm_providers.base import LLMProviderBase
from .llm_providers.openai_provider import OpenAIProvider
from .llm_providers.anthropic_provider import AnthropicProvider
from .llm_providers.google_provider import GoogleProvider
from .llm_providers.ollama_provider import OllamaProvider
from .llm_providers.lmstudio_provider import LMStudioProvider
from .llm_providers.llamacpp_provider import LlamaCppProvider
from .llm_providers.vllm_provider import VLLMProvider


class LLMFactory:
    """Factory class to create LLM provider instances."""

    PROVIDERS = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "gemini": GoogleProvider,
        "ollama": OllamaProvider,
        "lmstudio": LMStudioProvider,
        "llamacpp": LlamaCppProvider,
        "vllm": VLLMProvider,
    }

    @classmethod
    def create(
        cls, provider_name: str, config: Dict[str, Any]
    ) -> Optional[LLMProviderBase]:
        """Create an LLM provider instance based on provider name."""
        provider_class = cls.PROVIDERS.get(provider_name.lower())

        if not provider_class:
            print(f"[LLMFactory] Unknown provider: {provider_name}")
            return None

        try:
            # Merge config with defaults
            full_config = {
                "provider": provider_name,
                "model": config.get("model", ""),
                "api_key": config.get("api_key", ""),
                "base_url": config.get("base_url", ""),
                "system_prompt": config.get("system_prompt", ""),
                "max_tokens": config.get("max_tokens", 512),
                "temperature": config.get("temperature", 0.7),
            }

            return provider_class(full_config)
        except Exception as e:
            print(f"[LLMFactory] Error creating provider {provider_name}: {e}")
            return None

    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a custom provider."""
        cls.PROVIDERS[name.lower()] = provider_class

    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available provider names."""
        return list(cls.PROVIDERS.keys())

    @classmethod
    def create_from_config(cls, llm_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create main, fallback, and agents from full config."""
        result = {"main": None, "fallback": None, "agents": {}, "available_agents": []}

        # Create main provider
        if "main" in llm_config:
            main_cfg = llm_config["main"]
            provider = main_cfg.get("provider", "llamacpp")
            result["main"] = cls.create(provider, main_cfg)

        # Create fallback provider
        if "fallback" in llm_config and llm_config["fallback"]:
            fallback_cfg = llm_config["fallback"]
            provider = fallback_cfg.get("provider", "ollama")
            result["fallback"] = cls.create(provider, fallback_cfg)

        # Create agents
        if "agents" in llm_config:
            for agent_name, agent_config in llm_config["agents"].items():
                provider = agent_config.get("provider", "ollama")
                agent = cls.create(provider, agent_config)
                if agent:
                    result["agents"][agent_name] = agent

        return result
