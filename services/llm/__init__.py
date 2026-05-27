"""LLM provider factory for easy provider switching."""

import logging
from typing import Optional

from .anthropic import AnthropicProvider
from .base import LLMProvider, LLMProviderError
from .gemini import GeminiProvider
from .openai import OpenAIProvider

logger = logging.getLogger(__name__)

# Supported providers
PROVIDERS = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def create_llm_provider(
    provider_name: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """
    Factory function to create LLM provider instances.
    
    Args:
        provider_name: Provider identifier ('gemini', 'openai', 'anthropic')
        model: Model identifier (uses provider default if None)
        temperature: Generation temperature (0.0-1.0)
        api_key: API key (uses env var if None)
        **kwargs: Provider-specific configuration
    
    Returns:
        LLMProvider: Initialized provider instance
    
    Raises:
        LLMProviderError: If provider not found or initialization fails
    
    Example:
        >>> provider = create_llm_provider("gemini", model="gemini-2.5-flash")
        >>> response = provider.invoke([{"role": "user", "content": "Hello"}])
        >>> print(response.content)
    """
    if provider_name.lower() not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise LLMProviderError(
            f"Unknown provider: {provider_name}. Available: {available}",
            provider=provider_name,
        )

    provider_class = PROVIDERS[provider_name.lower()]
    
    try:
        instance = provider_class(model=model, temperature=temperature, api_key=api_key, **kwargs)
        
        if not instance.validate_api_key():
            raise LLMProviderError(
                f"API key not found for {provider_name}. Set environment variable or pass api_key.",
                provider=provider_name,
            )
        
        logger.info(f"LLM provider created: {instance.get_provider_name()} ({instance.model})")
        return instance
    except LLMProviderError:
        raise
    except Exception as e:
        raise LLMProviderError(
            f"Failed to create provider {provider_name}: {str(e)}", provider=provider_name, original_error=e
        )


def get_provider_info() -> dict:
    """
    Get information about available providers.
    
    Returns:
        dict: Provider metadata including name and default model
    """
    return {
        "gemini": {"default_model": "gemini-2.5-flash", "class": GeminiProvider},
        "openai": {"default_model": "gpt-4", "class": OpenAIProvider},
        "anthropic": {"default_model": "claude-3-opus-20240229", "class": AnthropicProvider},
    }


__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "GeminiProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "create_llm_provider",
    "get_provider_info",
]
