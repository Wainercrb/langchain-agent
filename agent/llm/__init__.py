"""LLM Providers — Strategy Pattern.

Each provider implements LLMProvider (ABC). Instantiate directly:

    from llm import GoogleProvider

    # Change this line to swap providers
    llm = GoogleProvider(model="gemini-2.5-flash", api_key="...")
    # llm = OpenAIProvider(model="gpt-4", api_key="...")
    # llm = AnthropicProvider(model="claude-3-5-sonnet", ...)
"""

from .base import LLMProvider, LLMResponse
from .multi import CircuitBreaker, CircuitState, MultiProviderLLM, MultiProviderChatModel
from shared.exceptions import LLMProviderError, TransientLLMError, PermanentLLMError, AllProvidersExhaustedError
from .google import GoogleProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "CircuitBreaker",
    "CircuitState",
    "MultiProviderLLM",
    "MultiProviderChatModel",
    "LLMProviderError",
    "TransientLLMError",
    "PermanentLLMError",
    "AllProvidersExhaustedError",
    "GoogleProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
]
