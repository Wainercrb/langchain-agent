"""LLM Providers — Strategy Pattern.

Each provider implements LLMProvider (ABC). Instantiate directly:

    from llm import GoogleProvider

    # Change this line to swap providers
    llm = GoogleProvider(model="gemini-2.5-flash", api_key="...")
    # llm = OpenAIProvider(model="gpt-4", api_key="...")
    # llm = AnthropicProvider(model="claude-3-5-sonnet", ...)

Multi-provider routing (circuit breaker, failover, backoff) lives in
:mod:`core.router`. Import it as:
    ``from core.router import MultiProviderLLM, CircuitBreaker, CircuitState``
"""

from .base import LLMProvider, LLMResponse
from shared.exceptions import LLMProviderError, TransientLLMError, PermanentLLMError, AllProvidersExhaustedError
from .google import GoogleProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LLMProviderError",
    "TransientLLMError",
    "PermanentLLMError",
    "AllProvidersExhaustedError",
    "GoogleProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
]
