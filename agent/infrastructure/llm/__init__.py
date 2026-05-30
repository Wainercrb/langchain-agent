"""LLM Providers — Strategy Pattern.

Each provider implements LLMProvider (ABC). Instantiate directly:

    from infrastructure.llm import GoogleProvider

    # Change this line to swap providers
    llm = GoogleProvider(model="gemini-2.5-flash", api_key="...")
    # llm = OpenAIProvider(model="gpt-4", api_key="...")
    # llm = AnthropicProvider(model="claude-3-5-sonnet", ...)
"""

from .base import LLMProvider
from utils.exceptions import LLMProviderError, TransientLLMError, PermanentLLMError
from .google import GoogleProvider
from .openai import OpenAIProvider
from .openrouter import OpenRouterProvider

__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "TransientLLMError",
    "PermanentLLMError",
    "GoogleProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
]
