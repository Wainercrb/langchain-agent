"""LLM Providers — Strategy Pattern.

Cada provider implementa LLMProvider (ABC). Se instancian directamente:

    from services.llm import GoogleProvider

    # Cambiás esta línea para cambiar de provider
    llm = GoogleProvider(model="gemini-2.5-flash", api_key="...")
    # llm = OpenAIProvider(model="gpt-4", api_key="...")
    # llm = AnthropicProvider(model="claude-3-5-sonnet", ...)
"""

from .base import LLMProvider, LLMProviderError
from .google import GoogleProvider
from .openai import OpenAIProvider

__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "GoogleProvider",
    "OpenAIProvider",
]
