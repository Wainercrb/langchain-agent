"""Abstract LLM provider interface for pluggable AI backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Strategy Pattern: swap Gemini ↔ OpenAI ↔ Anthropic by instantiating the provider you need.
    """

    def __init__(self, model: str, temperature: float = 0.7, **kwargs):
        self.model = model
        self.temperature = temperature
        self.config = kwargs

    @abstractmethod
    def invoke(self, messages: List[Dict[str, str]]) -> "LLMResponse":
        """Generate response from messages.

        Args:
            messages: List of dicts con 'role' y 'content'
                     Ej: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

        Returns:
            LLMResponse with content and metadata

        Raises:
            LLMProviderError: si el provider falla
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, temperature={self.temperature})"


class LLMResponse:
    """Standardized response from LLM provider."""

    def __init__(self, content: str, model: str, provider: str, usage: Optional[Dict[str, int]] = None, **metadata):
        self.content = content
        self.model = model
        self.provider = provider
        self.usage = usage or {}
        self.metadata = metadata

    def __str__(self) -> str:
        return self.content


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""

    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        self.message = message
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")
