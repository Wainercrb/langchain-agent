"""Abstract LLM provider interface for pluggable AI backends."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from utils.exceptions import TransientLLMError, PermanentLLMError

_PERMANENT_ERROR_KEYWORDS = (
    "authentication",
    "api key",
    "permission",
    "not found",
    "invalid",
)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Strategy Pattern: swap Gemini ↔ OpenAI ↔ Anthropic by instantiating the provider you need.
    """

    def __init__(self, model: str, temperature: float = 0.7, **kwargs):
        self.model = model
        self.temperature = temperature
        self.config = kwargs

    @property
    @abstractmethod
    def chat_model(self):
        """Expose the underlying LangChain chat model for tool calling.

        All providers must expose their native ChatModel instance so that
        tool-calling agents can use bind_tools() and other LangChain protocols
        that operate on the raw model object rather than the LLMResponse wrapper.
        """
        ...

    @abstractmethod
    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> "LLMResponse":
        """Generate response from messages.

        Args:
            messages: List of dicts with 'role' and 'content'
                     E.g.: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            **kwargs: Additional keyword arguments forwarded to the underlying LLM call

        Returns:
            LLMResponse with content and metadata

        Raises:
            LLMProviderError: if the provider fails
        """
        pass

    def _classify_error(self, error: Exception, provider: str) -> Exception:
        error_msg = str(error)
        if any(kw in error_msg.lower() for kw in _PERMANENT_ERROR_KEYWORDS):
            return PermanentLLMError(
                f"{provider} invoke failed: {error_msg}",
                provider=provider,
                original_error=error,
            )
        return TransientLLMError(
            f"{provider} invoke failed: {error_msg}",
            provider=provider,
            original_error=error,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model}, temperature={self.temperature})"


class LLMResponse:
    """Standardized response from LLM provider."""

    def __init__(
        self,
        content: str,
        model: str,
        provider: str,
        usage: Optional[Dict[str, int]] = None,
        **metadata,
    ):
        self.content = content
        self.model = model
        self.provider = provider
        self.usage = usage or {}
        self.metadata = metadata

    def __str__(self) -> str:
        return self.content
