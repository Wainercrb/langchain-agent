"""Abstract LLM provider interface for pluggable AI backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from shared.exceptions import TransientLLMError, PermanentLLMError

_PERMANENT_ERROR_KEYWORDS = (
    "authentication",
    "api key",
    "permission",
    "not found",
    "invalid",
)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Strategy Pattern: swap Gemini <-> OpenAI <-> Anthropic by instantiating the provider you need.
    """

    name: str = "unknown"

    def __init__(self, model: str, temperature: float = 0.7, **kwargs):
        self.model = model
        self.temperature = temperature
        self.config = kwargs

    @property
    @abstractmethod
    def chat_model(self):
        """Expose the underlying LangChain chat model for tool calling."""
        ...

    @abstractmethod
    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> "LLMResponse":
        """Generate response from messages.

        Args:
            messages: List of dicts with 'role' and 'content'
            **kwargs: Additional keyword arguments forwarded to the underlying LLM call

        Returns:
            LLMResponse with content and metadata

        Raises:
            LLMProviderError: if the provider fails
        """
        pass

    def _invoke_provider(self, messages: List[Dict[str, str]], **kwargs) -> "LLMResponse":
        """Template method: invoke chat model, wrap response in LLMResponse.

        Args:
            messages: List of dicts with 'role' and 'content'
            **kwargs: Additional keyword arguments forwarded to the underlying LLM call

        Returns:
            LLMResponse with content and metadata
        """
        response = self.chat_model.invoke(messages, **kwargs)
        return LLMResponse(
            content=response.content if hasattr(response, "content") else str(response),
            model=self.model,
            provider=self.name,
            usage=getattr(response, "usage_metadata", None),
        )

    def classify_error(self, error: Exception, provider: str) -> Exception:
        """Classify an error as transient (retriable) or permanent.

        Args:
            error: The exception to classify.
            provider: Name of the provider that raised the error.

        Returns:
            TransientLLMError for retriable errors, PermanentLLMError otherwise.
        """
        error_msg = str(error).lower()
        if any(kw in error_msg for kw in _PERMANENT_ERROR_KEYWORDS):
            return PermanentLLMError(
                f"{provider} invoke failed: {error}",
                provider=provider,
                original_error=error,
            )
        return TransientLLMError(
            f"{provider} invoke failed: {error}",
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
