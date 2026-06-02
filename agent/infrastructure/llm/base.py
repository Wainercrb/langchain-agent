"""Abstract LLM provider interface for pluggable AI backends."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from langsmith.run_trees import _context as run_tree_context

from utils.exceptions import TransientLLMError, PermanentLLMError, AllProvidersExhaustedError

logger = logging.getLogger(__name__)

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

    name: str = "unknown"

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


class ResilientLLMProvider(LLMProvider):
    """Wrapper that implements per-request failover across multiple LLM providers.

    Tries providers in order. If a provider fails with a transient error,
    immediately moves to the next provider. Permanent errors propagate
    immediately. When all providers are exhausted, raises AllProvidersExhaustedError.
    """

    name = "resilient"

    def __init__(self, providers: list[LLMProvider]):
        super().__init__(model="resilient", temperature=0.7)
        self._providers = providers

    @property
    def chat_model(self):
        """Returns the resolved chat_model (set during resolve_chat_model)."""
        if not hasattr(self, "_resolved_chat_model"):
            raise RuntimeError(
                "chat_model not resolved yet. Call resolve_chat_model() first."
            )
        return self._resolved_chat_model

    def resolve_chat_model(self):
        """Test providers in order, return first working chat_model.

        Returns:
            The first available provider's chat_model.

        Raises:
            RuntimeError: If no providers are available.
        """
        for provider in self._providers:
            try:
                chat_model = provider.chat_model
                if chat_model is not None:
                    self._resolved_chat_model = chat_model
                    return chat_model
            except Exception:
                continue

        raise RuntimeError("No LLM providers available")

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Try providers in order, failover immediately on transient error.

        Args:
            messages: List of dicts with 'role' and 'content'.
            **kwargs: Additional keyword arguments forwarded to the provider.

        Returns:
            LLMResponse from the first successful provider.

        Raises:
            PermanentLLMError: If a provider raises a permanent error.
            AllProvidersExhaustedError: If all providers fail with transient errors.
        """
        attempted = []
        errors = []

        for provider in self._providers:
            try:
                response = provider.invoke(messages, **kwargs)
                self._enrich_response(response, provider.name, attempted)
                return response
            except PermanentLLMError:
                raise
            except TransientLLMError as e:
                logger.warning(f"provider={provider.name} failed, trying next")
                attempted.append(provider.name)
                errors.append(e)

        raise AllProvidersExhaustedError(
            message=f"All {len(attempted)} LLM providers exhausted",
            attempted_providers=attempted,
            errors=errors,
        )

    def _enrich_response(
        self,
        response: LLMResponse,
        provider_name: str,
        attempted: list[str],
    ) -> None:
        """Add failover metadata and log the result.

        Args:
            response: The LLMResponse to enrich.
            provider_name: Name of the provider that succeeded.
            attempted: List of provider names that failed before this one.
        """
        response.metadata["provider"] = provider_name
        response.metadata["failover"] = len(attempted) > 0
        if attempted:
            response.metadata["failed_providers"] = attempted

        # Tag LangSmith trace with actual provider for dashboard filtering
        try:
            current_run = run_tree_context.get_current_run_tree()
            if current_run:
                current_run.add_tags([f"provider:{provider_name}"])
                current_run.add_metadata({
                    "actual_provider": provider_name,
                    "failed_providers": attempted if attempted else [],
                })
        except Exception:
            pass

        if attempted:
            logger.info(
                f"provider={provider_name}, failover=true, failed_provider={attempted[-1]}"
            )
        else:
            logger.info(f"provider={provider_name}, failover=false")
