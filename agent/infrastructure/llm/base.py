"""Abstract LLM provider interface for pluggable AI backends."""

import logging
import random
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.runnables import Runnable, RunnableConfig
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


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, skip provider
    HALF_OPEN = "half_open" # Testing recovery


class CircuitBreaker:
    """Stateful circuit breaker for a single provider.

    States:
    - CLOSED: Normal operation. Failures increment the counter.
    - OPEN: Provider is failing. All calls are rejected immediately.
      Transitions to HALF_OPEN after recovery_timeout seconds.
    - HALF_OPEN: One probe call is allowed. If it succeeds, circuit closes.
      If it fails, circuit re-opens.

    Args:
        failure_threshold: Number of consecutive failures before opening.
        recovery_timeout: Seconds to wait before attempting recovery.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        """Current circuit state, with automatic OPEN -> HALF_OPEN transition."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN")
        return self._state

    def can_execute(self) -> bool:
        """Return True if a call to the provider should be attempted."""
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        """Record a successful call. Resets circuit to CLOSED."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0

    def record_failure(self) -> None:
        """Record a failed call. May transition to OPEN."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Probe failed, re-open
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning("Circuit breaker re-opened after HALF_OPEN failure")
            return

        if self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                f"Circuit breaker OPEN after {self._failure_count} failures "
                f"(threshold: {self._failure_threshold})"
            )

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(state={self.state.value}, "
            f"failures={self._failure_count}/{self._failure_threshold})"
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


def _exponential_backoff_with_jitter(attempt: int, base: float = 1.0, max_wait: float = 30.0) -> float:
    """Calculate exponential backoff with full jitter.

    Uses the "decorrelated jitter" formula from AWS:
        sleep = min(cap, random.uniform(base, sleep * 3))

    Simplified here to: min(max_wait, random.uniform(0, base * 2^attempt))

    Args:
        attempt: Zero-based attempt number.
        base: Base delay in seconds.
        max_wait: Maximum delay cap in seconds.

    Returns:
        Seconds to sleep.
    """
    cap = min(max_wait, base * (2 ** attempt))
    return random.uniform(0, cap)


class ResilientChatModel(Runnable):
    """Wrapper around a provider's chat_model that implements failover across all providers.

    LangChain agents call bind_tools() on the chat_model, then invoke() on the
    resulting runnable. This wrapper intercepts both calls and tries providers
    in order, skipping those with OPEN circuit breakers.

    Inherits from Runnable so it works with LangChain's chain syntax (| operator).
    """

    def __init__(
        self,
        providers: list,
        circuit_breakers: Dict[str, CircuitBreaker],
    ):
        self._providers = providers
        self._circuit_breakers = circuit_breakers
        self._tools: list = []

    def bind_tools(self, tools: list, **kwargs):
        """Bind tools to ALL providers and return self for chained invocation."""
        self._tools = tools
        return self

    def _normalize_content(self, response: Any) -> Any:
        """Normalize response content: if .content is a list of blocks, extract text."""
        if not hasattr(response, "content"):
            return response
        content = response.content
        if isinstance(content, list):
            # Extract text from content blocks: [{'type': 'text', 'text': '...'}]
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif "text" in block:
                        text_parts.append(block["text"])
                elif hasattr(block, "text"):
                    text_parts.append(block.text)
                else:
                    text_parts.append(str(block))
            if text_parts:
                response.content = " ".join(text_parts)
        return response

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs) -> Any:
        """Try providers in order, failover on transient error with backoff."""
        errors = []

        for provider in self._providers:
            cb = self._circuit_breakers[provider.name]

            if not cb.can_execute():
                logger.info(
                    f"ResilientChatModel: provider={provider.name} circuit OPEN, skipping"
                )
                continue

            try:
                # Get the raw chat_model and bind tools for this specific call
                raw_model = provider.chat_model
                if self._tools:
                    bound_model = raw_model.bind_tools(self._tools, **kwargs)
                else:
                    bound_model = raw_model

                response = bound_model.invoke(input, config=config, **kwargs)
                response = self._normalize_content(response)
                cb.record_success()

                # Tag LangSmith trace with actual provider
                try:
                    current_run = run_tree_context.get_current_run_tree()
                    if current_run:
                        current_run.add_tags([f"provider:{provider.name}"])
                        current_run.add_metadata({
                            "actual_provider": provider.name,
                        })
                except Exception:
                    pass

                logger.info(f"ResilientChatModel: provider={provider.name}, success")
                return response

            except Exception as e:
                cb.record_failure()
                classified = provider._classify_error(e, provider=provider.name)

                if isinstance(classified, PermanentLLMError):
                    logger.error(
                        f"ResilientChatModel: provider={provider.name} permanent error: {e}"
                    )
                    raise

                logger.warning(
                    f"ResilientChatModel: provider={provider.name} transient error, "
                    f"trying next ({cb})"
                )
                errors.append(e)

        raise AllProvidersExhaustedError(
            message=f"All {len(errors)} LLM providers exhausted in ResilientChatModel",
            attempted_providers=[p.name for p in self._providers],
            errors=errors,
        )

    async def ainvoke(self, input: Any, config: Optional[dict] = None, **kwargs) -> Any:
        """Async version — try providers in order with failover."""
        errors = []

        for provider in self._providers:
            cb = self._circuit_breakers[provider.name]

            if not cb.can_execute():
                logger.info(
                    f"ResilientChatModel: provider={provider.name} circuit OPEN, skipping"
                )
                continue

            try:
                raw_model = provider.chat_model
                if self._tools:
                    bound_model = raw_model.bind_tools(self._tools, **kwargs)
                else:
                    bound_model = raw_model

                response = await bound_model.ainvoke(input, config=config, **kwargs)
                response = self._normalize_content(response)
                cb.record_success()

                try:
                    current_run = run_tree_context.get_current_run_tree()
                    if current_run:
                        current_run.add_tags([f"provider:{provider.name}"])
                        current_run.add_metadata({
                            "actual_provider": provider.name,
                        })
                except Exception:
                    pass

                logger.info(f"ResilientChatModel: provider={provider.name}, success")
                return response

            except Exception as e:
                cb.record_failure()
                classified = provider._classify_error(e, provider=provider.name)

                if isinstance(classified, PermanentLLMError):
                    logger.error(
                        f"ResilientChatModel: provider={provider.name} permanent error: {e}"
                    )
                    raise

                logger.warning(
                    f"ResilientChatModel: provider={provider.name} transient error, "
                    f"trying next ({cb})"
                )
                errors.append(e)

        raise AllProvidersExhaustedError(
            message=f"All {len(errors)} LLM providers exhausted in ResilientChatModel",
            attempted_providers=[p.name for p in self._providers],
            errors=errors,
        )

    def __repr__(self) -> str:
        return (
            f"ResilientChatModel(providers={[p.name for p in self._providers]}, "
            f"tools={[t.name if hasattr(t, 'name') else str(t) for t in self._tools]})"
        )


class ResilientLLMProvider(LLMProvider):
    """Wrapper that implements per-request failover across multiple LLM providers.

    Tries providers in order. If a provider fails with a transient error,
    waits with exponential backoff + jitter before trying the next provider.
    Each provider has an independent circuit breaker that opens after
    consecutive failures, skipping that provider until recovery timeout.
    Permanent errors propagate immediately. When all providers are exhausted,
    raises AllProvidersExhaustedError (or returns a fallback response if configured).
    """

    name = "resilient"

    def __init__(
        self,
        providers: list[LLMProvider],
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        fallback_response: Optional[str] = None,
    ):
        super().__init__(model="resilient", temperature=0.7)
        self._providers = providers
        self._circuit_breakers: Dict[str, CircuitBreaker] = {
            p.name: CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
            for p in providers
        }
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._fallback_response = fallback_response

    @property
    def chat_model(self):
        """Returns a ResilientChatModel that tries all providers with failover.

        Unlike the old behavior (which returned the first provider's raw model),
        this wrapper implements bind_tools() and invoke() with automatic failover
        across ALL configured providers.
        """
        return ResilientChatModel(
            providers=self._providers,
            circuit_breakers=self._circuit_breakers,
        )

    def resolve_chat_model(self):
        """Verify at least one provider is available.

        The chat_model property now returns a ResilientChatModel wrapper
        that handles failover automatically, so this just validates
        that the provider list is non-empty.

        Returns:
            The ResilientChatModel wrapper.

        Raises:
            RuntimeError: If no providers are available.
        """
        if not self._providers:
            raise RuntimeError("No LLM providers available")
        return self.chat_model

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Try providers in order, failover on transient error with backoff.

        Skips providers whose circuit breakers are OPEN. Waits with
        exponential backoff + jitter between attempts.

        Args:
            messages: List of dicts with 'role' and 'content'.
            **kwargs: Additional keyword arguments forwarded to the provider.

        Returns:
            LLMResponse from the first successful provider.

        Raises:
            PermanentLLMError: If a provider raises a permanent error.
            AllProvidersExhaustedError: If all providers fail or are unavailable.
        """
        attempted = []
        errors = []
        attempt_index = 0

        for provider in self._providers:
            cb = self._circuit_breakers[provider.name]

            if not cb.can_execute():
                logger.info(
                    f"provider={provider.name} circuit OPEN, skipping "
                    f"({cb})"
                )
                continue

            try:
                response = provider.invoke(messages, **kwargs)
                cb.record_success()
                self._enrich_response(response, provider.name, attempted)
                return response
            except PermanentLLMError:
                raise
            except TransientLLMError as e:
                cb.record_failure()
                logger.warning(
                    f"provider={provider.name} failed, trying next "
                    f"({cb})"
                )
                attempted.append(provider.name)
                errors.append(e)

                # Backoff before next provider attempt
                sleep_time = _exponential_backoff_with_jitter(
                    attempt=attempt_index,
                    base=self._backoff_base,
                    max_wait=self._backoff_max,
                )
                logger.debug(f"Backing off {sleep_time:.2f}s before next provider")
                time.sleep(sleep_time)
                attempt_index += 1

        raise AllProvidersExhaustedError(
            message=f"All {len(attempted)} LLM providers exhausted",
            attempted_providers=attempted,
            errors=errors,
        )

    def invoke_with_fallback(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Like invoke() but returns a fallback response instead of raising.

        When all providers are exhausted and a fallback_response is configured,
        returns a graceful degradation response instead of raising
        AllProvidersExhaustedError.

        Args:
            messages: List of dicts with 'role' and 'content'.
            **kwargs: Additional keyword arguments forwarded to the provider.

        Returns:
            LLMResponse from the first successful provider, or a fallback response.
        """
        try:
            return self.invoke(messages, **kwargs)
        except AllProvidersExhaustedError:
            if self._fallback_response:
                logger.error(
                    "All LLM providers exhausted, returning fallback response"
                )
                return LLMResponse(
                    content=self._fallback_response,
                    model="fallback",
                    provider="fallback",
                    metadata={
                        "degraded": True,
                        "message": (
                            "All AI providers are currently unavailable. "
                            "Please try again in a few minutes."
                        ),
                    },
                )
            raise

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

    def circuit_status(self) -> Dict[str, str]:
        """Return current circuit breaker status for all providers.

        Returns:
            Dict mapping provider name to circuit state string.
        """
        return {
            name: cb.state.value
            for name, cb in self._circuit_breakers.items()
        }
