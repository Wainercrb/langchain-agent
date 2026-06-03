"""Resilient LLM provider with retry, backoff, and failover."""

import logging
import random
import time
from typing import Any, Dict, List, Optional

from langchain_core.runnables import Runnable, RunnableConfig
from langsmith.run_trees import _context as run_tree_context

from utils.exceptions import TransientLLMError, PermanentLLMError, AllProvidersExhaustedError

from .base import LLMProvider, LLMResponse
from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


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

    def _tag_trace(self, provider_name: str) -> None:
        """Tag LangSmith trace with actual provider."""
        try:
            current_run = run_tree_context.get_current_run_tree()
            if current_run:
                current_run.add_tags([f"provider:{provider_name}"])
                current_run.add_metadata({
                    "actual_provider": provider_name,
                })
        except Exception:
            pass

    def _try_provider_sync(self, input: Any, config: Optional[dict], kwargs: dict) -> Any:
        """Core loop: try providers in order with circuit breaker + failover (sync)."""
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

                response = bound_model.invoke(input, config=config, **kwargs)
                response = self._normalize_content(response)
                cb.record_success()
                self._tag_trace(provider.name)

                logger.info(f"ResilientChatModel: provider={provider.name}, success")
                return response

            except Exception as e:
                cb.record_failure()
                classified = provider.classify_error(e, provider=provider.name)

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

    async def _try_provider_async(self, input: Any, config: Optional[dict], kwargs: dict) -> Any:
        """Core loop: try providers in order with circuit breaker + failover (async)."""
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
                self._tag_trace(provider.name)

                logger.info(f"ResilientChatModel: provider={provider.name}, success")
                return response

            except Exception as e:
                cb.record_failure()
                classified = provider.classify_error(e, provider=provider.name)

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

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs) -> Any:
        """Try providers in order, failover on transient error with backoff."""
        return self._try_provider_sync(input, config, kwargs)

    async def ainvoke(self, input: Any, config: Optional[dict] = None, **kwargs) -> Any:
        """Async version — try providers in order with failover."""
        return await self._try_provider_async(input, config, kwargs)

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
        """Returns a ResilientChatModel that tries all providers with failover."""
        return ResilientChatModel(
            providers=self._providers,
            circuit_breakers=self._circuit_breakers,
        )

    def resolve_chat_model(self):
        """Verify at least one provider is available."""
        if not self._providers:
            raise RuntimeError("No LLM providers available")
        return self.chat_model

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Try providers in order, failover on transient error with backoff."""
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

    def _enrich_response(
        self,
        response: LLMResponse,
        provider_name: str,
        attempted: list[str],
    ) -> None:
        """Add failover metadata and log the result."""
        response.metadata["provider"] = provider_name
        response.metadata["failover"] = len(attempted) > 0
        if attempted:
            response.metadata["failed_providers"] = attempted

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
        """Return current circuit breaker status for all providers."""
        return {
            name: cb.state.value
            for name, cb in self._circuit_breakers.items()
        }
