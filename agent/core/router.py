"""Consolidated LLM resilience: circuit breaker, backoff, and provider failover."""

import logging
import random
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.runnables import Runnable

from agent.observability.base import get_observability_provider
from shared.exceptions import (
    AllProvidersExhaustedError,
    PermanentLLMError,
    TransientLLMError,
)

from llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

_PERMANENT_KEYWORDS = (
    "authentication",
    "api key",
    "permission",
    "not found",
    "invalid",
)


class CircuitState(Enum):
    """Circuit breaker states: closed (normal), open (failing), half_open (probing)."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Stateful circuit breaker for a single LLM provider.

    Prevents cascading failures by opening the circuit after
    ``failure_threshold`` consecutive failures. After ``recovery_timeout``
    seconds the circuit transitions to HALF_OPEN, allowing one probe
    call. A successful probe closes the circuit; a failed one re-opens it.

    Args:
        failure_threshold: Consecutive failures before opening.
        recovery_timeout: Seconds to wait before probing.
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
        self._opened_at: float = 0.0

    @property
    def state(self) -> CircuitState:
        """Current circuit state with automatic OPEN → HALF_OPEN transition."""
        if self._state == CircuitState.OPEN and (
            time.monotonic() - self._opened_at >= self._recovery_timeout
        ):
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

    def record_failure(self) -> None:
        """Record a failed call. May transition to OPEN."""
        self._failure_count += 1

        if self._state == CircuitState.HALF_OPEN:
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


# ── Module-level helpers ────────────────────────────────────────────────


def _classify(error: Exception, provider_name: str) -> PermanentLLMError | TransientLLMError:
    """Return PermanentLLMError if message contains a permanent-failure keyword, else TransientLLMError."""
    msg = str(error).lower()
    if any(kw in msg for kw in _PERMANENT_KEYWORDS):
        return PermanentLLMError(
            f"{provider_name} invoke failed: {error}",
            provider=provider_name,
            original_error=error,
        )
    return TransientLLMError(
        f"{provider_name} invoke failed: {error}",
        provider=provider_name,
        original_error=error,
    )


def _backoff(attempt: int, base: float = 1.0, max_wait: float = 30.0) -> float:
    """Full-jitter exponential backoff: random(0, min(max_wait, base * 2^attempt))."""
    cap = min(max_wait, base * (2 ** attempt))
    return random.uniform(0, cap)


def _tag_trace(provider_name: str) -> None:
    """Tag the active LangSmith trace with the actual provider. Swallows observability errors."""
    try:
        obs = get_observability_provider()
        run_id = obs.get_current_run_id()
        if run_id:
            obs.apply_tags(run_id, [f"provider:{provider_name}"])
            obs.apply_metadata(run_id, {"actual_provider": provider_name})
    except Exception:
        pass


# ── Shared failover loop ─────────────────────────────────────────────────


def _try_providers(
    providers: list,
    circuit_breakers: Dict[str, CircuitBreaker],
    call_provider: Any,  # Callable[[Any], Any] — per-provider invocation logic
    name_fn: Any,  # Callable[[Any], str] — extracts the provider name for keys/logs
    backoff_base: float,
    backoff_max: float,
    on_transient: Any = None,  # Optional[Callable[[str, Exception], None]]
) -> Any:
    """Try providers in order with circuit breaker, backoff, and error classification.

    The *only* thing that differs between callers is ``call_provider`` —
    everything else (circuit checks, error classification, backoff, exhausting)
    is handled here once.

    Args:
        providers: Ordered list of provider instances.
        circuit_breakers: Dict mapping provider name -> CircuitBreaker.
        call_provider: Invoked as ``call_provider(provider)``. Must raise
            ``PermanentLLMError`` on non-retryable errors; any other exception
            is classified internally.
        name_fn: ``name_fn(provider) -> str`` — used for circuit breaker keys,
            logging, and trace tagging.
        backoff_base: Base delay in seconds for exponential backoff.
        backoff_max: Maximum delay cap in seconds.
        on_transient: Optional callback invoked as ``on_transient(name, exc)``
            after each transient failure. Used by callers that need to track
            which providers failed before eventual success (e.g. for metadata).

    Raises:
        PermanentLLMError: If any provider raises a permanent-failure error.
        AllProvidersExhaustedError: If all providers fail transiently.

    Returns:
        The result of the first successful ``call_provider(provider)``.
    """
    errors: list = []
    attempt_index: int = 0

    for i, provider in enumerate(providers):
        name = name_fn(provider)
        cb = circuit_breakers[name]

        if not cb.can_execute():
            logger.info(f"Circuit OPEN, skipping {name} ({cb})")
            continue

        try:
            result = call_provider(provider)
            cb.record_success()
            _tag_trace(name)
            return result
        except PermanentLLMError:
            raise
        except Exception as e:
            classified = _classify(e, name)
            if isinstance(classified, PermanentLLMError):
                logger.error(
                    f"Provider {name} permanent error: {e}"
                )
                raise
            cb.record_failure()
            logger.warning(
                f"Provider {name} transient error, trying next ({cb})"
            )
            errors.append(e)
            if on_transient is not None:
                on_transient(name, e)
            if i < len(providers) - 1:
                time.sleep(
                    _backoff(
                        attempt=attempt_index,
                        base=backoff_base,
                        max_wait=backoff_max,
                    )
                )
            attempt_index += 1

    raise AllProvidersExhaustedError(
        message=f"All {len(errors)} LLM providers exhausted",
        attempted_providers=[name_fn(p) for p in providers],
        errors=errors,
    )


# ── MultiProviderChatModel (LangChain Runnable) ─────────────────────────


class MultiProviderChatModel(Runnable):
    """Runnable that fails over across multiple LLM providers. Inherits from Runnable so LangChain's create_tool_cutting_agent can pipe it."""

    def __init__(
        self,
        providers: list,
        circuit_breakers: Dict[str, CircuitBreaker],
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
    ):
        self._providers = providers
        self._circuit_breakers = circuit_breakers
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._tools: list = []

    def bind_tools(self, tools: list, **kwargs):
        """Bind tools to ALL providers and return self for chained invocation."""
        self._tools = tools
        return self

    @staticmethod
    def _normalize_content(response: Any) -> Any:
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

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs) -> Any:
        """Try providers in order, failover on transient error (sync)."""
        def _call(provider: object) -> Any:
            raw_model = provider.chat_model  # type: ignore[attr-defined]
            if self._tools:
                bound = raw_model.bind_tools(self._tools, **kwargs)
            else:
                bound = raw_model
            response = bound.invoke(input, config=config, **kwargs)
            return self._normalize_content(response)

        return _try_providers(
            providers=self._providers,
            circuit_breakers=self._circuit_breakers,
            call_provider=_call,
            name_fn=lambda p: p.name,
            backoff_base=self._backoff_base,
            backoff_max=self._backoff_max,
        )


# ── MultiProviderLLM (LLMProvider interface) ────────────────────────────


class MultiProviderLLM(LLMProvider):
    """LLMProvider that tries multiple underlying providers with circuit breaker, backoff, and error classification.

    Args:
        providers: Ordered list of LLMProvider instances.
        failure_threshold: Consecutive failures before opening circuit.
        recovery_timeout: Seconds before probing a closed circuit.
        backoff_base: Base delay for exponential backoff.
        backoff_max: Maximum delay cap.
    """

    name = "multi"

    def __init__(
        self,
        providers: list[LLMProvider],
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
    ):
        super().__init__(model="multi", temperature=0.7)
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

    @property
    def chat_model(self):
        """Return a MultiProviderChatModel for LangChain tool-calling flows."""
        return MultiProviderChatModel(
            providers=self._providers,
            circuit_breakers=self._circuit_breakers,
            backoff_base=self._backoff_base,
            backoff_max=self._backoff_max,
        )

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Try providers in order with circuit breaker, backoff, and error classification.

        Raises PermanentLLMError on permanent errors, AllProvidersExhaustedError if all fail.
        """
        failed_providers: List[str] = []

        def _call(provider: LLMProvider) -> LLMResponse:
            response = provider.invoke(messages, **kwargs)
            response.metadata["provider"] = provider.name
            response.metadata["failover"] = bool(failed_providers)
            if failed_providers:
                response.metadata["failed_providers"] = list(failed_providers)
            return response

        result: LLMResponse = _try_providers(
            providers=self._providers,
            circuit_breakers=self._circuit_breakers,
            call_provider=_call,
            name_fn=lambda p: p.name,
            backoff_base=self._backoff_base,
            backoff_max=self._backoff_max,
            on_transient=lambda name, _: failed_providers.append(name),
        )

        # Logging: after successful result, _call already enriched the metadata.
        # We log here (outside the hot loop) once, instead of per-provider.
        if failed_providers:
            logger.info(
                f"provider={result.metadata.get('provider')}, "
                f"failover=true, failed_providers={list(failed_providers)}"
            )
        else:
            logger.info(
                f"provider={result.metadata.get('provider')}, failover=false"
            )

        return result

    def circuit_status(self) -> Dict[str, str]:
        """Return current circuit breaker status for all providers."""
        return {
            name: cb.state.value
            for name, cb in self._circuit_breakers.items()
        }
