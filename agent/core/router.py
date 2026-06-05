"""Consolidated LLM resilience: circuit breaker, backoff, and provider failover."""

import random
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable

from langchain_core.runnables import Runnable

from shared.exceptions import (
    AllProvidersExhaustedError,
    PermanentLLMError,
    TransientLLMError,
)
from llm.base import LLMProvider, LLMResponse
from loggers import logger

_PERMANENT_KEYWORDS = (
    "authentication",
    "api key",
    "permission",
    "not found",
    "invalid",
)


# ── Circuit Breaker (immutable state + pure transition functions) ────────


class CircuitState(Enum):
    """Circuit breaker states: closed (normal), open (failing), half_open (probing)."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class CircuitBreaker:
    """Immutable circuit breaker state for a single LLM provider.

    Pure data object — every state transition produces a new instance so
    there are no hidden mutations. The caller (``_try_providers``) owns
    the mutable ``dict[str, CircuitBreaker]`` and replaces entries on
    each transition.

    Attributes:
        failure_threshold: Consecutive failures before opening the circuit.
        recovery_timeout:  Seconds to wait before allowing a probe.
        state:             Current circuit state (CLOSED / OPEN / HALF_OPEN).
        failure_count:     Consecutive failure counter.
        opened_at:         ``time.monotonic()`` timestamp of the last OPEN transition.
    """
    failure_threshold: int = 3
    recovery_timeout: float = 60.0
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    opened_at: float = 0.0


def _cb_initial(failure_threshold: int, recovery_timeout: float) -> CircuitBreaker:
    """Return a fresh CLOSED breaker."""
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )


def _cb_can_execute(cb: CircuitBreaker) -> tuple[bool, CircuitBreaker]:
    """Return ``(can_execute, resolved_circuit)``.

    If the circuit is OPEN and ``recovery_timeout`` has elapsed, it
    auto-transitions to HALF_OPEN (allowing one probe call).
    """
    if cb.state == CircuitState.OPEN:
        if time.monotonic() - cb.opened_at >= cb.recovery_timeout:
            resolved = replace(cb, state=CircuitState.HALF_OPEN)
            logger.info("Circuit breaker transitioning to HALF_OPEN")
            return True, resolved
        return False, cb
    return True, cb


def _cb_after_success(cb: CircuitBreaker) -> CircuitBreaker:
    """Return a new CLOSED breaker (failure counters reset)."""
    return CircuitBreaker(
        failure_threshold=cb.failure_threshold,
        recovery_timeout=cb.recovery_timeout,
    )


def _cb_after_failure(cb: CircuitBreaker) -> CircuitBreaker:
    """Return a new breaker with incremented failures, possibly transitioning to OPEN.

    - HALF_OPEN + failure  →  re-opens immediately.
    - threshold reached    →  transitions to OPEN (records timestamp).
    - otherwise            →  only increments the failure counter.
    """
    failure_count = cb.failure_count + 1

    if cb.state == CircuitState.HALF_OPEN:
        logger.warning("Circuit breaker re-opened after HALF_OPEN failure")
        return CircuitBreaker(
            failure_threshold=cb.failure_threshold,
            recovery_timeout=cb.recovery_timeout,
            state=CircuitState.OPEN,
            failure_count=failure_count,
            opened_at=time.monotonic(),
        )

    if failure_count >= cb.failure_threshold:
        logger.warning(
            f"Circuit breaker OPEN after {failure_count} failures "
            f"(threshold: {cb.failure_threshold})"
        )
        return CircuitBreaker(
            failure_threshold=cb.failure_threshold,
            recovery_timeout=cb.recovery_timeout,
            state=CircuitState.OPEN,
            failure_count=failure_count,
            opened_at=time.monotonic(),
        )

    return replace(cb, failure_count=failure_count)


# ── Pure helpers ─────────────────────────────────────────────────────────


def _classify(error: Exception, provider_name: str) -> PermanentLLMError | TransientLLMError:
    """Classify an exception as permanent (non-retryable) or transient (retryable).

    Permanent keywords are checked case-insensitively against the error message.
    """
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
    """Full-jitter exponential backoff: ``random(0, min(max_wait, base * 2^attempt))``."""
    cap = min(max_wait, base * (2 ** attempt))
    return random.uniform(0, cap)


def _normalize_content(response: Any) -> Any:
    """Collapse multi-block ``.content`` into a flat text string.

    Some LLM providers return content as a list of text/tool-call blocks.
    This extracts text-only blocks and joins them so downstream consumers
    always see plain text.
    """
    if not hasattr(response, "content"):
        return response

    content = response.content
    if not isinstance(content, list):
        return response

    text_parts: list[str] = []
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


# ── Impure side-effects ──────────────────────────────────────────────────


def _tag_trace(provider_name: str) -> None:
    """Tag the active LangSmith trace with the actual provider. Swallows observability errors."""
    try:
        from observability.base import get_observability_provider

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
    circuit_breakers: dict[str, CircuitBreaker],
    call_provider: Callable[[Any], Any],
    name_fn: Callable[[Any], str],
    backoff_base: float,
    backoff_max: float,
    on_transient: Callable[[str, Exception], None] | None = None,
) -> Any:
    """Try providers in order with circuit breaker, backoff, and error classification.

    The *only* thing that differs between callers is ``call_provider`` —
    everything else (circuit checks, error classification, backoff, exhaustion)
    is handled here once.

    Args:
        providers:       Ordered list of provider instances.
        circuit_breakers: Mutable dict mapping provider name -> CircuitBreaker.
        call_provider:   Invoked as ``call_provider(provider)``. Should raise
                         ``PermanentLLMError`` on non-retryable errors; any
                         other exception is classified internally.
        name_fn:         ``name_fn(provider) -> str`` — used for circuit breaker
                         keys, logging, and trace tagging.
        backoff_base:    Base delay in seconds for exponential backoff.
        backoff_max:     Maximum delay cap in seconds.
        on_transient:    Optional callback ``on_transient(name, exc)`` invoked
                         after each transient failure.

    Raises:
        PermanentLLMError:         If any provider raises a permanent error.
        AllProvidersExhaustedError: If all providers fail transiently.

    Returns:
        The result of the first successful ``call_provider(provider)``.
    """
    errors: list[Exception] = []
    attempt_index: int = 0

    for i, provider in enumerate(providers):
        name = name_fn(provider)

        # Resolve possible OPEN→HALF_OPEN transition, persist any change.
        cb = circuit_breakers[name]
        can_execute, cb = _cb_can_execute(cb)
        circuit_breakers[name] = cb

        if not can_execute:
            logger.info(
                f"Circuit OPEN, skipping {name} "
                f"(failures={cb.failure_count}/{cb.failure_threshold})"
            )
            continue

        try:
            result = call_provider(provider)
            circuit_breakers[name] = _cb_after_success(cb)
            _tag_trace(name)
            return result
        except PermanentLLMError:
            raise
        except Exception as e:
            classified = _classify(e, name)
            if isinstance(classified, PermanentLLMError):
                logger.error(f"Provider {name} permanent error: {e}")
                raise

            circuit_breakers[name] = _cb_after_failure(cb)
            errors.append(e)
            logger.warning(
                f"Provider {name} transient error, trying next "
                f"(failures={circuit_breakers[name].failure_count}/"
                f"{circuit_breakers[name].failure_threshold})"
            )
            if on_transient is not None:
                on_transient(name, e)

            if i < len(providers) - 1:
                time.sleep(_backoff(
                    attempt=attempt_index,
                    base=backoff_base,
                    max_wait=backoff_max,
                ))
            attempt_index += 1

    raise AllProvidersExhaustedError(
        message=f"All {len(errors)} LLM providers exhausted",
        attempted_providers=[name_fn(p) for p in providers],
        errors=errors,
    )


# ── MultiProviderChatModel (LangChain Runnable) ─────────────────────────


class MultiProviderChatModel(Runnable):
    """Runnable that fails over across multiple LLM providers.

    Inhabits the ``ChatModelLike`` protocol so ``ToolCallingAgent`` can use
    it as a standard LangChain chat model with tool-binding support.
    """

    def __init__(
        self,
        providers: list,
        circuit_breakers: dict[str, CircuitBreaker],
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
    ):
        self._providers = providers
        self._circuit_breakers = circuit_breakers
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._tools: list = []
        self._last_model: str | None = None

    @property
    def model(self) -> str:
        """Model identifier — the **actual** model used in the last invocation
        if available, otherwise all configured models joined."""
        if self._last_model:
            return self._last_model
        names = [getattr(p, "model", p.name) for p in self._providers]
        return "|".join(names)

    def bind_tools(self, tools: list, **kwargs):
        """Bind tools to ALL providers and return self for chained invocation."""
        self._tools = tools
        return self

    def invoke(self, input: Any, config: dict | None = None, **kwargs) -> Any:
        """Try providers in order, failover on transient error (sync).

        Records which provider actually responded so ``.model`` returns
        the real model identifier.
        """
        used: list[object] = []

        def _call(provider: object) -> Any:
            raw_model = provider.chat_model  # type: ignore[attr-defined]
            bound = raw_model.bind_tools(self._tools, **kwargs) if self._tools else raw_model
            response = bound.invoke(input, config=config, **kwargs)
            used.append(provider)
            return _normalize_content(response)

        result = _try_providers(
            providers=self._providers,
            circuit_breakers=self._circuit_breakers,
            call_provider=_call,
            name_fn=lambda p: p.name,
            backoff_base=self._backoff_base,
            backoff_max=self._backoff_max,
        )

        if used:
            p = used[0]
            self._last_model = getattr(p, "model", getattr(p, "name", "unknown"))

        return result


# ── MultiProviderLLM (LLMProvider interface) ────────────────────────────


class MultiProviderLLM(LLMProvider):
    """LLMProvider that tries multiple underlying providers with circuit breaker,
    backoff, and error classification.

    Args:
        providers:         Ordered list of LLMProvider instances.
        failure_threshold: Consecutive failures before opening circuit.
        recovery_timeout:  Seconds before probing a closed circuit.
        backoff_base:      Base delay for exponential backoff.
        backoff_max:       Maximum delay cap.
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
        self._circuit_breakers: dict[str, CircuitBreaker] = {
            p.name: _cb_initial(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
            for p in providers
        }
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max

    @property
    def chat_model(self):
        """Return a ``MultiProviderChatModel`` for LangChain tool-calling flows."""
        return MultiProviderChatModel(
            providers=self._providers,
            circuit_breakers=self._circuit_breakers,
            backoff_base=self._backoff_base,
            backoff_max=self._backoff_max,
        )

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """Try providers in order with circuit breaker, backoff, and error classification.

        Raises PermanentLLMError on permanent errors, AllProvidersExhaustedError if all fail.
        """
        failed_providers: list[str] = []

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

    def circuit_status(self) -> dict[str, str]:
        """Return current circuit breaker state for all providers."""
        return {
            name: cb.state.value
            for name, cb in self._circuit_breakers.items()
        }
