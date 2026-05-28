"""Shared retry decorator using tenacity — replaces hand-rolled _retry_with_backoff."""

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from utils.exceptions import TransientLLMError, LLMProviderError


def _is_transient(exc: BaseException) -> bool:
    """Retry only transient errors (TransientLLMError or generic non-LLM errors)."""
    if isinstance(exc, TransientLLMError):
        return True
    if isinstance(exc, LLMProviderError) and not exc.is_transient:
        return False
    # Non-LLM errors (network, etc.) are retried by default
    return not isinstance(exc, LLMProviderError)


def retry_llm(max_retries: int = 3, base_wait: int = 2):
    """Tenacity-based retry decorator for LLM providers.

    Retries on TransientLLMError and generic exceptions.
    Does NOT retry PermanentLLMError.
    """
    return retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=base_wait, min=1, max=30),
        retry=retry_if_exception(_is_transient),
        reraise=True,
    )
