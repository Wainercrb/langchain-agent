"""Provider-agnostic @trace decorator.

Replaces direct ``from langsmith import traceable`` imports. The decorator
delegates to the configured ObservabilityProvider, so swapping backends
requires zero changes to agent code.

Usage::

    from observability.decorator import trace

    @trace(name="MyAgent.invoke", run_type="chain")
    def invoke(self, query: str) -> ChatResponse:
        ...
"""

import functools
from typing import Any, Callable, TypeVar

from .base import get_observability_provider

T = TypeVar("T")


def trace(name: str, run_type: str = "chain"):
    """Decorate a function to run under the configured observability provider.

    When LangSmith is configured, this wraps the call with @traceable so
    that RunTree context is available inside the function body (enabling
    get_current_run_id(), apply_tags(), apply_metadata()).

    When NoOp is configured, the function is called directly with no overhead.

    Args:
        name: Human-readable name for the trace span.
        run_type: Type of run ("chain", "llm", "tool", etc.).

    Returns:
        A decorator that wraps the target function.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            provider = get_observability_provider()
            return provider.trace_call(fn, name, run_type, args, kwargs)

        return wrapper

    return decorator
