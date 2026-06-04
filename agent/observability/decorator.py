"""Provider-agnostic ``@trace`` decorator.

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

    When no provider is configured the function is called directly.
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            provider = get_observability_provider()
            if provider is None:
                return fn(*args, **kwargs)
            return provider.trace_call(fn, name, run_type, args, kwargs)
        return wrapper
    return decorator
