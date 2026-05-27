"""Retry and backoff decorators for resilient API calls."""

import logging
import time
from functools import wraps
from typing import Callable

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = 3, base_wait: int = 2):
    """
    Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retries (default: 3)
        base_wait: Base wait time in seconds (default: 2, increases as 2^retry_count)

    Usage:
        @retry_with_backoff(max_retries=3)
        def my_function():
            ...

    Raises:
        Exception: Original exception if all retries exhausted
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt >= max_retries - 1:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {str(e)}"
                        )
                        raise
                    wait_time = base_wait ** attempt
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed, retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)

        return wrapper

    return decorator
