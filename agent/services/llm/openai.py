"""OpenAI LLM provider implementation."""

import time
from functools import wraps
from typing import Callable, Dict, List

from langchain_openai import ChatOpenAI

from .base import LLMProvider, LLMProviderError, LLMResponse
from services.logging import Console

logger = Console()


def _retry_with_backoff(max_retries: int = 3, base_wait: int = 2):
    """Exponential backoff retry decorator (solo usado por OpenAIProvider)."""
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


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7, api_key: str = None, **kwargs):
        super().__init__(model=model, temperature=temperature, **kwargs)
        self._llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            **kwargs,
        )
        logger.info(f"OpenAI provider initialized: model={self.model}")

    @_retry_with_backoff(max_retries=3, base_wait=2)
    def invoke(self, messages: List[Dict[str, str]]) -> LLMResponse:
        try:
            response = self._llm.invoke(messages)
            return LLMResponse(
                content=response.content if hasattr(response, "content") else str(response),
                model=self.model,
                provider="openai",
                usage=getattr(response, "usage_metadata", None),
            )
        except Exception as e:
            logger.error(f"OpenAI invoke failed: {str(e)}", exc_info=True)
            raise LLMProviderError(f"OpenAI invoke failed: {str(e)}", provider="openai", original_error=e)
