"""OpenRouter LLM provider implementation (OpenAI-compatible API)."""

import time
from functools import wraps
from typing import Callable, Dict, List

from langchain_openai import ChatOpenAI

from .base import LLMProvider, LLMProviderError, LLMResponse
from services.logging import Console

logger = Console()


def _retry_with_backoff(max_retries: int = 3, base_wait: int = 2):
    """Exponential backoff retry decorator (solo usado por OpenRouterProvider)."""
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


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider (OpenAI-compatible gateway)."""

    def __init__(self, model: str = "openai/gpt-4o", temperature: float = 0.7, max_tokens: int = 4000, api_key: str = None, **kwargs):
        super().__init__(model=model, temperature=temperature, **kwargs)
        self._llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs,
        )
        logger.info(f"OpenRouter provider initialized: model={self.model}")

    @_retry_with_backoff(max_retries=3, base_wait=2)
    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        try:
            response = self._llm.invoke(messages, **kwargs)
            return LLMResponse(
                content=response.content if hasattr(response, "content") else str(response),
                model=self.model,
                provider="openrouter",
                usage=getattr(response, "usage_metadata", None),
            )
        except Exception as e:
            logger.error(f"OpenRouter invoke failed: {str(e)}", exc_info=True)
            raise LLMProviderError(f"OpenRouter invoke failed: {str(e)}", provider="openrouter", original_error=e)
