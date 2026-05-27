"""OpenAI LLM provider implementation (ready for future use)."""

import logging
from typing import Dict, List

from .base import LLMProvider, LLMProviderError, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider implementation.
    
    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
    
    Note: Implementation ready for immediate use when needed.
          Uses LangChain's ChatOpenAI wrapper for consistency.
    """

    def __init__(self, model: str = "gpt-4", temperature: float = 0.7, api_key: str = None, **kwargs):
        """
        Initialize OpenAI provider.
        
        Args:
            model: OpenAI model identifier (default: "gpt-4")
            temperature: Generation temperature 0.0-1.0 (default: 0.7)
            api_key: OpenAI API key (can be None, will use OPENAI_API_KEY env var)
            **kwargs: Additional config passed to ChatOpenAI
        """
        super().__init__(model=model, temperature=temperature, **kwargs)
        self.api_key = api_key
        self._llm = None

    def _get_llm(self):
        """Lazy-load LLM instance."""
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI

                self._llm = ChatOpenAI(
                    model=self.model,
                    temperature=self.temperature,
                    api_key=self.api_key,
                    **self.config,
                )
                logger.info(f"OpenAI provider initialized: model={self.model}")
            except ImportError:
                raise LLMProviderError(
                    "langchain-openai not installed. Install: pip install langchain-openai",
                    provider="openai",
                )
            except Exception as e:
                raise LLMProviderError(f"Failed to initialize OpenAI: {str(e)}", provider="openai", original_error=e)

        return self._llm

    def invoke(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using OpenAI."""
        try:
            llm = self._get_llm()
            response = llm.invoke(messages)

            return LLMResponse(
                content=response.content if hasattr(response, "content") else str(response),
                model=self.model,
                provider="openai",
                usage=getattr(response, "usage_metadata", None),
            )
        except Exception as e:
            logger.error(f"OpenAI invoke failed: {str(e)}", exc_info=True)
            raise LLMProviderError(f"OpenAI invoke failed: {str(e)}", provider="openai", original_error=e)

    def stream(self, messages: List[Dict[str, str]]):
        """Stream response from OpenAI."""
        try:
            llm = self._get_llm()
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"OpenAI stream failed: {str(e)}", exc_info=True)
            raise LLMProviderError(f"OpenAI stream failed: {str(e)}", provider="openai", original_error=e)

    def validate_api_key(self) -> bool:
        """Validate OpenAI API key is available."""
        import os

        return bool(self.api_key or os.getenv("OPENAI_API_KEY"))

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "OpenAI"
