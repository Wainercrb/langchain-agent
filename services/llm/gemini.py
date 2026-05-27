"""Gemini LLM provider implementation."""

import logging
from typing import Dict, List, Optional

from rag.utils import retry_with_backoff

from .base import LLMProvider, LLMProviderError, LLMResponse

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider implementation.
    
    Wraps LangChain's ChatGoogleGenerativeAI for seamless integration with Gemini models.
    """

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.7, api_key: str = None, **kwargs):
        """
        Initialize Gemini provider.
        
        Args:
            model: Gemini model identifier (default: "gemini-2.5-flash")
            temperature: Generation temperature 0.0-1.0 (default: 0.7)
            api_key: Google API key (can be None, will use GOOGLE_API_KEY env var)
            **kwargs: Additional config passed to ChatGoogleGenerativeAI
        """
        super().__init__(model=model, temperature=temperature, **kwargs)
        self.api_key = api_key
        self._llm = None

    def _get_llm(self):
        """Lazy-load LLM instance."""
        if self._llm is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self._llm = ChatGoogleGenerativeAI(
                    model=self.model,
                    temperature=self.temperature,
                    google_api_key=self.api_key,
                    **self.config,
                )
                logger.info(f"Gemini provider initialized: model={self.model}")
            except ImportError:
                raise LLMProviderError(
                    "langchain_google_genai not installed. Install: pip install langchain-google-genai",
                    provider="gemini",
                )
            except Exception as e:
                raise LLMProviderError(f"Failed to initialize Gemini: {str(e)}", provider="gemini", original_error=e)

        return self._llm

    @retry_with_backoff(max_retries=3, base_wait=2)
    def invoke(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """
        Generate response using Gemini.
        
        Includes retry logic with exponential backoff for rate limiting.
        """
        try:
            llm = self._get_llm()
            response = llm.invoke(messages)

            return LLMResponse(
                content=response.content if hasattr(response, "content") else str(response),
                model=self.model,
                provider="gemini",
                usage=getattr(response, "usage_metadata", None),
            )
        except Exception as e:
            logger.error(f"Gemini invoke failed: {str(e)}", exc_info=True)
            raise LLMProviderError(f"Gemini invoke failed: {str(e)}", provider="gemini", original_error=e)

    def stream(self, messages: List[Dict[str, str]]):
        """Stream response from Gemini."""
        try:
            llm = self._get_llm()
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Gemini stream failed: {str(e)}", exc_info=True)
            raise LLMProviderError(f"Gemini stream failed: {str(e)}", provider="gemini", original_error=e)

    def validate_api_key(self) -> bool:
        """Validate Gemini API key is available."""
        import os

        return bool(self.api_key or os.getenv("GOOGLE_API_KEY"))

    def get_provider_name(self) -> str:
        """Return provider name."""
        return "Google Gemini"
