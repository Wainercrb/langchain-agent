"""Google Gemini LLM provider implementation."""

from typing import Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from .base import LLMProvider, LLMResponse
from infrastructure.logging import logger


class GoogleProvider(LLMProvider):
    """Google Gemini LLM provider."""

    name = "google"

    @classmethod
    def from_settings(cls):
        if not settings.google_api_key:
            return None
        return cls(
            model=settings.gemini_model,
            temperature=settings.gemini_temperature,
            max_tokens=settings.gemini_max_tokens,
            api_key=settings.google_api_key,
        )

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        api_key: str = None,
        timeout: int = 60,
        **kwargs,
    ):
        super().__init__(model=model, temperature=temperature, **kwargs)
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=api_key,
            timeout=timeout,
            **kwargs,
        )
        logger.info(f"Google Gemini provider initialized: model={self.model}")

    @property
    def chat_model(self):
        """Expose the underlying LangChain chat model for tool calling."""
        return self._llm

    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        try:
            response = self._llm.invoke(messages, **kwargs)
            return LLMResponse(
                content=(
                    response.content if hasattr(response, "content") else str(response)
                ),
                model=self.model,
                provider=self.name,
                usage=getattr(response, "usage_metadata", None),
            )
        except Exception as e:
            raise self._classify_error(e, provider=self.name)
