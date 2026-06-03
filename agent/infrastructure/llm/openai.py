"""OpenAI LLM provider implementation."""

from typing import Dict, List

from langchain_openai import ChatOpenAI

from config import settings
from .base import LLMProvider, LLMResponse
from infrastructure.logging import logger


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    name = "openai"

    @classmethod
    def from_settings(cls):
        if not settings.openai_api_key:
            return None
        return cls(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            api_key=settings.openai_api_key,
        )

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        api_key: str = None,
        timeout: int = 60,
        **kwargs,
    ):
        super().__init__(model=model, temperature=temperature, **kwargs)
        self._llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            timeout=timeout,
            **kwargs,
        )
        logger.info(f"OpenAI provider initialized: model={self.model}")

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
