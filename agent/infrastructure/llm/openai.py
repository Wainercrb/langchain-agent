"""OpenAI LLM provider implementation."""

from typing import Dict, List

from langchain_openai import ChatOpenAI

from .base import LLMProvider, LLMResponse
from utils.retry import retry_llm
from infrastructure.logging import logger


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_key: str = None,
        timeout: int = 60,
        **kwargs,
    ):
        super().__init__(model=model, temperature=temperature, **kwargs)
        self._timeout = timeout
        self._llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            **kwargs,
        )
        logger.info(f"OpenAI provider initialized: model={self.model}")

    @property
    def chat_model(self):
        """Expose the underlying LangChain chat model for tool calling / agent usage."""
        return self._llm

    @retry_llm()
    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        try:
            response = self._llm.invoke(messages, **kwargs)
            return LLMResponse(
                content=(
                    response.content if hasattr(response, "content") else str(response)
                ),
                model=self.model,
                provider="openai",
                usage=getattr(response, "usage_metadata", None),
            )
        except Exception as e:
            raise self._classify_error(e, provider="openai")
