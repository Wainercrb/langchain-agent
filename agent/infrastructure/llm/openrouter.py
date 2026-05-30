"""OpenRouter LLM provider implementation (OpenAI-compatible API)."""

from typing import Dict, List

from langchain_openai import ChatOpenAI

from .base import LLMProvider, LLMResponse
from utils.retry import retry_llm
from infrastructure.logging import logger


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider (OpenAI-compatible gateway)."""

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        api_key: str = None,
        timeout: int = 60,
        **kwargs,
    ):
        super().__init__(model=model, temperature=temperature, **kwargs)
        self._timeout = timeout
        self._llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs,
        )
        logger.info(f"OpenRouter provider initialized: model={self.model}")

    @property
    def chat_model(self):
        """Expose the underlying LangChain chat model for tool calling / agent usage.

        The agent needs the raw ChatOpenAI instance (not the LLMResponse-wrapped
        version) because LangChain's tool calling / bind_tools protocol operates
        on the native model object.
        """
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
                provider="openrouter",
                usage=getattr(response, "usage_metadata", None),
            )
        except Exception as e:
            raise self._classify_error(e, provider="openrouter")
