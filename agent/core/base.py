"""Abstract agent interface — strategy pattern for RAG processors.

Swappable implementations:
- ToolCallingAgent: LLM decides which tools to use
- Future: PlanAndExecuteAgent, ReActAgent, etc.

The container (services/container.py) wires the chosen implementation.
Changing agent strategy = changing ONE line in container.py.
"""

from abc import ABC, abstractmethod
from typing import Iterator

from models import ChatResponse


class Agent(ABC):
    """Abstract base for query-processing agents.

    All agents produce a ChatResponse from a natural-language query.
    Implementations decide whether to use tools, retrieval, or direct answers.
    """

    @abstractmethod
    def invoke(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7,
        include_sources: bool = True,
    ) -> ChatResponse:
        """Process a user query and return a response.

        Args:
            query: Natural language question.
            top_k: Number of documents to retrieve when relevant.
            temperature: LLM creativity level.
            include_sources: Whether to include source documents.

        Returns:
            ChatResponse with answer, sources, timing, and model info.
        """
        pass

    @abstractmethod
    def stream(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7,
        include_sources: bool = True,
    ) -> Iterator[dict]:
        """Process a user query and yield streaming events.

        Yields dicts with event data. The last event has ``type: "done"``
        and includes the full ``ChatResponse``-compatible payload.

        Intermediate event types:
            - ``{"type": "token", "content": str}`` — text delta.
            - ``{"type": "tool_call", "tool": str, "args": dict}`` — tool
              invocation request from the LLM.
            - ``{"type": "tool_result", "tool": str, "summary": str}`` —
              tool execution result.

        Args:
            query: Natural language question.
            top_k: Number of documents to retrieve when relevant.
            temperature: LLM creativity level.
            include_sources: Whether to include source documents.

        Yields:
            Streaming event dicts.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
