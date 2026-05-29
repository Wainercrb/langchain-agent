"""RAG chain agent — wraps the legacy hardcoded RAGChain as an Agent.

This is the "legacy" strategy: always retrieve documents, then call LLM.
Kept for backward compatibility and as a baseline comparison.
"""

from services.agent.base import Agent
from rag.core.chain import RAGChain


class RAGChainAgent(Agent):
    """Adapter that wraps the legacy RAGChain to satisfy the Agent ABC.

    This always runs retrieval, so it costs more for irrelevant queries,
    but it's simpler and more predictable.
    """

    def __init__(self, chain: RAGChain):
        self._chain = chain

    def invoke(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7,
        include_sources: bool = True,
        latest_only: bool = True,
    ):
        """Delegate directly to the wrapped RAGChain.

        Args:
            query: Natural language question.
            top_k: Number of documents to retrieve.
            temperature: LLM creativity level.
            include_sources: Whether to include source documents in response.
            latest_only: Only use latest document versions.
        """
        return self._chain.invoke(
            query=query,
            top_k=top_k,
            temperature=temperature,
            include_sources=include_sources,
            latest_only=latest_only,
        )

    def __repr__(self) -> str:
        return f"RAGChainAgent(chain={self._chain!r})"
