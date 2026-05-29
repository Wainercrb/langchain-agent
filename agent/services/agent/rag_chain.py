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

    def invoke(self, **kwargs):
        """Delegate directly to the wrapped RAGChain."""
        return self._chain.invoke(**kwargs)

    def __repr__(self) -> str:
        return f"RAGChainAgent(chain={self._chain!r})"
