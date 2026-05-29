"""Agent implementations — Strategy Pattern.

Swappable query processors:

    from services.agent import ToolCallingAgent, RAGChainAgent

    # Container wires ONE of these
    agent = ToolCallingAgent(llm=llm, tools=[...])   # intelligent tool selection
    # agent = RAGChainAgent(chain=rag_chain)          # legacy always-retrieve

All implement Agent (ABC) from .base.
"""

from .base import Agent
from .tool_calling import ToolCallingAgent
from .rag_chain import RAGChainAgent

__all__ = [
    "Agent",
    "ToolCallingAgent",
    "RAGChainAgent",
]
