"""Agent implementations — Strategy Pattern.

Swappable query processors:

    from infrastructure.agent import ToolCallingAgent, RAGChainAgent

    # Container wires ONE of these
    agent = ToolCallingAgent(llm=llm, tools=[...])   # intelligent tool selection
    # agent = RAGChainAgent(chain=rag_chain)          # legacy always-retrieve

All implement Agent (ABC) from domain.agents.
"""

from domain.agents import Agent
from .tool_calling import ToolCallingAgent
from domain.agents.rag_chain import RAGChainAgent

__all__ = [
    "Agent",
    "ToolCallingAgent",
    "RAGChainAgent",
]
