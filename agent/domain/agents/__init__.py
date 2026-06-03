"""Agent domain interfaces — strategy pattern for query processors.

The Agent ABC defines the domain contract. Concrete implementations
live in infrastructure (ToolCallingAgent) or alongside domain core (RAGChainAgent).
"""

from .base import Agent
from .rag_chain import RAGChainAgent

__all__ = ["Agent", "RAGChainAgent"]
