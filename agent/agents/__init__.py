"""Agents package — merged from domain/agents, domain/chains, and infrastructure/agent.

Re-exports all agent types for easy import from a single location.
"""

from .base import Agent
from .tool_calling import ToolCallingAgent

__all__ = [
    "Agent",
    "ToolCallingAgent",
]
