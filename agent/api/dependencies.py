"""Service dependencies for the RAG API.

All pluggable singletons (including the agent) live in services/container.py.
This file provides FastAPI Depends() wrappers only.
"""

from infrastructure.container import (
    agent,
    decision_tracker,
    feedback_service,
)


def get_agent():
    """Return the pre-wired Agent singleton from the composition root."""
    return agent


def get_feedback_service():
    """Return the FeedbackService singleton from container."""
    return feedback_service


def get_decision_tracker():
    """Return the DecisionTracker singleton from container."""
    return decision_tracker
