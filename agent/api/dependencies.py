"""Service dependencies for the RAG API.

All pluggable singletons (including the agent) live in services/container.py.
This file provides FastAPI Depends() wrappers only.
"""

from config import settings, is_langsmith_enabled
from infrastructure.container import (
    agent,
    decision_tracker,
    embeddings,
    feedback_service,
    llm,
    vector_store,
)
from infrastructure.logging import logger


def get_agent():
    """Return the pre-wired Agent singleton from the composition root."""
    return agent


def get_feedback_service():
    """Return the FeedbackService singleton from container."""
    return feedback_service


def get_decision_tracker():
    """Return the DecisionTracker singleton from container."""
    return decision_tracker


async def check_health() -> dict:
    """Perform system health check including DB, LangSmith, and embedding services.

    Each service is pinged with a minimal payload to verify actual connectivity,
    not just object existence.

    For LangSmith, uses list_projects() — a lightweight API call that verifies
    the tracing API is reachable without invoking an LLM (saves tokens/cost).
    """
    health = {
        "status": "ok",
        "db_connected": False,
        "llm_connected": False,
        "langsmith_connected": False,
        "embedding_connected": False,
    }

    # DB check
    try:
        result = await vector_store.health_check()
        health["db_connected"] = bool(result)
    except Exception as e:
        logger.error(f"DB health check failed: {str(e)}")

    # LLM check — CONFIGURATION only, no real LLM invoke.
    # Reports whether the system is ready to make LLM calls (i.e. at least one
    # provider has an API key). For verification dashboards this avoids the
    # cost of a real invoke while still showing readiness accurately.
    health["llm_connected"] = bool(
        settings.google_api_key
        or settings.openrouter_api_key
        or settings.openai_api_key
    )

    # LangSmith check — lightweight API call, no LLM invocation needed
    if is_langsmith_enabled():
        try:
            from langsmith import Client as LangSmithClient
            client = LangSmithClient()
            # list_projects() is a lightweight API call that verifies connectivity
            list(client.list_projects(limit=1))
            health["langsmith_connected"] = True
        except Exception as e:
            logger.error(f"LangSmith health check failed: {str(e)}")

    # Embedding check — generate embedding for a short string
    try:
        vec = embeddings.embed_query("health")
        health["embedding_connected"] = bool(vec and len(vec) > 0)
    except Exception as e:
        logger.error(f"Embedding health check failed: {str(e)}")

    # Overall status: error if any critical dependency is down
    if not health["db_connected"]:
        health["status"] = "error"

    return health
