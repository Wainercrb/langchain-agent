"""Service dependencies for the RAG API.

All pluggable singletons (including the agent) live in services/container.py.
This file provides FastAPI Depends() wrappers only.
"""

from infrastructure.container import (
    agent,
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


async def check_health() -> dict:
    """Perform system health check including DB, LLM, and embedding services."""
    health = {
        "status": "ok",
        "db_connected": False,
        "llm_connected": False,
        "embedding_connected": False,
    }

    # DB check
    try:
        result = await vector_store.health_check()
        health["db_connected"] = bool(result)
    except Exception as e:
        logger.error(f"DB health check failed: {str(e)}")

    # LLM check (lightweight — verify provider is initialized)
    try:
        health["llm_connected"] = llm is not None and hasattr(llm, "_llm")
    except Exception as e:
        logger.error(f"LLM health check failed: {str(e)}")

    # Embedding check
    try:
        health["embedding_connected"] = embeddings is not None
    except Exception as e:
        logger.error(f"Embedding health check failed: {str(e)}")

    # Overall status: error if any critical dependency is down
    if not health["db_connected"]:
        health["status"] = "error"

    return health
