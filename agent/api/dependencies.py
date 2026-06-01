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
    """Perform system health check including DB, LLM, and embedding services.

    Each service is pinged with a minimal payload to verify actual connectivity,
    not just object existence.
    """
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

    # LLM check — lightweight invoke to verify the API responds
    try:
        response = llm.invoke([{"role": "user", "content": "ping"}])
        health["llm_connected"] = bool(response and response.content)
    except Exception as e:
        logger.error(f"LLM health check failed: {str(e)}")

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
