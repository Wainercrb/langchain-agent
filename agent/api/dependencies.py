"""Service dependencies for the RAG API.

All pluggable singletons live in services/container.py.
This file assembles the agent from container components.

Strategy Pattern: the logic of which agent is used and which tools are active
lives HERE (in a function with local imports) to avoid circular imports
with rag.* which also import services.container.logger.
"""

from services.container import llm, vector_store, embeddings, logger, feedback_service

# ── Agent Assembly (local imports avoid circular deps) ───────────────


def _build_agent():
    """Build the configured Agent strategy.

    Local imports prevent circular dependencies since rag.* modules
    import services.container.logger.
    """
    from rag.retrieval.retriever import Retriever
    from rag.core.chain import RAGChain
    from services.agent import ToolCallingAgent, RAGChainAgent
    from services.tools import (
        create_search_documents_tool,
        web_search_tool,
    )
    from config import settings

    retriever = Retriever(vector_store=vector_store, embeddings=embeddings)

    if settings.use_tool_agent:
        # Tool-calling agent — container decides which tools are active.
        # To add/remove tools, edit this list. No changes in agent code needed.
        _search_artifact_store = []
        tools = [
            create_search_documents_tool(
                retriever=retriever,
                artifact_store=_search_artifact_store,
                default_latest_only=True,
            ),
            web_search_tool,
        ]
        return ToolCallingAgent(
            llm=llm.chat_model,
            tools=tools,
            artifact_store=_search_artifact_store,
            default_top_k=5,
        )
    else:
        # Legacy RAG chain (always retrieves documents)
        chain = RAGChain(retriever=retriever, llm=llm)
        return RAGChainAgent(chain=chain)


# Singleton agent instance (lazy assembly avoids circular imports at import time)
_agent_instance = None


def get_agent():
    """Return the pre-wired Agent singleton.

    The dependency layer decides the strategy (tool-calling vs legacy)
    based on USE_TOOL_AGENT setting. The route is agnostic.
    """
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = _build_agent()
    return _agent_instance


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
