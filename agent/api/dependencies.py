"""Service dependencies for the RAG API.

Todos los singletons pluggeables viven en services/container.py.
"""


from services.container import llm, vector_store, embeddings, logger, feedback_service



def get_retriever():
    """Create a new Retriever instance per request."""
    try:
        from rag.retrieval.retriever import Retriever

        return Retriever(
            vector_store=vector_store,
            embeddings=embeddings,
        )
    except Exception as e:
        logger.error(f"Failed to create Retriever: {str(e)}", exc_info=True)
        raise


def get_rag_chain():
    """Create a new RAGChain instance per request."""
    try:
        from rag.core.chain import RAGChain

        return RAGChain(
            retriever=get_retriever(),
            llm=llm,
        )
    except Exception as e:
        logger.error(f"Failed to create RAGChain: {str(e)}", exc_info=True)
        raise


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
        health["llm_connected"] = llm is not None and hasattr(llm, '_llm')
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
