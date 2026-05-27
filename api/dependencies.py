"""Service dependencies for the RAG API.

Todos los singletons pluggeables viven en services/container.py.
"""


from services.container import llm, vector_store, embeddings, logger



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


async def check_health() -> dict:
    """Perform system health check including database connectivity."""
    try:
        result = await vector_store.health_check()

        return {
            "status": "ok" if result else "error",
            "db_connected": bool(result),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "db_connected": False,
            "error": str(e),
        }
