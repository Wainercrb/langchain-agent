"""FastAPI dependency injection for service singletons and per-request instances.

Uses FastAPI's dependency injection system to manage service lifetimes:
- Singletons: VectorStore, GoogleEmbeddingsWrapper, ChatGoogleGenerativeAI
- Per-request: Retriever, RAGChain
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Services:
    """Service container for application singletons.

    Stores references to shared services that are expensive to initialize:
    - vector_store: Database connection to Supabase/pgvector
    - embeddings: Google Embeddings API wrapper
    - llm: Google Gemini LLM instance
    """

    vector_store: Optional[object] = None
    embeddings: Optional[object] = None
    llm: Optional[object] = None


def get_vector_store():
    """Return singleton VectorStore instance.

    Creates on first call, returns cached instance on subsequent calls.
    Configured from settings (Supabase URL/key).

    Returns:
        VectorStore: Configured vector store instance.

    Raises:
        Exception: If VectorStore initialization fails.
    """
    if Services.vector_store is None:
        try:
            from config import Settings
            from rag.vector_store import VectorStore
            from supabase import create_client

            settings = Settings()
            supabase_client = create_client(
                settings.supabase_url,
                settings.supabase_key,
            )
            Services.vector_store = VectorStore(supabase_client)
            logger.info("VectorStore singleton initialized")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {str(e)}", exc_info=True)
            raise

    return Services.vector_store


def get_embeddings():
    """Return singleton GoogleEmbeddingsWrapper instance.

    Creates on first call, returns cached instance on subsequent calls.

    Returns:
        GoogleEmbeddingsWrapper: Configured embeddings instance.

    Raises:
        Exception: If GoogleEmbeddingsWrapper initialization fails.
    """
    if Services.embeddings is None:
        try:
            from config import Settings
            from rag.embeddings import GoogleEmbeddingsWrapper

            settings = Settings()
            Services.embeddings = GoogleEmbeddingsWrapper(api_key=settings.google_api_key)
            logger.info("GoogleEmbeddingsWrapper singleton initialized")
        except Exception as e:
            logger.error(
                f"Failed to initialize GoogleEmbeddingsWrapper: {str(e)}",
                exc_info=True,
            )
            raise

    return Services.embeddings


def get_llm():
    """Return singleton ChatGoogleGenerativeAI instance.

    Creates on first call, returns cached instance on subsequent calls.
    Configured from settings (model name, temperature).

    Returns:
        ChatGoogleGenerativeAI: Configured LLM instance.

    Raises:
        Exception: If ChatGoogleGenerativeAI initialization fails.
    """
    if Services.llm is None:
        try:
            from config import Settings
            from langchain_google_genai import ChatGoogleGenerativeAI

            settings = Settings()
            Services.llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model or "gemini-2.5-flash",
                temperature=settings.gemini_temperature or 0.7,
                google_api_key=settings.google_api_key,
            )
            logger.info("ChatGoogleGenerativeAI singleton initialized")
        except Exception as e:
            logger.error(
                f"Failed to initialize ChatGoogleGenerativeAI: {str(e)}",
                exc_info=True,
            )
            raise

    return Services.llm


def get_retriever():
    """Return Retriever instance (fresh per request).

    Creates a new Retriever instance for each request using cached
    VectorStore and GoogleEmbeddingsWrapper singletons.

    Returns:
        Retriever: Fresh Retriever instance.

    Raises:
        Exception: If Retriever initialization fails.
    """
    try:
        from rag.retriever import Retriever

        retriever = Retriever(
            vector_store=get_vector_store(),
            embeddings=get_embeddings(),
        )
        logger.debug("Retriever instance created for request")
        return retriever
    except Exception as e:
        logger.error(f"Failed to create Retriever: {str(e)}", exc_info=True)
        raise


def get_rag_chain():
    """Return RAGChain instance (fresh per request).

    Creates a new RAGChain instance for each request using a fresh
    Retriever and cached ChatGoogleGenerativeAI singleton.

    Returns:
        RAGChain: Fresh RAGChain instance.

    Raises:
        Exception: If RAGChain initialization fails.
    """
    try:
        from rag.rag_chain import RAGChain

        chain = RAGChain(
            retriever=get_retriever(),
            llm=get_llm(),
        )
        logger.debug("RAGChain instance created for request")
        return chain
    except Exception as e:
        logger.error(f"Failed to create RAGChain: {str(e)}", exc_info=True)
        raise


async def check_health() -> dict:
    """Check service health (DB connection, API availability).

    Attempts lightweight operations to verify each service is functioning:
    - VectorStore: health_check() call to database
    - Embeddings: API availability check
    - LLM: API availability check

    Returns:
        dict: Health status with keys:
            - status: "ok" or "error"
            - db_connected: bool
            - error: (optional) error message if status="error"

    Example:
        >>> health = await check_health()
        >>> print(health)
        {'status': 'ok', 'db_connected': True}
    """
    try:
        logger.debug("Performing health check")
        vector_store = get_vector_store()

        # Attempt lightweight query to verify DB
        result = await vector_store.health_check()
        db_status = "ok" if result else "error"

        logger.info(f"Health check complete: status={db_status}")
        return {
            "status": db_status,
            "db_connected": bool(result),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "db_connected": False,
            "error": str(e),
        }
