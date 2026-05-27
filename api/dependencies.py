import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Services:
    """
    Singleton service registry for shared RAG dependencies.

    Manages lazy initialization and caching of expensive resources:
    - VectorStore: Supabase connection with pgvector embeddings
    - GoogleEmbeddingsWrapper: Embedding model for vector generation
    - ChatGoogleGenerativeAI: LLM for response generation

    Pattern: Services are initialized on first request, then reused across
    all subsequent requests to minimize connection overhead and latency.
    """

    vector_store: Optional[object] = None
    embeddings: Optional[object] = None
    llm: Optional[object] = None


def get_vector_store():
    """
    Get or initialize VectorStore singleton.

    Lazily initializes connection to Supabase with pgvector support.
    On first call, creates connection using credentials from config.Settings.

    Returns:
        VectorStore: Connected vector store instance for document retrieval

    Raises:
        Exception: If Supabase connection fails or credentials missing

    Note:
        Connection is cached. Subsequent calls return the same instance.
    """
    if Services.vector_store is None:
        try:
            from supabase import create_client
            from config import Settings
            from rag.vector_store import VectorStore

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
    """
    Get or initialize GoogleEmbeddingsWrapper singleton.

    Lazily initializes Google Generative AI embeddings model.
    On first call, creates wrapper using API key from config.Settings.

    Returns:
        GoogleEmbeddingsWrapper: Embedding model instance for document vectorization

    Raises:
        Exception: If Google API key missing or initialization fails

    Note:
        Wrapper is cached. Subsequent calls return the same instance.
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
    """
    Get or initialize ChatGoogleGenerativeAI singleton.

    Lazily initializes LangChain's Gemini chat model wrapper.
    On first call, creates LLM instance using credentials from config.Settings.

    Returns:
        ChatGoogleGenerativeAI: LLM instance for response generation

    Raises:
        Exception: If Google API key missing or initialization fails

    Note:
        LLM is cached. Subsequent calls return the same instance.
        Default model: "gemini-2.5-flash" (configurable via Settings.gemini_model)
    """
    if Services.llm is None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from config import Settings

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
    """
    Create a new Retriever instance for document retrieval.

    Composes VectorStore and GoogleEmbeddingsWrapper for similarity-based retrieval.
    A new instance is created per request to ensure stateless operation, though
    underlying services (VectorStore, embeddings) are cached singletons.

    Returns:
        Retriever: Document retrieval pipeline instance

    Raises:
        Exception: If VectorStore or embeddings initialization fails

    Note:
        New instance per request ensures thread-safety and request isolation.
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
    """
    Create a new RAGChain instance for query processing.

    Composes Retriever and LLM for full RAG pipeline:
    1. Retrieve relevant documents via similarity search
    2. Construct context-aware prompt
    3. Generate response via LLM

    Returns:
        RAGChain: Complete RAG pipeline instance

    Raises:
        Exception: If Retriever or LLM initialization fails

    Note:
        New instance per request ensures request isolation.
        Underlying singletons (VectorStore, Embeddings, LLM) are reused.
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
    """
    Perform system health check including database connectivity.

    Validates core service availability:
    - VectorStore connection to Supabase
    - Database responsiveness
    - Overall system readiness

    Returns:
        dict containing:
            - status: "ok" if healthy, "error" if degraded
            - db_connected: Boolean database connectivity status
            - error: (optional) Error message if status is "error"

    Note:
        This function gracefully handles errors, returning degraded status
        rather than raising exceptions. Clients can assess partial system health.
    """
    try:
        logger.debug("Performing health check")
        vector_store = get_vector_store()

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
