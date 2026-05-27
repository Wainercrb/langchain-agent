"""FastAPI application factory for Retrieval API.

Initializes FastAPI app with routes, middleware, and lifecycle hooks.
Starts on port 8000 with auto-reload in development.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime
import logging

# Configure logging early
from utils.logging import setup_logging
from config import settings

setup_logging(level=settings.log_level)
logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(
    title="LangChain Agent RAG API",
    description="Chat with your stored documents using AI-powered retrieval and generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# Include API router
from api.routes import router

app.include_router(router)


# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler.

    Logs unhandled exceptions and returns structured error response.

    Args:
        request: FastAPI request object.
        exc: Exception instance.

    Returns:
        JSONResponse with error details and timestamp.
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup.

    Warms up singleton services so first request doesn't pay initialization cost:
    - VectorStore: Database connection pool
    - GoogleEmbeddingsWrapper: API client
    - ChatGoogleGenerativeAI: LLM instance

    Errors during startup are fatal and cause app to fail gracefully.
    """
    logger.info("=" * 60)
    logger.info("🚀 Application starting up...")
    logger.info("=" * 60)

    try:
        from api.dependencies import (
            get_vector_store,
            get_embeddings,
            get_llm,
        )

        # Warm up singletons
        logger.info("Initializing VectorStore...")
        get_vector_store()
        logger.info("✅ VectorStore initialized")

        logger.info("Initializing GoogleEmbeddingsWrapper...")
        get_embeddings()
        logger.info("✅ GoogleEmbeddingsWrapper initialized")

        logger.info("Initializing ChatGoogleGenerativeAI...")
        get_llm()
        logger.info("✅ ChatGoogleGenerativeAI initialized")

        logger.info("=" * 60)
        logger.info("✅ All services initialized successfully")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown.

    Called when application is shutting down. Can be used for
    closing database connections, saving state, etc.
    """
    logger.info("=" * 60)
    logger.info("🛑 Application shutting down...")
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )

