import logging
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import settings
from utils.logging import setup_logging
from api.routes import router

setup_logging(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LangChain Agent RAG API",
    description="Chat with your stored documents using AI-powered retrieval and generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.include_router(router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )

# TODO: Remove this
@app.on_event("startup")
async def startup_event():
    try:
        from api.dependencies import (
            get_embeddings,
            get_llm,
            get_vector_store,
        )
        get_vector_store()
        get_embeddings()
        get_llm()

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
