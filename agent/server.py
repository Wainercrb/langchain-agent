"""Start the REST API server."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

from api import router
from api.middleware.rate_limit import RateLimitMiddleware
from config import configure_tracing, settings
from infrastructure.container import alert_service
from infrastructure.logging import logger
from utils.correlation import set_correlation_id, get_correlation_id
from utils.exceptions import RAGException, Severity


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: configure tracing on startup."""
    configure_tracing()
    yield


app = FastAPI(
    title="LangChain Agent RAGAPI",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware (applies to /v1/chat and /v1/rag only)
app.add_middleware(RateLimitMiddleware)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Set a correlation ID for each request."""
    cid = request.headers.get("X-Correlation-ID", "")
    set_correlation_id(cid)
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = get_correlation_id()
    return response


app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with alert dispatch.

    Convention:
        - Generic Exception → CRITICAL (unexpected)
        - TransientLLMError   → WARNING
        - PermanentLLMError   → ERROR
        - Other RAGException  → ERROR
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    # Map exception to alert severity
    if isinstance(exc, RAGException):
        alert_severity = exc.severity
    else:
        alert_severity = Severity.CRITICAL

    await alert_service.send_alert(
        severity=alert_severity,
        message=str(exc)[:200] or "Unhandled server exception",
        error=exc,
        metadata={
            "path": str(request.url.path),
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
