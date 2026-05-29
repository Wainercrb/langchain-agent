"""Start the REST API server."""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from api import router
from config import settings
from services.container import alert_service, logger
from utils.correlation import set_correlation_id, get_correlation_id
from utils.exceptions import RAGException, Severity

app = FastAPI(title="LangChain Agent RAG API", version="1.0.0", docs_url="/docs")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://localhost:3000", "http://127.0.0.1:4321", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.get("/test-discord")
async def test_discord():
    """Trigger a test Discord alert to verify the webhook works.

    Returns immediately; check your Discord channel for the alert.
    """
    await alert_service.send_alert(
        severity=Severity.ERROR,
        message="This is a test alert from langchain-agent",
        metadata={"test": True, "source": "manual_test_endpoint"},
    )
    return {"status": "alert_sent", "check_your_discord": True}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, log_level=settings.log_level.lower())
