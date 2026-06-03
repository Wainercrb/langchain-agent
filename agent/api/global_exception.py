"""Global exception handler with alert dispatch."""

from fastapi import Request
from fastapi.responses import JSONResponse

from api.api_errors import internal_error_response
from config.constants import TRUNCATE_ALERT_MESSAGE
from container import alert_service
from logging import logger
from shared.exceptions import Severity, RAGException


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler with alert dispatch.

    Convention:
        - Generic Exception -> CRITICAL (unexpected)
        - TransientLLMError   -> WARNING
        - PermanentLLMError   -> ERROR
        - Other RAGException  -> ERROR
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    if isinstance(exc, RAGException):
        alert_severity = exc.severity
    else:
        alert_severity = Severity.CRITICAL

    await alert_service.send_alert(
        severity=alert_severity,
        message=str(exc)[:TRUNCATE_ALERT_MESSAGE] or "Unhandled server exception",
        error=exc,
        metadata={
            "path": str(request.url.path),
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=500,
        content=internal_error_response("Internal server error"),
    )
