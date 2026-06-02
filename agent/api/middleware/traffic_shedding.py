"""Health-based traffic shedding middleware.

When the system is degraded (health check returns non-ok status),
this middleware returns 503 Service Unavailable with a Retry-After header
instead of accepting requests that will fail.

This prevents resource waste and user frustration during outages.
"""

from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

from infrastructure.container import _monitoring_scheduler
from infrastructure.logging import logger


class TrafficSheddingMiddleware:
    """Middleware that sheds traffic when the system is degraded.

    Checks the monitoring scheduler's overall status. If it's "degraded"
    or "error", returns 503 for all non-health-check requests.

    Args:
        app: The ASGI application to wrap.
        shed_on_status: List of health statuses that trigger shedding.
        retry_after_seconds: Seconds to suggest clients wait before retrying.
    """

    def __init__(
        self,
        app,
        shed_on_status: list[str] | None = None,
        retry_after_seconds: int = 60,
    ) -> None:
        self.app = app
        self._shed_on_status = shed_on_status or ["error"]
        self._retry_after_seconds = retry_after_seconds

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Never shed health check or monitoring endpoints
        path = scope.get("path", "")
        if path in ("/v1/health", "/v1/monitoring/status", "/v1/llm/circuits"):
            await self.app(scope, receive, send)
            return

        # Check system health
        overall = _monitoring_scheduler.overall_status
        if overall in self._shed_on_status:
            logger.warning(
                f"Traffic shed: status={overall}, path={path}"
            )
            response = JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "service_unavailable",
                    "message": (
                        "Service is temporarily degraded. "
                        "Please try again later."
                    ),
                    "status": overall,
                },
                headers={
                    "Retry-After": str(self._retry_after_seconds),
                },
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
