"""Middleware configuration and wiring."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from api.middleware.rate_limit import RateLimitMiddleware
from api.middleware.traffic_shedding import TrafficSheddingMiddleware
from config import settings
from utils.correlation import set_correlation_id, get_correlation_id


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Extracts or generates correlation ID from request headers and echoes it in response."""

    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID", "")
        set_correlation_id(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = get_correlation_id()
        return response


def configure_middleware(app: FastAPI) -> None:
    """Add all middleware to the FastAPI application."""
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting (applies to /v1/chat and /v1/rag only)
    app.add_middleware(RateLimitMiddleware)

    # Traffic shedding
    if settings.traffic_shedding_enabled:
        app.add_middleware(
            TrafficSheddingMiddleware,
            shed_on_status=["error"],
            retry_after_seconds=settings.traffic_shedding_retry_after,
        )


def configure_correlation_middleware(app: FastAPI) -> None:
    """Add correlation ID middleware."""
    app.add_middleware(CorrelationIdMiddleware)
