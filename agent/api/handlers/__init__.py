"""Handler routers for all API endpoints."""

from .circuits import router as circuits_router
from .chat import router as chat_router
from .decisions import router as decisions_router
from .feedback import router as feedback_router
from .health import router as health_router
from .metrics_ep import router as metrics_router
from .monitoring import router as monitoring_router

__all__ = [
    "circuits_router",
    "chat_router",
    "decisions_router",
    "feedback_router",
    "health_router",
    "metrics_router",
    "monitoring_router",
]
