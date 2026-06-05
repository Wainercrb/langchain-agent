"""Handler routers for all API endpoints."""

from .chat import router as chat_router
from .circuits import router as circuits_router
from .decisions import router as decisions_router
from .feedback import router as feedback_router
from .system import router as system_router

__all__ = [
    "chat_router",
    "circuits_router",
    "decisions_router",
    "feedback_router",
    "system_router",
]
