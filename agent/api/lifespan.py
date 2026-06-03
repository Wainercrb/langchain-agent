"""Application lifespan context manager."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from infrastructure.container import _monitoring_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: configure tracing and start monitoring scheduler."""
    from config import configure_tracing

    configure_tracing()
    await _monitoring_scheduler.start()
    try:
        yield
    finally:
        await _monitoring_scheduler.stop()
