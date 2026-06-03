"""Application lifespan context manager."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import settings
from infrastructure.container import _monitoring_scheduler
from infrastructure.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: configure tracing and start monitoring scheduler."""
    from config import configure_tracing

    configure_tracing()
    if settings.monitoring_enabled:
        await _monitoring_scheduler.start()
        logger.info("Monitoring scheduler enabled and started")
    try:
        yield
    finally:
        if settings.monitoring_enabled:
            await _monitoring_scheduler.stop()
            logger.info("Monitoring scheduler stopped")
