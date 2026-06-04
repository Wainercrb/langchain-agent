"""Application lifespan context manager."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from container import _monitoring_scheduler
from loggers import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start monitoring scheduler on startup, stop on shutdown."""
    logger.info("Application starting up")
    try:
        await _monitoring_scheduler.start()
        yield
    finally:
        await _monitoring_scheduler.stop()
        logger.info("Application shutting down")
