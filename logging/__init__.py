"""Logging services — Logger ABC + Console/CloudWatch backends + singleton logger.

Uso normal:
    from infrastructure.logging import logger
    logger.info("mensaje")

El backend se configura vía LOGGER_BACKEND en .env:
    - LOGGER_BACKEND=console (default) → logs JSON a stdout (local dev)
    - LOGGER_BACKEND=cloudwatch → logs JSON a AWS CloudWatch Logs (prod)
"""

from config.settings import settings

from .base import Logger
from .console import Console
from .cloudwatch import CloudWatchLogger


def _build_logger() -> Logger:
    if settings.logger_backend == "cloudwatch":
        return CloudWatchLogger()
    return Console()


logger = _build_logger()

__all__ = ["Logger", "Console", "CloudWatchLogger", "logger"]
