"""Logging services — Logger ABC + Console class + singleton logger.

Uso normal:
    from services.logging import logger
    logger.info("mensaje")
"""

from .base import Logger
from .console import Console

logger = Console()

__all__ = ["Logger", "Console", "logger"]
