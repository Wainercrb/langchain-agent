"""Logging services — Logger ABC + Console/File backends + singleton logger.

Uso normal:
    from infrastructure.logging import logger
    logger.info("mensaje")

El backend se configura vía LOGGER_BACKEND y LOG_FILE en .env:
    - LOGGER_BACKEND=console (default) → logs JSON a stdout
    - LOGGER_BACKEND=file + LOG_FILE=path → logs JSON a archivo
"""

from config.settings import settings

from .base import Logger
from .console import Console
from .file import FileLogger


def _build_logger() -> Logger:
    if settings.logger_backend == "file" and settings.log_file:
        return FileLogger(settings.log_file)
    return Console()


logger = _build_logger()

__all__ = ["Logger", "Console", "FileLogger", "logger"]
