"""Logger backend factory for switching between logging systems."""

import logging
from typing import Optional

from .base import LoggerBackend
from .json import JSONLoggerBackend

logger = logging.getLogger(__name__)

# Available logger backends
BACKENDS = {
    "json": JSONLoggerBackend,
    "default": JSONLoggerBackend,  # Default to JSON for structured logging
}


def create_logger_backend(backend_type: str = "default") -> LoggerBackend:
    """
    Factory function to create logger backend instances.
    
    Args:
        backend_type: Backend identifier ('json', 'default')
    
    Returns:
        LoggerBackend: Logger backend instance
    
    Raises:
        ValueError: If backend type not found
    
    Example:
        >>> backend = create_logger_backend("json")
        >>> backend.setup_logging(level="INFO")
        >>> log = backend.get_logger(__name__)
    """
    if backend_type.lower() not in BACKENDS:
        available = ", ".join(BACKENDS.keys())
        raise ValueError(f"Unknown logger backend: {backend_type}. Available: {available}")

    backend_class = BACKENDS[backend_type.lower()]
    logger.debug(f"Logger backend created: {backend_class.__name__}")
    return backend_class()


def setup_global_logging(
    backend_type: str = "default",
    level: str = "INFO",
    log_file: Optional[str] = None,
    include_console: bool = True,
) -> LoggerBackend:
    """
    Setup global logging with specified backend.
    
    Args:
        backend_type: Backend identifier ('json', 'default')
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path
        include_console: Whether to log to console
    
    Returns:
        LoggerBackend: Configured backend instance
    
    Example:
        >>> backend = setup_global_logging(backend_type="json", level="DEBUG")
        >>> log = logging.getLogger(__name__)
        >>> log.info("Application started")
    """
    backend = create_logger_backend(backend_type)
    backend.setup_logging(level=level, log_file=log_file, include_console=include_console)
    return backend


__all__ = [
    "LoggerBackend",
    "JSONLoggerBackend",
    "create_logger_backend",
    "setup_global_logging",
]
