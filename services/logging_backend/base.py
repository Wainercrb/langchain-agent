"""Abstract logger backend interface for pluggable logging systems."""

import logging
from abc import ABC, abstractmethod
from typing import Optional


class LoggerBackend(ABC):
    """
    Abstract base class for logger backends.
    
    Enables switching between different logging systems (JSON, Datadog, CloudLogging, etc.)
    while maintaining consistent interface throughout the application.
    """

    @abstractmethod
    def setup_logging(
        self, level: str = "INFO", log_file: Optional[str] = None, include_console: bool = True
    ) -> None:
        """
        Setup logging with this backend.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path to log to
            include_console: Whether to also log to console/stdout
        """
        pass

    @abstractmethod
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance configured with this backend.
        
        Args:
            name: Logger name (typically __name__)
        
        Returns:
            Logger instance
        """
        pass

    @abstractmethod
    def format_log_record(self, record: logging.LogRecord) -> str:
        """
        Format a log record for this backend.
        
        Args:
            record: LogRecord to format
        
        Returns:
            Formatted log string/JSON
        """
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Get human-readable backend name."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
