"""JSON logger backend implementation."""

import json
import logging
import sys
from datetime import datetime
from typing import Optional

from .base import LoggerBackend


class JSONLoggerBackend(LoggerBackend):
    """
    JSON logger backend for structured logging.
    
    Outputs logs as JSON for easy parsing and indexing by log aggregation systems
    (ELK stack, DataDog, Splunk, etc.).
    """

    class JSONFormatter(logging.Formatter):
        """Custom JSON formatter for structured logging."""

        def format(self, record: logging.LogRecord) -> str:
            """Format log record as JSON."""
            log_data = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

            # Add exception info if present
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)

            # Add extra fields if present (request_id, session_id, user_id, etc.)
            for field in ("request_id", "session_id", "user_id", "trace_id"):
                if hasattr(record, field):
                    log_data[field] = getattr(record, field)

            return json.dumps(log_data)

    def setup_logging(
        self, level: str = "INFO", log_file: Optional[str] = None, include_console: bool = True
    ) -> None:
        """Setup JSON logging."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        formatter = self.JSONFormatter()

        # Add console handler
        if include_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        logging.info(f"JSON logging configured: level={level}, file={log_file}, console={include_console}")

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance."""
        return logging.getLogger(name)

    def format_log_record(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        return self.JSONFormatter().format(record)

    def get_backend_name(self) -> str:
        """Return backend name."""
        return "JSON"
