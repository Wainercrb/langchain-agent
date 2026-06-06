"""CloudWatch logger — log en formato JSON a AWS CloudWatch Logs.

Uses watchtower (https://github.com/kislyuk/watchtower) to send structured
JSON logs to a CloudWatch log group and stream. Supports boto3 credential
resolution: explicit keys, IAM role, or environment variables.
"""

import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
from watchtower import CloudWatchLogHandler

from config.settings import settings
from .base import Logger
from shared.correlation import get_correlation_id


class CloudWatchLogger(Logger):
    """Sends structured JSON logs to AWS CloudWatch Logs.

    Attributes:
        _log_group: CloudWatch log group name.
        _stream_name: CloudWatch log stream name.
        _py_logger: Underlying Python logger instance.
        _cw_handler: Watchtower handler for CloudWatch delivery.
    """

    _LOG_LEVEL_MAP = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    _DEFAULT_LOG_LEVEL = logging.INFO
    _LOGGER_NAME = "cloudwatch"

    def __init__(
        self,
        log_group: Optional[str] = None,
        stream_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
    ) -> None:
        """Initialize CloudWatch logger with settings or explicit parameters.

        Args:
            log_group: CloudWatch log group (uses settings.cloudwatch_log_group if None).
            stream_name: CloudWatch log stream (uses settings.cloudwatch_stream_name if None).
            aws_access_key_id: Optional explicit AWS access key.
            aws_secret_access_key: Optional explicit AWS secret key.
            aws_region: AWS region (uses settings.aws_region if None).

        Raises:
            ValueError: If AWS credentials are not configured.
        """
        self._log_group = log_group or settings.cloudwatch_log_group
        self._stream_name = stream_name or settings.cloudwatch_stream_name
        self._region = aws_region or settings.aws_region

        # Use provided credentials or fall back to settings
        access_key = aws_access_key_id or settings.aws_access_key_id
        secret_key = aws_secret_access_key or settings.aws_secret_access_key

        self._validate_credentials(access_key, secret_key)

        logs_client = self._create_logs_client(access_key, secret_key)
        self._py_logger = self._setup_logger()
        self._cw_handler = self._setup_handler(logs_client)

    def _validate_credentials(self, access_key: Optional[str], secret_key: Optional[str]) -> None:
        """Validate that AWS credentials are configured.

        Args:
            access_key: AWS access key ID.
            secret_key: AWS secret access key.

        Raises:
            ValueError: If credentials are missing or incomplete.
        """
        if not access_key or not secret_key:
            raise ValueError(
                "AWS credentials not configured. Set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY environment variables, or pass them explicitly."
            )

    def _create_logs_client(
        self,
        access_key: Optional[str],
        secret_key: Optional[str],
    ) -> Any:
        """Create boto3 CloudWatch Logs client with explicit or default credentials.

        Args:
            access_key: Optional AWS access key ID.
            secret_key: Optional AWS secret access key.

        Returns:
            Configured boto3 Logs client.

        Raises:
            RuntimeError: If client creation fails due to invalid credentials.
        """
        try:
            client_kwargs = {"region_name": self._region}
            if access_key and secret_key:
                client_kwargs["aws_access_key_id"] = access_key
                client_kwargs["aws_secret_access_key"] = secret_key
            return boto3.client("logs", **client_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create CloudWatch Logs client: {e}\n"
                f"Region: {self._region}\n"
                f"Verify AWS credentials are valid and have CloudWatch Logs permissions."
            ) from e

    def _setup_logger(self) -> logging.Logger:
        """Create and configure the underlying Python logger.

        Returns:
            Configured Python logger instance.
        """
        logger = logging.getLogger(self._LOGGER_NAME)
        logger.setLevel(logging.DEBUG)
        return logger

    def _setup_handler(self, logs_client: Any) -> CloudWatchLogHandler:
        """Create watchtower handler and attach to logger.

        Args:
            logs_client: boto3 CloudWatch Logs client.

        Returns:
            Configured CloudWatchLogHandler.

        Raises:
            RuntimeError: If handler setup fails (e.g., invalid credentials).
        """
        try:
            handler = CloudWatchLogHandler(
                log_group=self._log_group,
                stream_name=self._stream_name,
                boto3_client=logs_client,
            )
            self._py_logger.addHandler(handler)
            return handler
        except Exception as e:
            raise RuntimeError(
                f"Failed to create CloudWatch handler for log group '{self._log_group}': {e}\n"
                f"Check that AWS credentials are valid and have permission to create logs."
            ) from e

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug-level message."""
        self._dispatch_log("DEBUG", msg, args, kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an info-level message."""
        self._dispatch_log("INFO", msg, args, kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning-level message."""
        self._dispatch_log("WARNING", msg, args, kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log an error-level message."""
        self._dispatch_log("ERROR", msg, args, kwargs)

    def _dispatch_log(
        self, level: str, msg: str, args: tuple, kwargs: dict
    ) -> None:
        """Dispatch a log message to the underlying logger with structured data.

        Args:
            level: Log level as string (DEBUG, INFO, WARNING, ERROR).
            msg: Message template.
            args: Positional arguments for message formatting.
            kwargs: Keyword arguments (exc_info handled specially).
        """
        exc_info = kwargs.pop("exc_info", False)
        structured_data = self._build_structured_log(level, msg, args, kwargs, exc_info)
        log_level = self._LOG_LEVEL_MAP.get(level, self._DEFAULT_LOG_LEVEL)
        self._py_logger.log(log_level, msg, extra=structured_data)

    def _build_structured_log(
        self, level: str, msg: str, args: tuple, extra_kwargs: dict, exc_info: bool
    ) -> dict:
        """Build structured log data with metadata.

        Args:
            level: Log level.
            msg: Message template.
            args: Positional format arguments.
            extra_kwargs: Additional fields to include.
            exc_info: Whether to include exception traceback.

        Returns:
            Structured log dictionary.
        """
        formatted_msg = msg % args if args else msg
        structured_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "correlation_id": get_correlation_id(),
            "text": formatted_msg,
        }
        if extra_kwargs:
            structured_data["extra"] = extra_kwargs
        if exc_info:
            structured_data["stack_trace"] = traceback.format_exc()
        return structured_data

    def __repr__(self) -> str:
        """Return string representation of logger instance."""
        return (
            f"CloudWatchLogger(group={self._log_group!r}, "
            f"stream={self._stream_name!r})"
        )

